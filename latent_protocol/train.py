"""On-policy REINFORCE with a leave-one-out baseline; frozen shared-base receiver.

No learned reward model, SFT filtering, hidden answer or embedding reward is used.
The sampled token distribution is exactly the one differentiated in the update.
"""
import argparse
import contextlib
import hashlib
import json
import random
import shutil
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForImageTextToText, AutoTokenizer, LogitsProcessor
from peft import LoraConfig, get_peft_model, PeftModel

from latent_protocol.core import content_tokens, reward, leave_one_out_advantages, selected_positions
from latent_protocol.context import context_hash, receiver_parts, sender_ids, REWRITE_INSTRUCTION


SENDER_INSTRUCTION = (
    'Rewrite the instruction below into a compact message for another language model. '
    'Preserve its intent, constraints, and supplied data. '
    'You may use abbreviations, symbols, or arbitrary ordinary vocabulary tokens. '
    'Output only the message, without commentary or a solution to the instruction.'
)


def emit(path, record):
    with Path(path).open('a') as f:
        f.write(json.dumps(record, ensure_ascii=False) + '\n')


def read_data(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


class RestrictTokens(LogitsProcessor):
    def __init__(self, excluded): self.excluded = excluded

    def __call__(self, input_ids, scores):
        scores[..., self.excluded] = -float('inf')
        return scores


class Experiment:
    def __init__(self, args):
        self.args = args
        self.root = Path(args.output)
        self.root.mkdir(parents=True, exist_ok=False)
        (self.root / 'states').mkdir()
        shutil.copytree(Path(__file__).parent, self.root / 'source',
                        ignore=shutil.ignore_patterns('__pycache__'))
        self.start = time.time()
        self.serial = 0
        self.current_step = 0
        self.training_order = []
        self.references = {}
        self.parts_cache = {}
        self.receiver_cache = {}
        self.data_dir = Path(getattr(args, 'data_dir', 'data'))
        random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
        self.tok = AutoTokenizer.from_pretrained(args.model)
        base = AutoModelForImageTextToText.from_pretrained(
            args.model, device_map={'': 0}, dtype=torch.bfloat16, attn_implementation='sdpa')
        if args.adapter:
            self.model = PeftModel.from_pretrained(base, args.adapter, is_trainable=True)
        else:
            self.model = get_peft_model(base, LoraConfig(
                r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0,
                target_modules=['q_proj', 'v_proj'], task_type='CAUSAL_LM'))
        # PEFT freezes the base. Do not cast its large embedding/head to float32.
        trainable = [(n, p) for n, p in self.model.named_parameters() if p.requires_grad]
        assert trainable and all('lora_' in n for n, _ in trainable)
        for module in self.model.modules():
            if isinstance(module, torch.nn.Dropout): module.p = 0
        self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        self.model.enable_input_require_grads()
        self.opt = torch.optim.AdamW([p for _, p in trainable], lr=args.lr, weight_decay=0)
        layers = base.model.language_model.layers
        self.layer_index = args.layer or len(layers) // 2
        if not 0 < self.layer_index < len(layers):
            raise ValueError('Choose an intermediate layer, counted from 1')
        self.layer = layers[self.layer_index - 1]
        eos = base.generation_config.eos_token_id
        self.stops = set(eos if isinstance(eos, list) else [eos])
        vocab_size = base.config.text_config.vocab_size
        self.excluded = sorted((set(self.tok.all_special_ids) - self.stops) | set(range(len(self.tok), vocab_size)))
        self.restrict = RestrictTokens(self.excluded)
        marker = 'LATENT_MESSAGE_PLACEHOLDER_73498'
        rendered = self.tok.apply_chat_template([{'role': 'user', 'content': marker}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)
        assert rendered.count(marker) == 1
        left, right = rendered.split(marker)
        self.prefix = self.tok.encode(left, add_special_tokens=False)
        self.suffix = self.tok.encode(right, add_special_tokens=False)
        config = dict(vars(args), layer_counted_from_one=self.layer_index,
            distance='1 - cosine_similarity(float32 states)',
            last_token='last content token, excluding terminal EOS/im_end',
            receiver='same frozen base, LoRA disabled, greedy decoding',
            policy='REINFORCE; independent samples; leave-one-out reward baseline',
            trainable_parameters=sum(p.numel() for _, p in trainable),
            torch_version=torch.__version__,
            model_config=base.config.to_dict(), tokenizer_size=len(self.tok),
            sender_instruction=REWRITE_INSTRUCTION if getattr(args,'contextual',False) else SENDER_INSTRUCTION,
            data_sha256={s: hashlib.sha256((self.data_dir/f'{s}.jsonl').read_bytes()).hexdigest()
                         for s in ['train', 'validation', 'test']})
        (self.root / 'config.json').write_text(json.dumps(config, ensure_ascii=False, indent=2))
        shutil.copytree(self.data_dir, self.root/'data',ignore=shutil.ignore_patterns('raw'))
        self.say('loaded', allocated_gib=torch.cuda.memory_allocated() / 2**30,
                 trainable_parameters=config['trainable_parameters'], layer=self.layer_index)

    def say(self, event, **kwargs):
        entry = {'event': event, 'elapsed_s': round(time.time() - self.start, 2), **kwargs}
        print(json.dumps(entry, ensure_ascii=False), flush=True)
        emit(self.root / 'events.jsonl', entry)

    def tensor(self, ids): return torch.tensor([ids], dtype=torch.long, device='cuda')

    def encode(self, text): return self.tok.encode(text, add_special_tokens=False)

    def sender_prompt(self, text):
        if isinstance(text, dict):
            if getattr(self.args,'contextual',False): return sender_ids(self.tok,text)
            text=text['prompt']
        rendered = self.tok.apply_chat_template([
            {'role': 'system', 'content': SENDER_INSTRUCTION},
            {'role': 'user', 'content': text}], tokenize=False,
            add_generation_prompt=True, enable_thinking=False)
        return self.encode(rendered)

    @torch.no_grad()
    def message(self, text, sample=True):
        self.model.eval()
        prompt = self.sender_prompt(text)
        ids = self.tensor(prompt)
        seq = self.model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
            max_new_tokens=self.args.message_tokens, do_sample=sample, use_cache=True,
            temperature=self.args.temperature if sample else 1.0, top_k=0, top_p=1.0,
            repetition_penalty=1.0, logits_processor=[self.restrict],
            pad_token_id=self.tok.pad_token_id, eos_token_id=sorted(self.stops))
        actions = seq[0, len(prompt):].tolist()
        content, ended = content_tokens(actions, self.stops)
        return {'prompt_ids': prompt, 'actions': actions, 'message_ids': content, 'message_stopped': ended}

    @torch.no_grad()
    def receiver(self, message_ids, item=None, force_recompute=False):
        self.model.eval()
        cache_key=(context_hash(item) if item is not None else None,tuple(message_ids))
        if not force_recompute and cache_key in self.receiver_cache:
            return dict(self.receiver_cache[cache_key],cache_hit=True)
        left,right=self.prefix,self.suffix
        if item is not None and getattr(self.args,'contextual',False):
            key=context_hash(item)
            if key not in self.parts_cache:self.parts_cache[key]=receiver_parts(self.tok,item)
            left,right=self.parts_cache[key]
        prompt = left + message_ids + right
        ids = self.tensor(prompt)
        with self.model.disable_adapter():
            seq = self.model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                max_new_tokens=self.args.response_tokens, do_sample=False, use_cache=True,
                top_k=0, top_p=1.0, temperature=1.0,
                pad_token_id=self.tok.pad_token_id, eos_token_id=sorted(self.stops))
            generated = seq[0, len(prompt):].tolist()
            content, ended = content_tokens(generated, self.stops)
            if not content:
                raise RuntimeError('Receiver produced no content: there is no content-last state.')
            positions = selected_positions(len(prompt), len(content))
            states = {}
            def capture(module, inputs, output):
                h = output[0] if isinstance(output, tuple) else output
                states['pre'] = h[0, positions[0]].detach().float().cpu().clone()
                states['last'] = h[0, positions[1]].detach().float().cpu().clone()
            handle = self.layer.register_forward_hook(capture)
            try:
                # Full replay explicitly processes the last generated CONTENT token.
                full = self.tensor(prompt + content)
                self.model(input_ids=full, attention_mask=torch.ones_like(full),
                           use_cache=False, logits_to_keep=1)
            finally:
                handle.remove()
            if not all(torch.isfinite(states[k]).all() for k in ['pre','last']):
                raise RuntimeError('Non-finite receiver state; refusing to turn NaN distance into reward')
        result = {'message_ids': message_ids, 'receiver_prompt_ids': prompt,
                  'cache_hit':False,
                  'shared_prefix_ids':left,'shared_suffix_ids':right,
                  'context_sha256':context_hash(item) if item is not None else None,
                  'generated_ids': generated, 'content_ids': content,
                  'text': self.tok.decode(content, skip_special_tokens=False),
                  'complete': ended, 'positions': positions, **states}
        if not ended:
            # Preserve failed evidence. Do not silently optimize a truncated-response proxy.
            torch.save(result, self.root / 'truncated_receiver.pt')
            raise RuntimeError('Receiver hit response cap. Evidence saved; increase --response-tokens.')
        self.receiver_cache[cache_key]=dict(result)
        return result

    def reference(self, item):
        key = item['id']
        if key not in self.references:
            obs = self.receiver(self.encode(item['prompt']),item)
            obs['artifact'] = f'states/reference-{key}.pt'
            torch.save(obs, self.root / obs['artifact'])
            self.references[key] = obs
            emit(self.root / 'references.jsonl', {'id': key, 'prompt': item['prompt'],
                'text': obs['text'], 'artifact': obs['artifact'],
                'message_length': len(obs['message_ids']), 'complete': obs['complete']})
        return self.references[key]

    def score(self, item, message, phase, step, force_receiver=False):
        ref = self.reference(item)
        obs = self.receiver(message['message_ids'],item,force_recompute=force_receiver)
        assert ref['shared_prefix_ids']==obs['shared_prefix_ids'] and ref['shared_suffix_ids']==obs['shared_suffix_ids']
        self.serial += 1
        artifact = f'states/{self.serial:06d}.pt'
        torch.save(obs, self.root / artifact)
        distance = max(0., float(1 - F.cosine_similarity(ref['last'], obs['last'], dim=0)))
        pre_distance = max(0., float(1 - F.cosine_similarity(ref['pre'], obs['pre'], dim=0)))
        length = len(message['message_ids'])
        record = {'id': item['id'], 'phase': phase, 'step': step,
            'trajectory_id':item.get('trajectory_id'),'target_message_index':item.get('target_message_index'),
            'context_sha256':obs['context_sha256'],
            'shared_context_and_wrapper_tokens':len(obs['shared_prefix_ids'])+len(obs['shared_suffix_ids']),
            'receiver_cache_hit':obs['cache_hit'],
            'prompt': item['prompt'], 'message_ids': message['message_ids'],
            'sender_action_ids': message.get('actions'),
            'message': self.tok.decode(message['message_ids'], skip_special_tokens=False),
            'message_stopped': message.get('message_stopped'),
            'original_text': ref['text'], 'text': obs['text'],
            'original_length': len(ref['message_ids']), 'length': length,
            'length_ratio': length / len(ref['message_ids']),
            'distance': distance, 'pre_distance': pre_distance,
            'reward': reward(distance, length, self.args.length_weight),
            'exact_output_match': obs['text'] == ref['text'],
            'complete': obs['complete'], 'artifact': artifact, 'reference_artifact': ref['artifact']}
        emit(self.root / 'observations.jsonl', record)
        return record

    def update(self, messages, rewards):
        advantages = leave_one_out_advantages(rewards)
        self.model.train()
        self.opt.zero_grad(set_to_none=True)
        losses = []
        for message, advantage in zip(messages, advantages):
            actions = message['actions']
            ids = self.tensor(message['prompt_ids'] + actions)
            # K+1 logits includes the prediction of the first action; drop final prediction.
            logits = self.model(input_ids=ids, attention_mask=torch.ones_like(ids),
                use_cache=False, logits_to_keep=len(actions) + 1).logits[0, :-1].float()
            logits = logits / self.args.temperature
            logits[:, self.excluded] = -float('inf')
            targets = torch.tensor(actions, device=logits.device)
            logp = -F.cross_entropy(logits, targets, reduction='sum')
            loss = -advantage * logp / len(messages)
            loss.backward()
            losses.append(float(loss.detach()))
            del logits, loss, logp
        params = [p for p in self.model.parameters() if p.requires_grad]
        norm = torch.nn.utils.clip_grad_norm_(params, 1.0)
        if not torch.isfinite(norm): raise RuntimeError('Non-finite gradient')
        self.opt.step(); self.opt.zero_grad(set_to_none=True)
        return {'policy_loss': sum(losses), 'grad_norm': float(norm),
                'advantage_range': max(advantages) - min(advantages)}

    def evaluate(self, items, phase, step):
        records = []
        for item in items:
            records.append(self.score(item, self.message(item, sample=False), phase, step))
        result = {key: sum(r[key] for r in records) / len(records)
                  for key in ['reward', 'distance', 'pre_distance', 'length', 'length_ratio', 'exact_output_match']}
        self.say('evaluation', phase=phase, step=step, count=len(records), **result)
        return result

    def save(self, name):
        path = self.root / name
        self.model.save_pretrained(path)
        self.tok.save_pretrained(path)
        torch.save({'step': self.current_step, 'optimizer': self.opt.state_dict(),
                    'python_rng': random.getstate(), 'torch_rng': torch.get_rng_state(),
                    'cuda_rng': torch.cuda.get_rng_state_all(),
                    'training_order': self.training_order}, path / 'training_state.pt')

    @torch.no_grad()
    def check_pre_state(self, ref):
        """Check that the stored pre-state is actually the prefix-only state."""
        captured = {}
        def hook(module, inputs, output):
            h = output[0] if isinstance(output, tuple) else output
            captured['pre'] = h[0, -1].float().cpu().clone()
        self.model.eval()
        handle = self.layer.register_forward_hook(hook)
        try:
            with self.model.disable_adapter():
                ids = self.tensor(ref['receiver_prompt_ids'])
                self.model(input_ids=ids, attention_mask=torch.ones_like(ids),
                           use_cache=False, logits_to_keep=1)
        finally:
            handle.remove()
        distance = max(0., float(1 - F.cosine_similarity(captured['pre'], ref['pre'], dim=0)))
        self.say('pre_state_causality', cosine_distance=distance)
        if distance > 1e-3: raise RuntimeError('Stored pre-state failed prefix-only check')

    def run(self):
        train = read_data(self.data_dir/'train.jsonl')
        val = read_data(self.data_dir/'validation.jsonl')[:self.args.validation_items]
        test = read_data(self.data_dir/'test.jsonl')
        # Real model invariant: identical messages reproduce identical states and output.
        ref = self.reference(train[0])
        self.check_pre_state(ref)
        identity = self.score(train[0], {'message_ids': self.encode(train[0]['prompt'])}, 'sanity_identity', 0,force_receiver=True)
        assert not identity['receiver_cache_hit']
        assert identity['distance'] < 1e-5 and identity['exact_output_match']
        baseline = self.evaluate(val, 'validation', 0)
        if getattr(self.args,'contextual',False):
            for i,item in enumerate(val):
                self.score(item,{'message_ids':[]},'control_empty',0)
                self.score(item,{'message_ids':self.encode(val[(i+1)%len(val)]['prompt'])},'control_mismatched',0)
        self.save('initial')
        best = baseline['reward']; best_step = 0
        self.save('best')
        order = list(train); random.shuffle(order)
        for step in range(1, self.args.steps + 1):
            if step > 1 and (step - 1) % len(order) == 0: random.shuffle(order)
            item = order[(step - 1) % len(order)]
            self.training_order = [x['id'] for x in order]
            messages = [self.message(item) for _ in range(self.args.group_size)]
            records = [self.score(item, m, 'train', step) for m in messages]
            stats = self.update(messages, [r['reward'] for r in records])
            self.current_step = step
            # A failed later rollout must not discard already completed updates.
            self.save('latest')
            self.say('update', step=step, id=item['id'],
                reward=sum(r['reward'] for r in records)/len(records),
                distance=sum(r['distance'] for r in records)/len(records),
                length=sum(r['length'] for r in records)/len(records),
                peak_gib=torch.cuda.max_memory_allocated()/2**30, **stats)
            if step % self.args.eval_every == 0 or step == self.args.steps:
                result = self.evaluate(val, 'validation', step)
                self.save(f'checkpoint-{step:04d}')
                if result['reward'] > best:
                    best = result['reward']; best_step = step; self.save('best')
        check = self.receiver(self.encode(train[0]['prompt']),train[0],force_recompute=True)
        frozen_error = float((check['last'] - ref['last']).abs().max())
        if check['text'] != ref['text'] or frozen_error > 1e-5:
            raise RuntimeError(f'Frozen receiver invariant failed: {frozen_error}')
        self.say('receiver_invariant', max_abs_error=frozen_error, same_text=True)
        summary = {'baseline_validation': baseline, 'best_validation_reward': best,
                   'best_step': best_step, 'receiver_frozen_max_abs_error': frozen_error}
        if self.args.final_test:
            # All tuning ends before this branch; test data never select a checkpoint.
            self.model.load_adapter(self.root/'initial', adapter_name='initial_eval')
            self.model.set_adapter('initial_eval')
            summary['test_initial'] = self.evaluate(test, 'test_initial', 0)
            self.model.load_adapter(self.root/'best', adapter_name='best_eval')
            self.model.set_adapter('best_eval')
            summary['test_best'] = self.evaluate(test, 'test_best', best_step)
        (self.root / 'summary.json').write_text(json.dumps(summary, indent=2))
        self.say('finished', **summary)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='/workspace/model_nf4_native')
    p.add_argument('--output', required=True)
    p.add_argument('--adapter')
    p.add_argument('--data-dir',default='data')
    p.add_argument('--contextual',action='store_true')
    p.add_argument('--steps', type=int, default=24)
    p.add_argument('--group-size', type=int, default=4)
    p.add_argument('--eval-every', type=int, default=8)
    p.add_argument('--validation-items', type=int, default=8)
    p.add_argument('--message-tokens', type=int, default=48)
    p.add_argument('--response-tokens', type=int, default=1024)
    p.add_argument('--length-weight', type=float, default=.005)
    p.add_argument('--lr', type=float, default=3e-5)
    p.add_argument('--temperature', type=float, default=.85)
    p.add_argument('--rank', type=int, default=8)
    p.add_argument('--layer', type=int)
    p.add_argument('--seed', type=int, default=73498)
    p.add_argument('--final-test', action='store_true')
    args = p.parse_args()
    if args.group_size < 2 or args.length_weight < 0 or args.temperature <= 0:
        p.error('Invalid group size, length weight or temperature')
    Experiment(args).run()


if __name__ == '__main__': main()
