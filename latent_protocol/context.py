"""Context-bound message rewriting; no future trajectory is included."""
import copy
import hashlib
import json

REWRITE_INSTRUCTION = (
    'You are a message encoder, not the agent executing the conversation. '
    'The supplied JSON contains a conversation history, available tools, and one target user message. '
    'Rewrite ONLY the target user message into a short message for the receiving agent. '
    'The receiving agent already has exactly the supplied history and tool definitions. '
    'Preserve the target intent and constraints in that shared context. '
    "Keep the user's speaker perspective and speech act; do not speak as the assistant. "
    'Do not answer the user, call tools, summarize the whole history, or add commentary. '
    'Output only the replacement message. Ordinary vocabulary tokens, abbreviations and symbols are allowed.'
)


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def context_hash(item):
    return hashlib.sha256(canonical({'history':item.get('history',[]), 'tools':item.get('tools',[])}).encode()).hexdigest()


def normalize_messages(messages):
    result=copy.deepcopy(messages)
    for m in result:
        for call in m.get('tool_calls',[]) or []:
            args=call.get('function',{}).get('arguments')
            if isinstance(args,str): call['function']['arguments']=json.loads(args)
    return result


def make_cut(row, index):
    messages=row['messages']
    if messages[index]['role']!='user' or not any(m['role']=='user' for m in messages[:index]):
        raise ValueError('Choose a non-initial user message')
    if index+1>=len(messages) or messages[index+1]['role']!='assistant':
        raise ValueError('Target must have an assistant continuation')
    history=normalize_messages(messages[:index])
    return {'id':row['uuid']+f'-turn-{index}', 'trajectory_id':row['uuid'],
            'target_message_index':index,'history':history,'tools':row.get('tools',[]),
            'prompt':messages[index]['content'], 'license':row.get('license','cc-by-4.0')}


def receiver_parts(tokenizer, item):
    marker='LATENT_REPLACEMENT_2f2d9a9be61f'
    assert marker not in canonical(item)
    messages=item.get('history',[])+[{'role':'user','content':marker}]
    rendered=tokenizer.apply_chat_template(messages,tools=item.get('tools') or None,
        tokenize=False,add_generation_prompt=True,enable_thinking=False)
    if rendered.count(marker)!=1: raise ValueError('Ambiguous replacement boundary')
    left,right=rendered.split(marker)
    return tokenizer.encode(left,add_special_tokens=False),tokenizer.encode(right,add_special_tokens=False)


def sender_ids(tokenizer, item):
    payload={'history':item.get('history',[]),'tools':item.get('tools',[]),'target_user_message':item['prompt']}
    rendered=tokenizer.apply_chat_template([
        {'role':'system','content':REWRITE_INSTRUCTION},
        {'role':'user','content':'Encode the target user message in this record:\n<conversation_record>\n'
         +canonical(payload)+'\n</conversation_record>\n'
         +"Output only a shorter replacement USER message, in the user's voice. Do not answer it."}],tokenize=False,
        add_generation_prompt=True,enable_thinking=False)
    return tokenizer.encode(rendered,add_special_tokens=False)
