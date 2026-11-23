import json
import unittest
from latent_protocol.context import make_cut,context_hash,receiver_parts,sender_ids


class FakeTokenizer:
    def encode(self,text,**kwargs):return list(text.encode())
    def apply_chat_template(self,messages,**kwargs):
        return json.dumps({'messages':messages,'tools':kwargs.get('tools')},sort_keys=True)


class ContextTests(unittest.TestCase):
    def fixture(self):
        return {'uuid':'trajectory-one','tools':[{'name':'lookup'}], 'messages':[
            {'role':'system','content':'policy'}, {'role':'user','content':'request'},
            {'role':'assistant','content':'checking'}, {'role':'tool','content':'observed'},
            {'role':'assistant','content':'clarify?'}, {'role':'user','content':'target'},
            {'role':'assistant','content':'FUTURE_ANSWER_SECRET'}]}

    def test_cut_excludes_all_future(self):
        row=self.fixture();item=make_cut(row,5)
        self.assertEqual(len(item['history']),5)
        self.assertNotIn('FUTURE_ANSWER_SECRET',json.dumps(item))
        row['messages'][6]['content']='ANOTHER_FUTURE'
        self.assertEqual(item,make_cut(row,5))

    def test_only_target_changes_between_conditions(self):
        item=make_cut(self.fixture(),5);tok=FakeTokenizer()
        before,after=receiver_parts(tok,item)
        a=before+tok.encode(item['prompt'])+after
        b=before+tok.encode('short')+after
        self.assertEqual(a[:len(before)],b[:len(before)])
        self.assertEqual(a[-len(after):],b[-len(after):])
        original=context_hash(item);item['prompt']='short'
        self.assertEqual(original,context_hash(item))
        item['history'][3]['content']='different tool result'
        self.assertNotEqual(original,context_hash(item))

    def test_sender_has_history_but_no_future(self):
        item=make_cut(self.fixture(),5)
        serialized=bytes(sender_ids(FakeTokenizer(),item)).decode()
        self.assertIn('observed',serialized);self.assertIn('target',serialized)
        self.assertNotIn('FUTURE_ANSWER_SECRET',serialized)

    def test_first_user_is_not_an_eligible_cut(self):
        with self.assertRaises(ValueError):make_cut(self.fixture(),1)


if __name__=='__main__':unittest.main()
