import unittest
from latent_protocol.tool_comparison import tool_comparison


class ToolComparisonTests(unittest.TestCase):
    def call(self,value):
        return '<tool_call><function=lookup><parameter=id>'+value+'</parameter></function></tool_call>'

    def test_argument_change_is_not_hidden_by_same_tool_name(self):
        self.assertFalse(tool_comparison(self.call('A'),self.call('B'))['tool_call_match'])

    def test_formatting_alone_does_not_change_call(self):
        self.assertTrue(tool_comparison(self.call('A'),self.call('\nA\n'))['tool_call_match'])

    def test_malformed_call_is_unknown_not_success(self):
        self.assertIsNone(tool_comparison(self.call('A'),'<tool_call>broken')['tool_call_match'])
        self.assertIsNone(tool_comparison(self.call('A'),self.call('A')+'<tool_call>broken')['tool_call_match'])

    def test_prose_is_not_a_tool_execution(self):
        self.assertIsNone(tool_comparison('hello','hello')['tool_call_match'])
        self.assertFalse(tool_comparison(self.call('A'),'I will check')['tool_call_match'])


if __name__=='__main__':unittest.main()
