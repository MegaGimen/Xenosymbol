"""Compare generated tool-call text without executing tools or grading task success."""
import json
import re


def parse_calls(text):
    bodies=re.findall(r'<tool_call>\s*(.*?)\s*</tool_call>',text,re.S)
    if len(bodies)!=text.count('<tool_call>'):return None
    if not bodies:
        return None if '<tool_call>' in text else []
    calls=[]
    for body in bodies:
        if body.startswith('{'):
            try:
                obj=json.loads(body)
                name=obj.get('name',obj.get('function',{}).get('name'))
                arguments=obj.get('arguments',obj.get('function',{}).get('arguments'))
                if isinstance(arguments,str):arguments=json.loads(arguments)
                if not name or not isinstance(arguments,dict):return None
                calls.append({'name':name,'arguments':arguments})
            except (ValueError,TypeError,AttributeError):return None
        else:
            fn=re.fullmatch(r'<function=([^>]+)>\s*(.*?)\s*</function>',body,re.S)
            if not fn:return None
            args={}
            for match in re.finditer(r'<parameter=([^>]+)>\s*(.*?)\s*</parameter>',fn[2],re.S):
                if match[1] in args:return None
                args[match[1]]=match[2].strip()
            remaining=re.sub(r'<parameter=[^>]+>.*?</parameter>','',fn[2],flags=re.S).strip()
            if remaining:return None
            calls.append({'name':fn[1],'arguments':args})
    return calls


def tool_comparison(original,text):
    a,b=parse_calls(original),parse_calls(text)
    return {'original_tool_calls':a,'rewritten_tool_calls':b,
            'tool_call_match':None if a is None or b is None or not (a or b) else a==b}
