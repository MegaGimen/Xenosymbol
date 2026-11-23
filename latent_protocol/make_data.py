"""Small, deterministic pilot corpus. Test items never enter the sender prompt."""
import json
from pathlib import Path


TRAIN = [
 'Please calculate the sum of the numbers 17, 24, and 39. Return only the final numerical result, without showing your working or adding any explanation.',
 'Take the list [9, 3, 7, 3, 1] and sort its elements into ascending order. Keep duplicate values. Respond with the resulting list only.',
 'Translate the English sentence "The window is open" into French. Provide the translation alone, without a preface or commentary.',
 'From the words apple, river, pear, mountain, peach, select only the names of fruits. Keep their original order and separate them with commas. Output nothing else.',
 'Rewrite this sentence in the past tense: "She walks to the station." Preserve the rest of its meaning and return just the rewritten sentence.',
 'Write a Python function named square that accepts one number and returns its square. Include only the function definition, without a markdown code fence or explanation.',
 'Given the string "stressed", reverse the order of its characters. Output the reversed string alone, with no quotation marks.',
 'Convert 2 hours and 15 minutes into a total number of minutes. Answer with the number alone and do not include the unit.',
 'Classify the sentiment of "I absolutely loved the concert" as positive, negative, or neutral. Return exactly one of those labels in lowercase.',
 'Create a JSON object with the key name set to Mira and the key age set to the integer 31. Output valid JSON only, with no surrounding markdown.',
 'Find the largest number in [12, -8, 43, 6, 29]. Reply with that number alone; do not explain how you found it.',
 'Give the chemical symbol for the element oxygen. Return only the symbol, preserving its standard capitalization.',
 'Please remove all vowels from the lowercase word "education". Treat a, e, i, o, and u as vowels. Return only the remaining letters in their original order.',
 'Determine whether the integer 29 is prime. Answer using only yes or no, written in lowercase.',
 'Format the date March 5, 2024 as an ISO date with a four-digit year, two-digit month, and two-digit day, separated by hyphens. Return just the date.',
 'Extract the email address from this text: "Please contact ada@example.org before Friday." Return the address only, without punctuation or commentary.',
 '请将数字列表 [8, 2, 5, 2] 按从小到大的顺序排列，重复的数字必须保留。只输出排序后的列表，不要解释过程。',
 '请把英文单词 "library" 翻译成中文，使用表示借阅书籍场所的常见译法。只输出译文，不添加解释。',
 '请计算长方形的面积，它的长是 12，宽是 7。只输出面积的数值，不写单位和计算过程。',
 '请把字符串 "Hello World" 的所有英文字母转为小写，保留空格。只输出转换后的字符串，不加引号。',
 '请从列表 [4, 9, 12, 15, 18] 中筛选出所有偶数，保持原来的顺序。只输出筛选后的列表。',
 '请用一句不超过二十个汉字的话解释什么是蒸发。直接给出解释，不要添加标题或其他段落。',
 '请输出一个 JSON 对象，键 city 的值为 Beijing，键 count 的值为数字 3。只输出合法 JSON，不使用代码块。',
 '请将这句话改为否定句：“他每天喝咖啡。”保持主语和时间信息不变，只输出改写后的句子。',
]
VALIDATION = [
 'Multiply 13 by 7, then subtract 5 from the product. Respond with only the resulting integer, with no explanation or additional text.',
 'Sort the words zebra, apple, lemon alphabetically, then join them with semicolons and no spaces. Output only that joined string.',
 'Translate "Good evening" into Spanish. Give only the standard translation, without commentary or quotation marks.',
 'Create valid JSON with a boolean key active set to true and an integer key score set to 8. Output only the JSON object, without a code fence.',
 '请从字符串 "a1b2c3" 中提取所有数字，按出现顺序连接。只输出结果，不加引号和说明。',
 '请计算 81 除以 9 的结果，然后加上 4。只输出最终数值，不要写计算过程或单位。',
 'Write a Python function named double that returns its argument multiplied by two. Output just the function definition without markdown or explanation.',
 'From [7, 2, 7, 4, 2], remove repeated values while preserving the order of first appearance. Return the resulting list only.',
]
TEST = [
 'Add 46 to 28, then divide the sum by 2. Give only the final number, without showing intermediate calculations or including any other text.',
 'Take [6, 11, 2, 11, 8], keep only values larger than 5, and sort those values in descending order while retaining duplicates. Return only the resulting list.',
 'Translate the sentence "I need some water" into German. Output the translation only, without a preface, quotation marks, or explanation.',
 'Produce a JSON object containing the key items with the array [2, 4] as its value and the key ready with the boolean false as its value. Output JSON only.',
 '请将字符串 "pYtHoN" 全部转换为大写，然后反转字符顺序。只输出最终字符串，不加引号，不解释。',
 '请将三小时四十分钟转换为总分钟数，再减去二十分钟。只输出最终数字，不要单位或过程。',
 'Write a Python function named is_even that returns whether an integer is divisible by two. Return only the function definition, without a code fence or explanation.',
 'Extract the domain from "research@sample.net" and convert it to uppercase. Output only the domain, excluding the at sign and username.',
]


def main():
    root = Path('data'); root.mkdir(exist_ok=True)
    seen = set()
    for split, texts in [('train', TRAIN), ('validation', VALIDATION), ('test', TEST)]:
        with (root / f'{split}.jsonl').open('w') as f:
            for i, prompt in enumerate(texts):
                assert prompt not in seen
                seen.add(prompt)
                f.write(json.dumps({'id': f'{split}-{i:03d}', 'prompt': prompt}, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    main()
