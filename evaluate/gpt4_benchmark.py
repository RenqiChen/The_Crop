# -*- coding: utf-8 -*-

from openai import OpenAI
import os
import json
import textwrap
from tqdm import tqdm

def create_message(file_path):
    # Read the txt file into a variable with 'utf-8' encoding
    with open(file_path, 'r', encoding='utf-8') as file:
        system_text2 = file.read()

    # Define the self-defined string
    self_defined_string = "请选出正确选项"

    system_text1 = "你是一名育种专家，同时是一位农业科学和生物科学的教授，请给出<doc></doc>标记之间的问答的正确选项"
    system_text3 = textwrap.dedent("""对于每一个问答对，你需要注意以下几点：
                                    1.你需要参照<example></example>标记之间的示例回答，请确保完全遵循该示例的格式输出回复。
                                    2.请将示例回答部分中'{}'之间的部分替换为正确答案。""")

    system_text4 = ""
    system_text5 = textwrap.dedent("""
                    {正确答案：E。}""")

    # Format the strings
    system_text = f"{system_text1}\n<doc>\n{system_text2}\n</doc>\n{system_text3}\n<example>\n{system_text4}\n{system_text5}\n</example>"

    os.environ["OPENAI_API_KEY"] = "sk"
    client = OpenAI()
    response = client.chat.completions.create(
    model="gpt-4-turbo-2024-04-09",
    messages=[
        {"role": "system", "content": system_text},
        {"role": "user", "content": self_defined_string}
    ],
    temperature=0,
    max_tokens=4096,
    top_p=1
    )

    return response.choices[0].message.content

# 指定原始文件夹和保存文件夹路径
original_folder = './问题集'
save_folder = './gpt4结果'

# 确保保存文件夹存在
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

# 遍历原始文件夹中的所有文件
for filename in tqdm(os.listdir(original_folder)):
    # 拼接文件的完整路径
    file_path = os.path.join(original_folder, filename)
    new_file_path = os.path.join(save_folder, filename)
    if os.path.exists(new_file_path):
        print('exists')
        continue
    # 判断文件是否是txt文件
    if os.path.isfile(file_path) and filename.endswith('.txt'):
        # 打开原始文件进行读取
        with open(file_path, 'r', encoding='utf-8') as original_file:
            content_origin = original_file.read()
        content = create_message(file_path)
        # 将文件内容写入新的文件中
        with open(new_file_path, 'w', encoding='utf-8') as new_file:
            new_file.write(content)