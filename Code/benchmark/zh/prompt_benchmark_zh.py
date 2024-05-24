# -*- coding: utf-8 -*-

from openai import OpenAI
import os
import json
import textwrap
from tqdm import tqdm

# specify the path to the input and output folders
input_folder = "./benchmark_zh"
output_folder = "./benchmark_zh_re"

# get a list of all files in the input folder
input_files = os.listdir(input_folder)

# set up the OpenAI client
os.environ["OPENAI_API_KEY"] = "sk"
client = OpenAI()

# loop over each file in the input folder
for filename in input_files:
    # construct the full path to the input file
    input_path = os.path.join(input_folder, filename)

    # read the input file
    with open(input_path, 'r', encoding='utf-8') as file:
        system_text2 = file.read()

    # Define the self-defined string
    self_defined_string = "请基于给定学术论文，生成N个问答对。"

    system_text1 = "你是一名育种专家，同时是一位农业科学和生物科学的教授，请基于在<doc></doc>标记之间的这篇学术论文，生成问答对。"
    system_text3 = textwrap.dedent("""对于每一个问答对，你需要注意以下几点：
                            1.问答所指定的任务为多项选择题（Multiple-Choice）。
                            2.问答对的问题部分为生成的问题，回答部分为4个选项和正确答案。
                            3.请确保每个选项的内容差异化很大，但是长度相近。
                            4.请确保生产的问题和答案与农作物种植知识或农作物性质相关。
                            5.你可以参照<example></example>标记之间的示例设计问答，但请确保完全遵循该示例的格式输出回复。
                            6.请将示例回答部分中'[]'之间的部分替换为4个选项，'{}'之间的部分替换为正确答案。""")

    system_text4 = "问题： “白梗一号”水稻的主要农艺特性有哪些？"
    system_text5 = textwrap.dedent("""
                    回答：[选项：A、“白梗一号”水稻的主要农艺特性包括中早熟品种，生育期131天，需活动积温2650℃，平均株高100cm，茎秆坚韧抗倒伏，分力强，株型紧凑，平均穴穗数27.2穗，平均穗粒数116.3粒，结实率90%以上，千粒重24.4g，谷粒长形，长宽比为2.2，无芒，颖壳黄色。
                    B、“白梗一号”水稻是晚熟品种，生育期150天，需活动积温3000℃，平均株高120cm，茎秆脆弱易倒伏，分力弱，株型松散，平均穴穗数20穗，平均穗粒数90粒，结实率80%，千粒重30g，谷粒短圆形，长宽比为1.5，有芒，颖壳绿色。
                    C、“白梗一号”水稻是中晚熟品种，生育期140天，需活动积温2800℃，平均株高110cm，茎秆坚韧抗倒伏，分力中等，株型紧凑，平均穴穗数25穗，平均穗粒数100粒，结实率85%，千粒重26g，谷粒长形，长宽比为2.0，无芒，颖壳红色。
                    D、“白梗一号”水稻是早熟品种，生育期120天，需活动积温2500℃，平均株高90cm，茎秆脆弱易倒伏，分力强，株型松散，平均穴穗数30穗，平均穗粒数130粒，结实率95%，千粒重22g，谷粒短圆形，长宽比为1.8，有芒，颖壳白色。]
                    {正确答案：A。}""")

    # Format the strings
    system_text = f"{system_text1}\n<doc>\n{system_text2}\n</doc>\n{system_text3}\n<example>\n{system_text4}\n{system_text5}\n</example>"

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

    # construct the full path to the output file
    output_path = os.path.join(output_folder, f"output_{filename}")

    # write the response to the output file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(response.choices[0].message.content)