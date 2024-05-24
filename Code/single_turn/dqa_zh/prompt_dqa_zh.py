from openai import OpenAI
import os
import json
import textwrap

# specify the path to the input and output folders
input_folder = "./dqa_zh"
output_folder = "./dqa_zh_re"

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
    self_defined_string = "请基于给定学术论文，生成3个问答对"

    system_text1 = "你是一名育种专家，同时是一位农业科学和生物科学的教授，请基于在<doc></doc>标记之间的这篇学术论文，生成3个问答对"
    system_text3 = textwrap.dedent("""对于每一个问答对，你需要注意以下几点：
                                    1.问答所指定的任务为直接问答（Direct Question Answering）。
                                    2.请截取论文的一部分（至少200字）作为任务的参考文本。
                                    3.问答对的问题部分包括参考文本和生成的问题，回答部分为匹配的回答。
                                    4.如果论文中有相关信息，请包含涉及选育过程；特征特性；栽培技术要点；适宜种植区域；病虫害防治；抗逆性等方面的问题。
                                    5.你可以参照<example></example>标记之间的示例设计问答，但请确保完全遵循该示例的格式输出回复。
                                    6.请将示例问题部分中'（）'之间的部分替换为所截取的参考文本，'[]'之间的部分替换为生成的问题。""")

    system_text4 = "问题：（“金龙1号”是一种水稻，通过基因工程在大米的可食用部分生物合成维生素a的前体β-胡萝卜素。它旨在生产一种强化食品，在膳食维生素a短缺的地区种植和食用。） [开发黄金大米的主要目的是什么？]"
    system_text5 = textwrap.dedent("""
                    回答：开发“金龙1号”的主要目的是生产一种强化食品，在膳食维生素a短缺的地区种植和食用。""")

    # Format the strings
    system_text = f"{system_text1}\n<doc>\n{system_text2}\n</doc>\n{system_text3}\n<example>\n{system_text4}\n{system_text5}\n</example>"

    response = client.chat.completions.create(
      model="gpt-4-turbo",
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