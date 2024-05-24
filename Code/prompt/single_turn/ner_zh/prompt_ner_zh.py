from openai import OpenAI
import os
import json
import textwrap

# specify the path to the input and output folders
input_folder = "./ner_zh"
output_folder = "./ner_zh_re"

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
    self_defined_string = "请基于给定学术论文，生成2个问答对"

    system_text1 = "你是一名育种专家，同时是一位农业科学和生物科学的教授，请基于在<doc></doc>标记之间的这篇学术论文，生成5个问答对"
    system_text3 = textwrap.dedent("""对于每一个问答对，你需要注意以下几点：
                                    1.问答所指定的任务为命名实体识别（NER）。
                                    2.请截取论文的一部分作为任务的参考文本。
                                    3.任务限制的实体类别为：地理位置，疾病，作物品种。
                                    4.请确保所截取部分涵盖尽可能多的实体类别。
                                    5.你可以参照<example></example>标记之间的示例设计问答，但请确保完全遵循该示例的格式输出回复。
                                    6.请将示例问题部分中'（）'之间的部分替换为所截取的参考文本，确保不要更改问题的其他部分。""")

    system_text4 = "问题：（“金龙1号”品种已经显示出对“稻瘟病”的抗性，这是水稻作物中的一种常见疾病。“金龙1号”适合在江苏南部种植。然而，农民们报告说，在他们的稻田里，一种害虫“稻象甲”的数量有所增加。） 阅读并理解给定文本，识别其中的实体并将其分类为以下类别：作物品种、地理位置和疾病。回复应该以字典格式给出，其中键为实体类别，值为含有所属该类别实体的列表"
    system_text5 = textwrap.dedent("""
                    回答：{"作物品种": [“金龙1号”], "疾病": [“稻瘟病”], "地理位置": [“江苏南部”]}""")

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