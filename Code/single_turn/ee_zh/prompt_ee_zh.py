from openai import OpenAI
import os
import json
import textwrap

# specify the path to the input and output folders
input_folder = "./ee_zh"
output_folder = "./ee_zh_re"

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

    # define the self-defined string
    self_defined_string = "请基于给定学术论文，生成2个问答对"

    system_text1 = "你是一名育种专家，同时是一位农业科学和生物科学的教授，请基于在<doc></doc>标记之间的这篇学术论文，生成若干问答对"
    system_text3 = textwrap.dedent("""对于每一个问答对，你需要注意以下几点：
                                    1.问答所指定的任务为事件提取（Event Extraction）。
                                    2.请截取论文的一部分作为任务的参考文本。
                                    3.任务限制的事件类别为：无事件，病虫害影响，新品种发布或选育，气候或天气事件，种植或生产试验，政策发布，产量新纪录，品种变异。
                                    4.请确保所截取部分仅属于一个事件类别，且与该事件类别高度匹配。
                                    5.你可以参照<example></example>标记之间的示例设计问答，但请确保完全遵循该示例的格式输出回复。
                                    6.请将示例问题部分中'（）'之间的部分替换为所截取的参考文本，确保不要更改问题的其他部分。""")

    system_text4 = "问题：（上周，“金龙1号”品种在菲律宾正式发布。该品种是由国际水稻研究所（IRRI）开发的，经过生物强化以对抗维生素A缺乏症。农民、科学家和政府官员参加了此次发布活动。活动中宣布，“金龙1号”种子将在下一个种植季节分发给当地农民。） 阅读并理解给定文本，识别其中的事件并将其分类。回复应该以字典格式给出，其中键为事件类别，值为含有事件高度概括的列表"
    system_text5 = textwrap.dedent("""
                    回答：{"新品种发布或选育":["国际水稻研究所（IRRI）为对抗维生素A缺乏症而开发的“黄金大米”品种在菲律宾正式发布。种子将在下一个种植季节分发给当地农民。"]""")

    # Format the strings
    system_text = f"{system_text1}\n<doc>\n{system_text2}\n</doc>\n{system_text3}\n<example>\n{system_text4}\n{system_text5}\n</example>"

    # call the OpenAI API
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