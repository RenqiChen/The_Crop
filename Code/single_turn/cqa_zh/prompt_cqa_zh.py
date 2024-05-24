from openai import OpenAI
import os
import json
import textwrap

# specify the path to the input and output folders
input_folder = "./cqa_zh"
output_folder = "./cqa_zh_re"

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
    self_defined_string = "请基于给定学术论文，生成若干问答对"

    system_text1 = "你是一名育种专家，同时是一位农业科学和生物科学的教授，请基于在<doc></doc>标记之间的这篇学术论文，生成若干问答对"
    system_text3 = textwrap.dedent("""对于每一个问答对，你需要注意以下几点：
                                    1.请根据论文长度决定生成问答对的具体数量，问题数量最小为8，最大为10。
                                    2.这些问题应当从不同的角度出发，不应当着重关注于数字部分，避免提出是/否的问题或有明显答案的问题。
                                    3.如果论文中有相关信息，请确保问答中包含涉及选育过程；特征特性；栽培技术要点；适宜种植区域；抗逆性；病虫害防治等方面的内容。
                                    4.请确保回答遵循合适的逻辑关系。
                                    5.你可以参照<example></example>标记之间的示例设计问答，但请确保完全遵循该示例的格式输出回复。""")

    system_text4 = "问题：需要AI Agent推荐一个水稻品种，适合在西藏等高海拔地区种植。"
    system_text5 = textwrap.dedent("""
                    回答：中国高海拔地区通常是指海拔在500-4000米之间的地区，主要包括云南、贵州、四川、西藏等地。按照全国稻区划分，这些地区大多属于西南高原单双季稻稻作区，水稻面积占全国的8%左右。高海拔造成这些区域年积温低和极端低温，光照和紫外线强烈。同时，海拔决定水稻的垂直分布。云南省海拔1750 m以下的为籼稻，海拔1750～2000 m的为籼粳混交带，海拔2000 m以上的为粳稻。2023年西藏自治区农科院农研所选育的水稻新品系“2021LS－68”在海拔3177米实现亩产1112.32斤。此外，其他品种赣73优8601、赣宁粳1号等经过多年测试，能够适应高海拔种植。高海拔地区的气候条件差异可能会很大。因此，要选择适宜种植品种，需要根据具体地区的平均光照温度报告或数据进行决策。""")

    # Format the strings
    system_text = f"{system_text1}\n<doc>\n{system_text2}\n</doc>\n{system_text3}\n<example>\n{system_text4}\n{system_text5}\n</example>"

    response = client.chat.completions.create(
      model="pt-4-turbo-2024-04-09",
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