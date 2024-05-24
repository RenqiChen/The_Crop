from openai import OpenAI
import os
import json
import textwrap
from tqdm import tqdm

# specify the path to the input and output folders
input_folder = "./summary_zh"
output_folder = "./summary_zh_re"

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

    system_text1 = "你是一名育种专家，同时是一位农业科学和生物科学的教授，请基于在<doc></doc>标记之间的这篇学术论文，生成概括总结。"
    system_text3 = textwrap.dedent("""对于每一个问答对，你需要注意以下几点：
                                1. 问答对的任务类型是概括总结。
                                2. 请提取文章的一部分（至少200字）作为问答对任务中的参考文本。
                                3. 请确保回答遵循合适的逻辑关系。
                                4. 你可以参照<example></example>标记之间的示例设计问答，但请确保完全遵循该示例的格式输出回复。
                                5.请将示例问题中'（）'内的部分替换为提取的参考文本，并确保不会更改问题的其他部分。""")

    system_text4 = "问题：(玉米生产过程中会出现各种病虫草害的问题，严重影响玉米的生产。通过合理的植保技术手段，可以减少病虫草害的影响，常规的地下害虫和苗期病害以及丝黑穗病都可以用种衣剂解决。玉米苗期低温冷害的问题由于气候多变，春季低温冷害时有发生，严重影响玉米种子正常发芽和出苗，对芽势弱的种子影响更大。解决方案包括适当晚播，选用含精甲霜灵和咯菌腈成份的种衣剂进行种子包衣处理，以及使用“亨达608”对种子进行包衣处理，可直接包衣，农民也可以对已包衣种衣剂的种子进行二次包衣处理。)阅读并理解给定文本，对该文本进行概括总结。"
    system_text5 = textwrap.dedent("""
                    回答：这段文本讨论了玉米生产中遇到的病虫草害问题及其解决方案。强调了通过使用种衣剂和合理的植保技术可以有效减少这些问题的影响。特别提到了春季低温冷害对玉米种子发芽和出苗的不利影响，以及通过晚播和种衣剂处理（包括使用“亨达608”）来提高种子的抗逆性和生长势。""")

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