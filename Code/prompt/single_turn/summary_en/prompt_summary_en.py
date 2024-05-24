from openai import OpenAI
import os
import json
import textwrap
from tqdm import tqdm

# specify the path to the input and output folders
input_folder = "./summary_en"
output_folder = "./summary_en_re"

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
    self_defined_string = "Please generate 4 question-answer pairs based on the given academic paper"

    system_text1 = "You are an expert in the field of study relevant to the academic paper enclosed within the <doc></doc> tags. Please generate a set of question-answer pairs that reflect the key findings and discussions in the paper."

    system_text3 = textwrap.dedent("""For each question-answer pair, please adhere to the following guidelines:
                                        1. The task specified for the question-answer pair is Summary.
                                        2. Please extract a portion of the paper (at least 200 words) as the reference text for the task.
                                        3. You may refer to the example provided within the <example></example> tags for designing the question-answer pairs, but ensure that the output strictly follows the format of this example.
                                        4. Please replace the part within '()' in the example question with the extracted reference text, and ensure not to change other parts of the question.""")

    system_text4 = "Question: (Folates, also known as vitamin B9, are one of the essential micronutrients for the human body. Due to the inability of the human body to synthesize on its own, it can only rely on food for intake. Adults need to consume at least 400 per day μ Folic acid is used to meet the needs of life activities, increasing to 600 during pregnancy μ G. But in many parts of the world, the daily intake of folic acid is around 400 μ As a result, folate deficiency has become a global health problem. Folate deficiency is more likely to worsen in pregnant and breastfeeding women. As the world's main cultivated crop and the largest grain crop in China, corn has an average folate content of 40-60% in dry seeds μ (g/100g) Only one tenth of the daily dietary requirement for folate in humans.) Read and understand the given text, and summarize it."

    system_text5 = textwrap.dedent("""
                    Answer: This paragraph points out the importance of folic acid in human health and the problem of insufficient folic acid intake worldwide. Special emphasis was placed on the health issues that may arise from folic acid deficiency during pregnancy and lactation. At the same time, it was mentioned that the folate content of corn, the main crop, is generally low, which is different from the daily needs of the human body.""")

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

