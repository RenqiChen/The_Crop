# -*- coding: utf-8 -*-

from openai import OpenAI
import os
import json
import textwrap
from tqdm import tqdm

# specify the path to the input and output folders
input_folder = "./ner_en"
output_folder = "./ner_en_re"

# get a list of all files in the input folder
input_files = os.listdir(input_folder)

# set up the OpenAI client
os.environ["OPENAI_API_KEY"] = "sk"
client = OpenAI()

# loop over each file in the input folder
for filename in input_files:
    # construct the full path to the input file
    input_path = os.path.join(input_folder, filename)
    output_path = os.path.join(output_folder, f"output_{filename}")
    if os.path.isfile(output_path): continue

    # read the input file
    with open(input_path, 'r', encoding='utf-8') as file:
        system_text2 = file.read()

    # Define the self-defined string
    self_defined_string = "Please generate 6 question-answer pairs based on the given academic paper"

    system_text1 = "You are an expert in the field of study relevant to the academic paper enclosed within the <doc></doc> tags. Please generate a set of question-answer pairs that reflect the key findings and discussions in the paper."

    system_text3 = textwrap.dedent("""For each question-answer pair, please adhere to the following guidelines:
                                        1. The task specified for the question-answer pair is Named Entity Recognition (NER).
                                        2. Please extract a portion of the paper as the reference text for the task.
                                        3. The entity categories restricted by the task are: Geographic Location, Disease, Pest, Crop Variety.
                                        4. Ensure that the extracted part covers as many entity categories as possible.
                                        5. You may refer to the example provided within the <example></example> tags for designing the question-answer pairs, but ensure that the output strictly follows the format of this example.
                                        6. Please replace the part within '()' in the example question with the extracted reference text, and ensure not to change other parts of the question.""")

    system_text4 = "Question: (The \"Golden Dragon 1\" variety has shown resistance to \"rice blast\", a common disease in rice crops. \"Golden Dragon 1\" is suitable for planting in Jiangsu Province. Farmers have reported an increase in the number of rice water weevil in their rice fields.) Read and understand the given text, identify the entities in it and classify them into the following categories: Crop Variety, Geographic Location, and Disease. The reply should be given in dictionary format, where the key is the entity category and the value is a list containing entities belonging to that category."

    system_text5 = textwrap.dedent("""
                        Answer: {\"Crop Variety\": [\"Golden Dragon 1\"], \"Disease\": [\"rice blast\"], \"Geographic Location\": [\"Jiangsu Province\"]}, \"Pest\": [\"rice water weevil\"]}""")

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