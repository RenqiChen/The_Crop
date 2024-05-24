# -*- coding: utf-8 -*-

from openai import OpenAI
import os
import json
import textwrap
from tqdm import tqdm

# specify the path to the input and output folders
input_folder = "./benchmark_en"
output_folder = "./benchmark_en_re"

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
    self_defined_string = "Please generate N question-answer pairs based on the given academic paper."

    system_text1 = "You are a breeding expert, as well as a professor of agricultural science and biology. Please generate a set of question-answer pairs for this academic paper between the <doc></doc> tags."
    system_text3 = textwrap.dedent("""For each QA pair, you need to consider the following:
                    1.The task specified for the question-answer pair is Multiple-Choice.
                    2.The question part of the question-answer pair should include the generated question. The answer part consists of 4 options and the correct answer.
                    3.Please ensure that each option has significant differences but similar lengths.
                    4.Please ensure that the production of questions and answers is related to the knowledge of crop cultivation or the properties of crops.
                    5.You can refer to the example between the <example></example> tags for the design of QA, but please ensure to strictly follow the format of that example.
                    6.Replace the part within '[]' in the example answer with 4 options, and the part within '{}' with the correct answer.""")

    system_text4 = "Question: What was the experimental design used to study the decomposition of Bt and non-Bt rice residues?"
    system_text5 = textwrap.dedent("""
                    Answer: [Options: A. The experimental design was a completely randomized design with two types of rice residues, KMD (Bt) and Xiushui 11 (non-Bt), cultivated in six plots over two successive years using conventional intensive rice cropping methods.
                    B. The experimental design was a stratified random design with two types of rice residues, KMD (Bt) and Xiushui 11 (non-Bt), cultivated in four plots over three successive years using conventional intensive rice cropping methods.
                    C. The experimental design was a block design with three types of rice residues, including KMD (Bt) and Xiushui 11 (non-Bt), cultivated in eight plots over two successive years using organic rice cropping methods.
                    D. The experimental design was a factorial design with two types of rice residues, KMD (Bt) and Xiushui 11 (non-Bt), cultivated in ten plots over one year using conventional intensive rice cropping methods.]
                    {Correct answer: A.}""")

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