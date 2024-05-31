from openai import OpenAI
import os
import json
import textwrap

# specify the path to the input and output folders
input_folder = "./cqa_en"
output_folder = "./cqa_en_re"

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

    # define the self-defined string
    self_defined_string = "Please generate a set of question-answer pairs based on the given academic paper"

    system_text1 = "You are an expert in the field of study relevant to the academic paper enclosed within the <doc></doc> tags. Please generate a set of question-answer pairs that reflect the key findings and discussions in the paper."

    system_text3 = textwrap.dedent("""For each question-answer pair, please adhere to the following guidelines:
                                        1. The number of question-answer pairs should be determined by the depth and breadth of the paper, with a minimum of 3 and a maximum of 5.
                                        2. The questions should cover various aspects of the paper, avoid focusing solely on numerical data, and refrain from asking yes/no questions or questions with obvious answers.
                                        3. If the paper provides relevant information, ensure that the question-answer pairs cover the following aspects: Stress Resistance, Pests and Diseases, Cultivation Techniques, Breeding, Characteristics, Suitable Planting Area, etc.
                                        4. Ensure that the answers follow a logical sequence and are supported by the information in the paper.
                                        5. You may refer to the example provided within the <example></example> tags for designing the question-answer pairs, but ensure that the output strictly follows the format of this example.""")

    system_text4 = "Question: How to achieve high yield in the cultivation of Y302 through fertilizer management?"

    system_text5 = textwrap.dedent("""
                    Answer: In the cultivation of Y302, high yield is achieved through the scientific ratio and timely application of base fertilizer, topdressing, ear fertilizer, and grain fertilizer. Considering the longer growth time and high yield potential of Y302, organic fertilizer needs to be added.""")
    # Format the strings
    system_text = f"{system_text1}\n<doc>\n{system_text2}\n</doc>\n{system_text3}\n<example>\n{system_text4}\n{system_text5}\n</example>"

    # call the OpenAI API
    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": system_text},
            {"role": "user", "content": self_defined_string}
        ],
        temperature=0.1,
        max_tokens=4096,
        top_p=0.9
    )

    # construct the full path to the output file
    output_path = os.path.join(output_folder, f"output_{filename}")

    # write the response to the output file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(response.choices[0].message.content)