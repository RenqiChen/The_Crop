from openai import OpenAI
import os
import json
import textwrap

# specify the path to the input and output folders
input_folder = "./dqa_en"
output_folder = "./dqa_en_re"

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
    self_defined_string = "Please generate 3 question-answer pairs based on the given academic paper"

    system_text1 = "You are an expert in the field of study relevant to the academic paper enclosed within the <doc></doc> tags. Please generate a set of question-answer pairs that reflect the key findings and discussions in the paper."

    system_text3 = textwrap.dedent("""For each question-answer pair, please adhere to the following guidelines:
                                        1. The task specified for the question-answer pair is Direct Question Answering.
                                        2. Please extract a portion of the paper (at least 300 words) as the reference text for the task.
                                        3. The question part of the question-answer pair should include the reference text and the generated question, and the answer part is the matching answer.
                                        4. If the paper provides relevant information, ensure to include the following aspects: Stress Resistance, Pests and Diseases, Cultivation Techniques, Breeding, Characteristics, Suitable Planting Area, etc.
                                        5. You can design the question-answer pair according to the example within the <example></example> tags, but ensure that the output strictly follows the format of this example.
                                        6. Please replace the part within '()' in the example question with the extracted reference text, and the part within '[]' with the generated question.""")

    system_text4 = "Question: (\"Golden Dragon 1\" is a type of rice produced through genetic engineering to biosynthesize beta-carotene, a precursor of vitamin A, in the edible parts of rice. It is intended to produce a fortified food to be grown and consumed in areas with a shortage of dietary vitamin A.) [What is the main purpose of developing Golden Rice?]"

    system_text5 = textwrap.dedent("""
                    Answer: The main purpose of developing \"Golden Dragon 1\" is to produce a fortified food to be grown and consumed in areas with dietary Vitamin A deficiency.""")

    # Format the strings
    system_text = f"{system_text1}\n<doc>\n{system_text2}\n</doc>\n{system_text3}\n<example>\n{system_text4}\n{system_text5}\n</example>"

    # call the OpenAI API
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