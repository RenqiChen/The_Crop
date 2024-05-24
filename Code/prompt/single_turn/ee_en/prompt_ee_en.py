from openai import OpenAI
import os
import json
import textwrap

# specify the path to the input and output folders
input_folder = "./ee_en"
output_folder = "./ee_en_re"

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
    self_defined_string = "Please generate 2 question-answer pairs based on the given academic paper"

    system_text1 = "You are an expert in the field of study relevant to the academic paper enclosed within the <doc></doc> tags. Please generate a set of question-answer pairs that reflect the key findings and discussions in the paper."

    system_text3 = textwrap.dedent("""For each question-answer pair, please adhere to the following guidelines:
                                        1. The task specified for the question-answer pair is Event Extraction.
                                        2. Please extract a portion of the paper as the reference text for the task.
                                        3. The event categories restricted by the task are: No Event, Pest Impact, New Variety Release or Breeding, Climate or Weather Event, Planting or Production Trial, Policy Release, Yield Record, Variety Mutation.
                                        4. Ensure that the extracted part belongs to only one event category and highly matches that event category.
                                        5. You may refer to the example provided within the <example></example> tags for designing the question-answer pairs, but ensure that the output strictly follows the format of this example.
                                        6. Please replace the part within '()' in the example question with the extracted reference text, and ensure not to change other parts of the question.""")

    system_text4 = "Question: (Last week, the \"Golden Dragon 1\" variety was officially released in the Philippines. This variety was developed by the International Rice Research Institute (IRRI) and was biofortified to combat Vitamin A deficiency. Farmers, scientists, and government officials attended the release event. It was announced at the event that the \"Golden Dragon 1\" seeds will be distributed to local farmers in the next planting season.) Identify the main event type and provide a summary of the event from the given text. The result should be presented in a dictionary format where the key is the event type and the value is a summary of the event."

    system_text5 = textwrap.dedent("""
                    Answer: {\"New Variety Release or Breeding\":[\"The 'Golden Dragon 1' variety, developed by the International Rice Research Institute (IRRI) to combat Vitamin A deficiency, was officially released in the Philippines. The seeds will be distributed to local farmers in the next planting season.\"]}""")

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