import requests
import os
import json
import textwrap
import time

url = "https://oneapi.run.place/v1/chat/completions"
header = {
    'Authorization': 'Bearer sk',
    'Content-Type': 'application/json'
}

# specify the path to the input and output folders
input_folder = "./benchmark_new/ques"
output_folder = "./benchmark_new/ans_output"

# get a list of all files in the input folder
input_files = os.listdir(input_folder)

# loop over each file in the input folder
for filename in input_files:
    # construct the full path to the input file
    input_path = os.path.join(input_folder, filename)
    output_path = os.path.join(output_folder, f"output_{filename}")
    if os.path.isfile(output_path): continue

    with open(input_path, 'r', encoding='utf-8') as file:
        system_text2 = file.read()

    # Define the self-defined string

    system_text1 = "Please select the most suitable choice based on your knowledge for the question enclosed within the <doc></doc> tags."
    system_text3 = textwrap.dedent("""Please adhere to the following guidelines:
                                    1.Response with your selected choice only without any explanation.
                                    2.Make sure the output strictly follows the format of the example provided within the <example></example> tags.
                                    3.Replace the part within '()' in the example with the code of the option you select, and ensure not to change other parts of the example.""")

    system_text4 = textwrap.dedent(""" Answer: (E) """)

    # Format the strings
    system_text = f"{system_text1}\n<doc>\n{system_text2}\n</doc>\n{system_text3}\n<example>\n{system_text4}\n</example>"

    data = {
        "model": "claude-3-opus-20240229",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": system_text
                    }
                ]
            }
        ],
        "stream": False
    }
    resp = requests.post(url, headers=header, json=data, stream=True)

    # construct the full path to the output file
    output_path = os.path.join(output_folder, f"output_{filename}")
    print(resp.json())
    # write the response to the output file
    with open(output_path, 'w', encoding='utf-8') as f:
        content = resp.json()
        f.write(content['choices'][0]['message']['content'])
    
    #time.sleep(2)