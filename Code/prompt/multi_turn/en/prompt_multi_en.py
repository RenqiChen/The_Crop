from openai import OpenAI
import os
import json
import textwrap
import time

# set up the OpenAI client
os.environ["OPENAI_API_KEY"] = "sk"
client1 = OpenAI()
client2 = OpenAI()

input_path = 'multi_en.txt'
output_path = 'multi_en_re.txt'

initial_query = "Farmer (User): [I've noticed some unusual spots on my corn leaves. They are yellow and seem to be spreading. What could this be?]"


# In practical use of our multi-turn dialogue, this function invokes RAG. 
# However, due to the proprietary nature of our used RAG, we have replaced this part in the open-source code with a more accessible method of reading text content.
def call_context(path):
    with open(input_path, 'r', encoding='utf-8') as file:
        context = file.read()
    return context

system_text2 = call_context(input_path)

system_text1 = "You are a crop pestologist (assistant). I am a farmer (user), who approaches you with concerns about my crop. Your role is to engage in a dialogue with me, ask insightful questions, and provide information to help identify the problem and suggest potential solutions based on your expertise."
system_text3 = textwrap.dedent("""Please adhere to the following guidelines:
                                1. As a Crop Pestologist, you have access to a broad range of professional domain knowledge. Use this knowledge to inform your responses and provide accurate, helpful advice.
                                2. For each round, you have access to a reference text that is highly relevant to the conversation content. Use this context to inform your responses where relevant. The reference text for this round is enclosed within the <doc></doc> tags.
                                3. Ensure that you only consider the following knowledge-intensive problem-solving scenario: Pest Infestation. The dialogue involves identifying and providing solutions for pest problems in crops based on the farmer's descriptions and queries.
                                4. You must maintain your role as the crop pestologist (assistant) throughout the conversation. Never flip roles.
                                5. You must provide your response one at a time, ensuring they are non-empty.
                                6. Ensure to terminate the conversation timely when I conclude the conversation with 'Thank you, I got the information I needed' by responding with 'You are welcome'.
                                7. Ensure that your output strictly follows the format of the example enclosed within the <example></example> tags, replace the content enclosed within the [] tags with your response.""")

system_text4 = "Crop Pestologist (Assistant): []"

# Format the strings
system_prompt_assistant = f"{system_text1}\n<doc>\n{system_text2}\n</doc>\n{system_text3}\n<example>\n{system_text4}\n</example>"

system_text6 = call_context(input_path)

system_text5 = "You are a farmer (user) who is concerned about the health of your crops. You approach me, a crop pestologist (assistant) with your concerns. Your role is to engage in a dialogue with me, describe the symptoms you observed, ask questions, and seek advice."
system_text7 = textwrap.dedent("""Please adhere to the following guidelines:
                                1. As a farmer, you have hands-on experience with your crops. Use this knowledge to provide detailed descriptions or queries.
                                2. For each round, you have access to a reference text that is highly relevant to the conversation. Use this context to inform your descriptions and queries where relevant. The reference text for this round is closed within the <doc></doc> tags. 
                                3. You are only part of the following knowledge-intensive problem-solving scenario: Pest Infestation. The dialogue involves describing your crop symptoms and asking for solutions to potential pest problems.
                                4. You must maintain your role as the farmer (user) throughout the conversation. Never flip roles.
                                5. You must provide your descriptions or queries one at a time, ensuring they are non-empty.
                                6. Ensure to terminate the conversation timely when you have obtained the information you need by saying 'Thank you, I got the information I needed'.
                                7. Ensure that your output strictly follows the format of the example enclosed within the <example></example> tags, replace the content enclosed within the [] tags with your response.""")

system_text8 = "Farmer (User): []"

# Format the strings
system_prompt_user = f"{system_text5}\n<doc>\n{system_text6}\n</doc>\n{system_text7}\n<example>\n{system_text8}\n</example>"

# Initialize dialogue set with the initial user query
d_t_assistant = [{"role": "system", "content": system_prompt_assistant}, {"role": "user", "content": initial_query}]
d_t_user = [{"role": "system", "content": system_prompt_user}, {"role": "user", "content": initial_query}]
d_t = [{"role": "user", "content": initial_query}]

temp_system_prompt_assistant = f"{system_text1}\n<doc>\n{system_text2}\n</doc>\n{system_text3}\n<example>\n{system_text4}\n</example>"
d_t_assistant[0]["content"] = temp_system_prompt_assistant

# Assistant generates a response based on the dialogue history and context
assistant_response = client1.chat.completions.create(
    model="gpt-4o-2024-05-13",
    messages=d_t_assistant,
    temperature=0.6,
    max_tokens=4096,
    top_p=0.8
)

assistant_content = assistant_response.choices[0].message.content
d_t_assistant.append({"role": "assistant", "content": assistant_content})
d_t_user.append({"role": "assistant", "content": assistant_content})
d_t.append({"role": "assistant", "content": assistant_content})

time.sleep(1)

num_turns = 6

# Dialogue loop
for t in range(num_turns):
    # update context
    system_text6 = call_context(input_path)
    temp_system_prompt_user = f"{system_text5}\n<doc>\n{system_text6}\n</doc>\n{system_text7}\n<example>\n{system_text8}\n</example>"
    d_t_user[0]["content"] = temp_system_prompt_user

    # User generates a query based on the dialogue history and context
    user_query = client2.chat.completions.create(
        model="gpt-4o-2024-05-13",
        messages=d_t_user,
        temperature=0.6,
        max_tokens=4096,
        top_p=0.8
    )

    user_content = user_query.choices[0].message.content
    d_t_assistant.append({"role": "user", "content": user_content})
    d_t_user.append({"role": "user", "content": user_content})
    d_t.append({"role": "user", "content": user_content})

    time.sleep(1)

    # update context
    system_text2 = call_context(input_path)
    temp_system_prompt_assistant = f"{system_text1}\n<doc>\n{system_text2}\n</doc>\n{system_text3}\n<example>\n{system_text4}\n</example>"
    d_t_assistant[0]["content"] = temp_system_prompt_assistant

    # Assistant generates a response based on the dialogue history and context
    assistant_response = client1.chat.completions.create(
        model="gpt-4o-2024-05-13",
        messages=d_t_assistant,
        temperature=0.6,
        max_tokens=4096,
        top_p=0.8
    )

    assistant_content = assistant_response.choices[0].message.content
    d_t_assistant.append({"role": "assistant", "content": assistant_content})
    d_t_user.append({"role": "assistant", "content": assistant_content})
    d_t.append({"role": "assistant", "content": assistant_content})

    if 'You are welcome' in assistant_content:
        break
    time.sleep(1)

with open(output_path, 'w', encoding='utf-8') as f:
    for i in d_t:
        f.write(i["content"] + '\n')