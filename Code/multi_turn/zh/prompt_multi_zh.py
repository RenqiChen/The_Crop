from openai import OpenAI
import os
import json
import textwrap
import time

# set up the OpenAI client
os.environ["OPENAI_API_KEY"] = "sk"
client1 = OpenAI()
client2 = OpenAI()

input_path = 'multi_zh.txt'
output_path = 'multi_zh_re.txt'

initial_query = "农民（用户）：【我注意到玉米叶子上有一些奇怪的斑点。 它们呈黄色，似乎正在蔓延。 这可能是什么？】"

# In practical use of our multi-turn dialogue, this function invokes RAG. 
# However, due to the proprietary nature of our used RAG, we have replaced this part in the open-source code with a more accessible method of reading text content.
def call_context(path):
    with open(input_path, 'r', encoding='utf-8') as file:
        context = file.read()
    return context

system_text2 = call_context(input_path)

system_text1 = "你是一位作物病虫害专家（助理）。我是一位农民（用户），我因对作物健康状况感到担忧而向你寻求帮助。你的任务是与我进行对话，根据你的专业知识提供信息，帮助识别问题并提出可能的解决方案。"
system_text3 = textwrap.dedent("""请遵循以下准则：
                                1. 你可以获取广泛的专业领域知识。使用这些知识来完善你的回答从而提供准确、有用的建议。
                                2. 对于每一轮对话，你都可以访问一段与对话内容高度相关的参考文本。请使用此上下文来补充你的回复。本轮的参考文本包含在<doc></doc>标记之间。
                                3. 请确保你只考虑以下知识密集型问题解决场景：农作物害虫侵害。这种场景中的对话涉及根据农民的描述和询问，识别农作物害虫问题并提供适配的解决方案。
                                4. 你必须在整个对话中保持你作为作物病虫害专家（助理）的角色。不要切换为其他角色。
                                5. 你必须一次仅提供一个回复，且该回复应当非空。
                                6. 当我以“谢谢，我得到了我需要的信息”请求结束对话时，确保及时通过回复“不客气”来终止对话。
                                7. 请确保完全遵循<example></example>标记之间示例的格式输出回复，将【】标记之间的内容替换为你的回复。""")

system_text4 = "作物病虫害专家（助理）：【】"

# Format the strings
system_prompt_assistant = f"{system_text1}\n<doc>\n{system_text2}\n</doc>\n{system_text3}\n<example>\n{system_text4}\n</example>"

system_text6 = call_context(input_path)

system_text5 = "你是一位农民（用户），你对作物的健康状况感到担忧。我是一位作物病虫害专家（助理），你向我寻求帮助。你的任务是与我进行对话，描述你观察到的症状，提出问题，并根据我的专业知识寻求可能的解决方案。"
system_text7 = textwrap.dedent("""请遵循以下准则：
                                1. 你拥有自己作物的第一手经验。请利用这些知识给出详细的描述或查询。
                                2. 对于每一轮对话，你都可以访问一段与对话内容高度相关的参考文本。请使用此上下文来补充你的描述或查询。本轮的参考文本包含在<doc></doc>标记之间。 
                                3. 你仅参与以下知识密集型问题解决场景：农作物害虫侵害。这种场景中的对话涉及描述作物的症状并寻求潜在害虫问题的解决方案。
                                4. 你必须在整个对话中保持你作为农民（用户）的角色。不要切换为其他角色。
                                5. 你必须一次仅提供一个描述或查询，且该描述或查询应当非空。
                                6. 当你已经获得所需的信息时，确保通过回复“谢谢，我得到了我需要的信息”来及时终止对话。
                                7. 请确保完全遵循<example></example>标记之间示例的格式输出回复，将【】标记之间的内容替换为你的回复。""")

system_text8 = "农民（用户）：【】"

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

    if '不客气' in assistant_content:
        break
    time.sleep(1)

with open(output_path, 'w', encoding='utf-8') as f:
    for i in d_t:
        f.write(i["content"] + '\n')