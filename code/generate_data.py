import re
from openai import OpenAI
import json
import sys
from tqdm import tqdm
import RAGInterface

def extract_dialogue(text: str):
    dialogue_list = []
    pattern = re.compile(r'^(患者|护士)：(.*)$', re.MULTILINE)
    
    for match in pattern.finditer(text):
        role, content = match.groups()
        entry = {
            "from": "user" if role == "患者" else "assistant",
            "value": content.strip()
        }
        dialogue_list.append(entry)
    
    return dialogue_list



numbers = 100
client = OpenAI(api_key="sk-c6abf11ab86f4bb6b38ef96c234ca89b", base_url="https://api.deepseek.com")
random_context = RAGInterface.random_sample(numbers)
sharegpt_data = []

json_file = "test_data.json"
# 1. 先写入一个 '['，表示 JSON 数组的开始
with open(json_file, "w", encoding="utf-8") as f:
    f.write("[\n")

for i in tqdm(range(numbers), desc="Generating data"):
    context = random_context[i]
    response = client.chat.completions.create(
        model="deepseek-reasoner",
        messages=[
            {"role": "system", "content": '''你的回答应该尽量简短。
    【全局背景】
    在中国医院的导诊台，一名护士正在和患者对话，患者想知道自己应该挂什么科室。
    【你的任务】
    根据背景，生成对话内容，对话尽量有多轮对话，但是患者和护士总共的说话次数不要超过6次，当护士给出科室名字之后，对话应该立刻结束，不允许出现客套话内容。
    【样例】
    患者：我头疼了两天了。
    护士：有发热，咳嗽等症状吗？
    患者：都有点。
    护士：根据您的描述，您应该去挂神经内科，但考虑到你也有咳嗽发热等症状，也可以考虑挂呼吸内科。
    【要求】
    严格根据样例的格式，仅输出类似样例中的内容。最终护士应该给出一个科室，或者两个如果症状比较多的话。'''},
            {"role": "user", "content": '''【参考资料】
            {context}
            根据样例，结合参考资料，如果参考资料和我要求的内容无关，则不考虑，根据你自己的知识库去生成。现在生成一个对话。'''.format(context=context)},
        ],
        stream=False,
        temperature=1.5
    )

    answer = response.choices[0].message.content
    try:
        answer = extract_dialogue(answer)
    except:
        continue
    
    single_data = {
        "conversations": answer,
        "system": "现在你要扮演一个医院中导诊台的护士，你的职责是根据患者的病情描述，告诉他们应该挂什么科室。如果病情描述较少，可以继续询问其他症状。最终必须给出具体的1个或者2个科室。给出科室之后，对话结束。"
    }
    # 2. 将该对象以 JSON 形式追加到文件末尾
    #    如果不是第一个，就在前面写一个逗号作为分隔
    with open(json_file, "a", encoding="utf-8") as f:
        if i > 0:
            f.write(",\n")
        json.dump(single_data, f, ensure_ascii=False, indent=4)
    # sharegpt_data.append({"conversations": answer, "system": "现在你要扮演一个医院中导诊台的护士，你的职责是根据患者的病情描述，告诉他们应该挂什么科室。如果病情描述较少，可以继续询问其他症状。要求对话尽量简短。最终必须给出具体的1个或者2个科室。给出科室之后，对话结束。"})

# print("Data generated, start to write to disk")
# # 写入 JSON 文件
# with open("dialogues.json", "w", encoding="utf-8") as f:
#     json.dump(sharegpt_data, f, ensure_ascii=False, indent=4)

# 3. 在全部完成后，写一个 ']' 标记数组结束
with open(json_file, "a", encoding="utf-8") as f:
    f.write("\n]\n")

print("Data saved to dialogues.json")


