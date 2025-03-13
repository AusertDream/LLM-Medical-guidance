import json
from datasets import load_dataset
from evaluate import inference_from_transforms
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import torch
from RAGInterface import get_context
import sys
from tqdm import tqdm

with open("./modelConfig.json", "r") as f:
        modelConfig = json.load(f)

def get_assistant_format(content):
    return {'role':'assistant', 'content':content}

def get_user_format(content):
    return {'role':'user', 'content':content}

def get_system_format(content):
    return {'role':'system', 'content':content}

def refactor_history(chat_history):
    history = []
    instruction = '现在你要扮演一个医院中导诊台的护士，你的职责是根据患者的病情描述，告诉他们应该挂什么科室。相关知识仅供参考，具体以患者的为准。给出的科室要求根据你自己的知识，不允许直接参考所给相关知识的内容。要求对话尽量简短。最终必须且只给出一个科室。现在根据已知对话内容，给出有应该去哪个科室。不需要进行询问'
    sys_commd = get_system_format(instruction)
    history.append(sys_commd)
    for chat in chat_history:
        if chat['from'] == 'assistant':
            history.append(get_assistant_format(chat['value']))
        elif chat['from'] == 'user':
            history.append(get_user_format(chat['value']))
        else:
            print("history has unknown role! Whatsup?")
    
    return history

def refactor_prompt(prompt, RAGon=True):
    if RAGon == False:
        return prompt
    else:
        res = """【相关知识】
        {context}
        【用户问题】
        {query}""".format(context=get_context(prompt, n_results=4), query=prompt)
        
        return res

def get_test_answers():
    model = AutoModelForCausalLM.from_pretrained(modelConfig["evaluate_model"])
    tokenizer = AutoTokenizer.from_pretrained(modelConfig["evaluate_model"])
    tokenizer.pad_token = tokenizer.eos_token
    generation_config = GenerationConfig(
            do_sample=True,
            temperature=0.8,
            num_beams=3,
            top_p=0.3,
            no_repeat_ngram_size=3,
            pad_token_id=0,
            max_new_tokens=256,
    )

    test_data = load_dataset("json", data_files=modelConfig["test_dataset_dir"])['train']

    model_answers = []

    for sample in tqdm(test_data):
        context = ''
        for t in sample['conversations'][:-1]:
            context += t['value']
        messages = refactor_history(sample['conversations'][:-1])
        context = get_context(context, n_results=1)
        str = """【相关知识】
        {context}
        【用户问题】
        {query}""".format(context=context, query=sample['conversations'][-2]['value'])
        messages[-1] = get_user_format(str)
        # print(messages[-1])
        model_answer = inference_from_transforms(messages, generation_config, model, tokenizer, modelConfig['device_map'])
        # print(model_answer)
        model_answers.append(model_answer['content'])

    return model_answers

    #     with open('./data/test_answers_noRAG2.txt', 'a') as f:
    #     for answer in model_answers:
    #         f.write(answer + '\n')

    # print("Done!")

def get_accuracy(model_answers):
    normal_correct_number = 0
    # model_answers = []
    # with open('./data/test_answers_noRAG2.txt', 'r') as f:
    #     model_answers = f.readlines()
    human_answers = []
    with open('./data/test_label.txt', 'r') as f:
        human_answers = f.readlines()
    
    for i in range(len(human_answers)):
        temp = human_answers[i].split('、')[1].strip()
        human_answers[i] = temp.split('/')
    
    

    for i in range(len(model_answers)):
        isTrue = False
        for j in range(len(human_answers[i])):

            if human_answers[i][j] in model_answers[i]:
                isTrue = True
                break
        
        if isTrue:
            normal_correct_number += 1
        # else:
        #     print("Human answer:", human_answers[i])
        #     print("Model answer:", model_answers[i])
        #     print("---------------------")
    # print(normal_correct_number)
    return normal_correct_number
ans = []
for i in range(5):
    a = get_accuracy(get_test_answers())
    ans.append(a)

print(ans)
