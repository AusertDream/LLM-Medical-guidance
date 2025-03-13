import json

# 读取jsonl文件并将其转换为所需格式的json文件
def convert_jsonl_to_json(jsonl_filename, json_filename):
    data = []

    with open(jsonl_filename, 'r', encoding='utf-8') as jsonl_file:
        for line in jsonl_file:
            
            line_data = json.loads(line.strip())
            
            
            dialog = line_data['dialog']
            label = line_data['label']

            
            instruction = "根据对话内容推荐就诊科室"
            input_text = dialog
            output_text = label

            
            formatted_data = {
                "instruction": instruction,
                "input": input_text,
                "output": output_text
            }
            data.append(formatted_data)

    with open(json_filename, 'w', encoding='utf-8') as json_file:
        json.dump(data, json_file, ensure_ascii=False, indent=2)

# 输入和输出文件路径
jsonl_filename = 'input.jsonl'  # 替换为你自己的输入文件路径
json_filename = 'output.json'   # 替换为你自己的输出文件路径

# 执行转换
convert_jsonl_to_json(jsonl_filename, json_filename)

print(f"转换完成，结果已保存到 {json_filename}")
