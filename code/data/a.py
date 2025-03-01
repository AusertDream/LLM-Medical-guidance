import json

# 读取jsonl文件并将其转换为所需格式的json文件
def convert_jsonl_to_jsonl(jsonl_filename, json_filename):
    data = []

    with open(jsonl_filename, 'r', encoding='utf-8') as jsonl_file:
        for line in jsonl_file:
            # 解析每一行的JSON对象
            line_data = json.loads(line.strip())
            
            # 通过"dialog"字段提取输入输出
            dialog = line_data['dialog']
            label = line_data['label']

            # 处理dialog为input和output
            instruction = "根据对话内容推荐就诊科室"
            input_text = dialog
            output_text = label

            # 构建新的数据格式
            formatted_data = {
                "instruction": instruction,
                "input": input_text,
                "output": output_text
            }
            
            # 添加到最终的列表中
            data.append(formatted_data)

    # 将转换后的数据写入到JSON文件中
    with open(json_filename, 'w', encoding='utf-8') as json_file:
        json.dump(data, json_file, ensure_ascii=False, indent=2)

# 输入和输出文件路径
jsonl_filename = 'input.jsonl'  # 替换为你自己的输入文件路径
json_filename = 'output.json'   # 替换为你自己的输出文件路径

# 执行转换
convert_jsonl_to_jsonl(jsonl_filename, json_filename)

print(f"转换完成，结果已保存到 {json_filename}")
