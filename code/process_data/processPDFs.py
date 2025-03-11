"""
If you want to run this python file, plz ensure the environment is "python >= 3.10"

"""
import os
import sys
from tqdm import tqdm
import subprocess

dir = "F:\\FastIn\\UniversityProduction\\GradutionProject\\LLM-Medical-guidance\\code\\process_data\\data"
files = os.listdir(dir)
idx = []
for i, file in enumerate(files):
    if " " in file or True:
        idx.append(i)


for i in tqdm(idx, desc="processing"):
    file = files[i]
    if " " in file:
        str = file.replace(" ", "")

    pdfpath = os.path.join(dir, file)
    command = f"magic-pdf -p \"{pdfpath}\" -o ./output"
    work_path = "./temp"
    # 在指定的目录下运行命令
    result = subprocess.run(command, shell=True, cwd=work_path, text=True)
    print(result)
