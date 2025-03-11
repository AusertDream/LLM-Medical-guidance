import os
import shutil

def extract_md_files(source_dir, target_dir):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    for root, dirs, files in os.walk(source_dir):
        if 'auto' in dirs:
            auto_dir = os.path.join(root, 'auto')
            for file in os.listdir(auto_dir):
                if file.endswith('.md'):
                    source_file = os.path.join(auto_dir, file)
                    target_file = os.path.join(target_dir, file)
                    if os.path.exists(target_file):
                        base, ext = os.path.splitext(file)
                        target_file = os.path.join(target_dir, f"{base}{ext}")

                    shutil.copy(source_file, target_file)



source_directory = './output'
target_directory = './data'


extract_md_files(source_directory, target_directory)