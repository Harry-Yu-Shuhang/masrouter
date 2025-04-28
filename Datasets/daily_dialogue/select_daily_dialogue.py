import json
import random
import os

def load_jsonl(path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f]

def save_jsonl(data, path):
    with open(path, "w") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

def select_daily_dialogue(input_path, output_path, num_samples=100):
    # 加载原始数据
    data = load_jsonl(input_path)
    print(f"📊 Total samples in dataset: {len(data)}")

    # 随机筛选
    selected_data = random.sample(data, num_samples)
    save_jsonl(selected_data, output_path)
    print(f"✅ Saved {num_samples} samples to {output_path}")

if __name__ == "__main__":
    input_file = "Datasets/daily_dialogue/daily_dialogue.jsonl"
    output_file = "Datasets/daily_dialogue/sample_daily_dialogue.jsonl"

    # 检查文件是否存在
    if not os.path.exists(input_file):
        print(f"❌ Input file {input_file} does not exist. Please run the download script first.")
    else:
        select_daily_dialogue(input_file, output_file)
