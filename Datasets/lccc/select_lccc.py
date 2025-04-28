import os
import json
import random
import argparse

def sample_lccc(input_path, save_path, sample_size, min_turns=3):
    # 读取 lccc_base.jsonl
    data = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            conv = json.loads(line.strip())
            if len(conv['dialog']) >= min_turns:
                data.append(conv)

    print(f"总对话数: {len(data)}（多轮对话数: 轮次 >= {min_turns}）")

    # 随机采样
    if sample_size > len(data):
        print(f"⚠️ 采样数超过多轮对话总数，调整为 {len(data)}")
        sample_size = len(data)

    sampled_data = random.sample(data, sample_size)

    # 保存为 jsonl
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w', encoding='utf-8') as f:
        for item in sampled_data:
            json.dump(item, f, ensure_ascii=False)
            f.write('\n')
    print(f"✅ 已保存 {sample_size} 条对话到 {save_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="LCCC 数据筛选器")
    parser.add_argument('--input_path', type=str, default="Datasets/lccc/lccc_base.jsonl", help="输入路径")
    parser.add_argument('--num_samples', type=int, default=200, help="要采样的对话数")
    parser.add_argument('--save_path', type=str, default="Datasets/lccc/lccc.jsonl", help="保存路径")
    args = parser.parse_args()

    sample_lccc(args.input_path, args.save_path, args.num_samples)