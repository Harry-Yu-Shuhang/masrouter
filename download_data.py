import os
import json
from datasets import load_dataset

def should_download(path: str) -> bool:
    return not os.path.exists(path)

def save_jsonl(dataset, output_path):
    with open(output_path, "w") as f:
        for item in dataset:
            f.write(json.dumps(item) + "\n")

def download_gsm8k():
    output_path = "Datasets/gsm8k/gsm8k.jsonl"
    if not should_download(output_path):
        print("⏩ Skipping GSM8K, already exists.")
        return

    print("📥 Downloading GSM8K...")
    os.makedirs("Datasets/gsm8k", exist_ok=True)
    data = load_dataset("gsm8k", "main")['test']
    save_jsonl(data, output_path)
    print("✅ Saved GSM8K")

def download_mbpp():
    output_path = "Datasets/mbpp/mbpp.jsonl"
    if not should_download(output_path):
        print("⏩ Skipping MBPP, already exists.")
        return

    print("📥 Downloading MBPP...")
    os.makedirs("Datasets/mbpp", exist_ok=True)
    data = load_dataset("mbpp")['test']
    save_jsonl(data, output_path)
    print("✅ Saved MBPP")

def download_humaneval():
    output_path = "Datasets/humaneval/humaneval-py.jsonl"
    if not should_download(output_path):
        print("⏩ Skipping HumanEval, already exists.")
        return

    print("📥 Downloading HumanEval...")
    os.makedirs("Datasets/humaneval", exist_ok=True)
    data = load_dataset("openai_humaneval")['test']
    save_jsonl(data, output_path)
    print("✅ Saved HumanEval")

def download_math():
    subject_configs = [
        "algebra", "counting_and_probability", "geometry", "intermediate_algebra",
        "number_theory", "prealgebra", "precalculus"
    ]

    all_train, all_test = [], []

    for subject in subject_configs:
        print(f"📥 Downloading MATH subject: {subject}...")
        train_data = load_dataset("EleutherAI/hendrycks_math", subject, split="train")
        test_data = load_dataset("EleutherAI/hendrycks_math", subject, split="test")
        all_train.extend(train_data)
        all_test.extend(test_data)

    os.makedirs("Datasets/MATH", exist_ok=True)
    save_jsonl(all_train, "Datasets/MATH/train.jsonl")
    save_jsonl(all_test, "Datasets/MATH/test.jsonl")
    print("✅ Saved MATH train and test splits.")

def download_mmlu():
    subjects = [
        'abstract_algebra', 'anatomy', 'astronomy', 'business_ethics', 'clinical_knowledge',
        'college_biology', 'college_chemistry', 'college_computer_science', 'college_mathematics',
        'college_medicine', 'college_physics', 'computer_security', 'conceptual_physics',
        'econometrics', 'electrical_engineering', 'elementary_mathematics', 'formal_logic',
        'global_facts', 'high_school_biology', 'high_school_chemistry', 'high_school_computer_science',
        'high_school_european_history', 'high_school_geography', 'high_school_government_and_politics',
        'high_school_macroeconomics', 'high_school_mathematics', 'high_school_microeconomics',
        'high_school_physics', 'high_school_psychology', 'high_school_statistics', 'high_school_us_history',
        'high_school_world_history', 'human_aging', 'human_sexuality', 'international_law', 'jurisprudence',
        'logical_fallacies', 'machine_learning', 'management', 'marketing', 'medical_genetics',
        'miscellaneous', 'moral_disputes', 'moral_scenarios', 'nutrition', 'philosophy', 'prehistory',
        'professional_accounting', 'professional_law', 'professional_medicine', 'professional_psychology',
        'public_relations', 'security_studies', 'sociology', 'us_foreign_policy', 'virology', 'world_religions'
    ]

    os.makedirs("Datasets/MMLU/data", exist_ok=True)

    for subject in subjects:
        output_path = f"Datasets/MMLU/data/{subject}.jsonl"
        if not should_download(output_path):
            print(f"⏩ Skipping MMLU {subject}, already exists.")
            continue
        print(f"📥 Downloading MMLU subject: {subject}...")
        data = load_dataset("cais/mmlu", subject, split="test")
        save_jsonl(data, output_path)
        print(f"✅ Saved MMLU {subject}")

def download_lccc(version="base"):
    output_path = f"Datasets/lccc/lccc_{version}.jsonl"
    if not should_download(output_path):
        print(f"⏩ Skipping LCCC-{version}, already exists.")
        return

    print(f"📥 Downloading LCCC-{version}...")
    os.makedirs("Datasets/lccc", exist_ok=True)
    data = load_dataset("lccc", version)['train']
    save_jsonl(data, output_path)
    print(f"✅ Saved LCCC-{version}")

def download_daily_dialogue():
    output_path = "Datasets/daily_dialogue/daily_dialogue.jsonl"
    if not should_download(output_path):
        print("⏩ Skipping DailyDialog, already exists.")
        return

    print("📥 Downloading DailyDialog...")
    os.makedirs("Datasets/daily_dialogue", exist_ok=True)
    data = load_dataset("li2017dailydialog/daily_dialog", trust_remote_code=True)['train']
    save_jsonl(data, output_path)
    print("✅ Saved DailyDialog")

if __name__ == "__main__":
    # download_gsm8k()
    # download_mbpp()
    # download_humaneval()
    # download_math()
    # download_mmlu()
    # download_lccc(version="base")
    download_daily_dialogue()
    print("\n🎉 All datasets downloaded and saved in MasRouter format!")