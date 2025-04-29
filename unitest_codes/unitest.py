import sys
import os
import importlib
import torch
from loguru import logger

# ============ 全局配置 =====================
# 当前路径
BASE_DIR = os.path.dirname(__file__)
CLASSIFIER_DIR = os.path.join(BASE_DIR, "Classifier")
LOG_DIR = os.path.join(BASE_DIR, "logs/unitest")
os.makedirs(LOG_DIR, exist_ok=True)

# 加载所有分类器
sys.path.append(CLASSIFIER_DIR)

# ============ 测试 Queries 和 真实标签 ===========
test_queries = [
    "I'm feeling really sad and overwhelmed.",    # EmotionalCare
    "Tell me a funny joke, I need to laugh.",      # PlayfulChat
    "Can you help me decide between two job offers?", # ThoughtfulGuide
    "I feel anxious about my upcoming exam.",      # EmotionalCare
    "Let's have some fun, flirt with me a little.", # PlayfulChat
    "What's the best way to improve my productivity?" # ThoughtfulGuide
]

true_labels = [
    "EmotionalCare",
    "PlayfulChat",
    "ThoughtfulGuide",
    "EmotionalCare",
    "PlayfulChat",
    "ThoughtfulGuide"
]

task_names = ["EmotionalCare", "PlayfulChat", "ThoughtfulGuide"]

# ========== 发现所有分类器 ==========
def find_classifiers():
    classifiers = []
    for filename in os.listdir(CLASSIFIER_DIR):
        if filename.endswith(".py") and not filename.startswith("__"):
            classifiers.append(filename[:-3])  # 去掉.py
    return classifiers

# ========== 测试一个分类器 ==========
def test_classifier(classifier_name):
    module = importlib.import_module(classifier_name)
    classifier = module.build_classifier()  # 需要每个分类器实现 build_classifier()

    # 编码 queries 和 tasks
    queries = test_queries
    tasks = [f"{name}: description" for name in task_names]

    tasks_probs = classifier(queries, tasks)

    predicted_labels = []
    correct_count = 0

    for i, probs in enumerate(tasks_probs):
        pred_idx = torch.argmax(probs).item()
        pred_label = task_names[pred_idx]
        predicted_labels.append(pred_label)
        if pred_label == true_labels[i]:
            correct_count += 1

    accuracy = correct_count / len(test_queries) * 100
    return predicted_labels, accuracy

# ========== 执行所有分类器 ==========
results = {}

for clf_name in find_classifiers():
    preds, acc = test_classifier(clf_name)
    results[clf_name] = {
        "predicted_labels": preds,
        "accuracy": acc
    }

# ========== 输出 Markdown ==========
markdown_path = os.path.join(LOG_DIR, "compare_classifiers.md")
with open(markdown_path, "w", encoding="utf-8") as f:
    f.write("| No. | Query | True Label | " + " | ".join(results.keys()) + " |\n")
    f.write("|:---:|:------|:----------:|" + "|".join([":---:" for _ in results]) + "|\n")
    for i, query in enumerate(test_queries):
        row = f"| {i} | {query} | {true_labels[i]} "
        for clf in results:
            pred = results[clf]["predicted_labels"][i]
            correct = "✅" if pred == true_labels[i] else "❌"
            row += f"| {pred} {correct} "
        row += "|\n"
        f.write(row)

    f.write("\n")
    for clf, info in results.items():
        f.write(f"**{clf} Accuracy: {info['accuracy']:.2f}%**\n\n")

print(f"Finished! Result saved to {markdown_path}")