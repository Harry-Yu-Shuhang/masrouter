# Datasets/daily_dialogue_dataset.py

def daily_dialogue_data_process(dataset):
    processed = []
    for item in dataset:
        if "dialog" in item:
            dialogue = item["dialog"]
            if len(dialogue) >= 2:
                task_text = f"""**Conversation Context**:
{dialogue[-2]}

**Expected Reply**:"""
                processed.append({
                    "task": task_text,
                    "answer": dialogue[-1]
                })
    return processed

def daily_dialogue_get_predict(predicted: str, ground_truth: str):
    # 简单的准确率判断，后续可以做模糊匹配
    return int(ground_truth.strip() in predicted.strip())
