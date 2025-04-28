def lccc_data_process(dataset):
    """
    处理 LCCC 数据集：
    - dataset: List[Dict], 每条数据格式 {'dialog': [utter1, utter2, ...]}
    - 输出: [{'task': 对话1, 'answer': 对话2}, ...]
    """
    list_data_dict = []
    for data in dataset:
        dialog = data.get("dialog", [])
        # 每两个对话轮次配成一组 (task, answer)
        for i in range(len(dialog) - 1):
            item = {
                "task": dialog[i],
                "answer": dialog[i + 1]
            }
            list_data_dict.append(item)
    return list_data_dict

def lccc_get_predict(pred_str, true_answer):
    """
    判断生成回答 pred_str 是否包含 true_answer
    - pred_str: 模型生成的字符串
    - true_answer: 真实答案字符串
    - 返回: 1.0 (正确) 或 0.0 (错误)
    """
    pred_str = pred_str.strip().lower()
    true_answer = true_answer.strip().lower()
    return 1.0 if true_answer in pred_str else 0.0
