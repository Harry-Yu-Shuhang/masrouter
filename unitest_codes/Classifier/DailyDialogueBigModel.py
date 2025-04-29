# Classifier/DailyDialogueBigModel.py
from sentence_transformers import SentenceTransformer
import torch
from typing import List

# ========== 外部调用接口 ==========
def build_classifier():
    model = SentenceTransformer('all-MiniLM-L6-v2')  # 这里用的小模型，你可以改成更大的，比如 gtr-t5-large 等
    model = model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    def classifier_fn(queries: List[str], tasks: List[str]):
        with torch.no_grad():
            query_emb = model.encode(queries, convert_to_tensor=True)
            task_emb = model.encode(tasks, convert_to_tensor=True)
            query_emb = torch.nn.functional.normalize(query_emb, p=2, dim=1)
            task_emb = torch.nn.functional.normalize(task_emb, p=2, dim=1)
            scores = torch.matmul(query_emb, task_emb.T)
        return scores

    return classifier_fn
