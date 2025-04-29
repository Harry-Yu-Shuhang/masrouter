# Classifier/TaskClassifier.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class TaskClassifier(nn.Module):
    def __init__(self, input_dim: int = 384, hidden_dim: int = 32, device=None):
        super().__init__()
        self.query_encoder = nn.Linear(input_dim, hidden_dim)
        self.task_encoder = nn.Linear(input_dim, hidden_dim)
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, queries: torch.Tensor, tasks: torch.Tensor):
        query_embedding = self.query_encoder(queries)
        task_embedding = self.task_encoder(tasks)
        query_embedding = F.normalize(query_embedding, p=2, dim=1)
        task_embedding = F.normalize(task_embedding, p=2, dim=1)
        scores = torch.matmul(query_embedding, task_embedding.T)
        return scores

# ========== 外部调用接口 ==========
def build_classifier():
    model = TaskClassifier()
    
    # 这里返回一个 callable，保持兼容主脚本
    def classifier_fn(queries: List[str], tasks: List[str]):
        # 简单用随机向量代替embeddings（你可以换成真正的embedding提取器）
        queries_emb = torch.randn(len(queries), 384).to(model.device)
        tasks_emb = torch.randn(len(tasks), 384).to(model.device)

        with torch.no_grad():
            scores = model(queries_emb, tasks_emb)
        
        return scores
    
    return classifier_fn