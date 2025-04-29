import torch
import torch.nn.functional as F
from llm2vec import LLM2Vec

device = torch.device('mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu'))

class LLM2VecClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = LLM2Vec.from_pretrained(
            "McGill-NLP/LLM2Vec-Sheared-LLaMA-mntp",  # ✅ 选的是公开不需要申请 access 的 Sheared-LLaMA 模型
            device_map="auto",
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        )
        self.input_dim = self.model.config.hidden_size
        self.query_encoder = torch.nn.Linear(self.input_dim, 128).to(device)
        self.task_encoder = torch.nn.Linear(self.input_dim, 128).to(device)

    def forward(self, queries, tasks):
        with torch.no_grad():
            queries_emb = self.model.encode(queries).to(device)
            tasks_emb = self.model.encode(tasks).to(device)

        queries_emb = F.normalize(self.query_encoder(queries_emb), p=2, dim=1)
        tasks_emb = F.normalize(self.task_encoder(tasks_emb), p=2, dim=1)

        scores = torch.matmul(queries_emb, tasks_emb.T)
        return scores

def build_classifier():
    return LLM2VecClassifier()