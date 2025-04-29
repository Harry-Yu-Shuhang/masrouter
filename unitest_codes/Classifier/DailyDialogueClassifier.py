import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

device = torch.device('mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu'))

class DailyDialogueClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.input_dim = self.encoder.get_sentence_embedding_dimension()
        self.query_encoder = torch.nn.Linear(self.input_dim, 128).to(device)
        self.task_encoder = torch.nn.Linear(self.input_dim, 128).to(device)

    def forward(self, queries, tasks):
        with torch.no_grad():
            queries_emb = self.encoder.encode(queries, convert_to_tensor=True).to(device)
            tasks_emb = self.encoder.encode(tasks, convert_to_tensor=True).to(device)

        queries_emb = F.normalize(self.query_encoder(queries_emb), p=2, dim=1)
        tasks_emb = F.normalize(self.task_encoder(tasks_emb), p=2, dim=1)

        scores = torch.matmul(queries_emb, tasks_emb.T)
        return scores

def build_classifier():
    return DailyDialogueClassifier()