import torch
from MAR.MasRouter.mas_router import MasRouter
from MAR.Prompts.tasks_profile import tasks_profile
from MAR.LLM.llm_profile import llm_profile
from MAR.Agent.reasoning_profile import reasoning_profile
from loguru import logger
import os

# 创建 logs/unitest 目录
log_dir = "logs/unitest"
os.makedirs(log_dir, exist_ok=True)

# 添加 loguru 文件输出
logger.add(os.path.join(log_dir, "unitest_results.txt"), rotation="500 KB", encoding="utf-8")

# 初始化 MasRouter
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
router = MasRouter(device=device)

# 测试 Queries
test_queries = [
    "I'm feeling really sad and overwhelmed.",                
    "Tell me a funny joke, I need to laugh.",                 
    "Can you help me decide between two job offers?",         
    "I feel anxious about my upcoming exam.",                 
    "Let's have some fun, flirt with me a little.",           
    "What's the best way to improve my productivity?"         
]

# 执行测试
results, _, _, _ = router.forward(
    queries=test_queries,
    tasks=tasks_profile,
    llms=llm_profile,
    reasonings=reasoning_profile,
)

# 日志打印结果
for i, query in enumerate(test_queries):
    logger.info(f"Query {i}: {query}")
    logger.info(f"Selected Result: {results[i]}")
    logger.info("-" * 50)

print(f"Results saved to {os.path.join(log_dir, 'unitest_results.txt')}")
