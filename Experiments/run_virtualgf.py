import sys
import os
import argparse
import json
import time
import torch
import torch.nn.functional as F
from loguru import logger
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.stdout.reconfigure(encoding='utf-8')

from MAR.MasRouter.mas_router import MasRouter
from MAR.LLM.llm_profile import llm_profile
from MAR.Agent.reasoning_profile import reasoning_profile
from MAR.Prompts.tasks_profile import tasks_profile
from MAR.Tools.reader.readers import JSONLReader
from MAR.Utils.utils import fix_random_seed, split_list
from MAR.Utils.globals import PromptTokens, CompletionTokens
from MAR.Utils.log import configure_logging
# from Datasets.lccc_dataset import lccc_data_process, lccc_get_predict
from Datasets.daily_dialogue_dataset import daily_dialogue_data_process, daily_dialogue_get_predict

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def dataloader(data_list, batch_size, i_batch):
    return data_list[i_batch*batch_size:i_batch*batch_size + batch_size]

def parse_args():
    parser = argparse.ArgumentParser(description="MasRouter Experiments on Dail Dialogue (Virtual Girlfriend)")
    parser.add_argument("--dataset_json", type=str, default="Datasets/daily_dialogue/sample_daily_dialogue.jsonl")
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--start_epoch', type=int, default=0)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    # 检查 Models 文件夹
    if not os.path.exists("Models/virtual_gf"):
        logger.warning("Models/virtual_gf 文件夹不存在，正在创建...")
        os.makedirs("Models/virtual_gf", exist_ok=True)
        logger.info("✅ Models/virtual_gf 文件夹已创建。")


    dataset = JSONLReader.parse_file(args.dataset_json)
    dataset = daily_dialogue_data_process(dataset)
    train_dataset, test_dataset = split_list(dataset, 0.2)

    current_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    log_file = f"Results/virtualgf_{current_time}.txt"
    fix_random_seed(1234)
    configure_logging(log_name=log_file)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    router = MasRouter().to(device)
    optimizer = torch.optim.Adam(router.parameters(), lr=args.lr)
    tasks = tasks_profile
    llms = llm_profile
    reasonings = reasoning_profile

    test_results = []  # 用于记录每个epoch的测试表现

    logger.info("Start training...")
    train_batches = int(len(train_dataset) / args.batch_size)
    for epoch in range(args.epochs):
        logger.info(f"Epoch {epoch}", 80 * '-')
        total_solved, total_executed = 0, 0

        if epoch < args.start_epoch:
            router.load_state_dict(torch.load(f"Models/virtual_gf/virtualgf_router_epoch{epoch}.pth", map_location=device))
            continue

        for i_batch in range(train_batches):
            logger.info(f"Batch {i_batch}", 80 * '-')
            current_batch = dataloader(train_dataset, args.batch_size, i_batch)
            queries = [item['task'] for item in current_batch]
            answers = [item['answer'] for item in current_batch]
            task_labels = [0 for _ in current_batch]
            tasks_y = torch.tensor(task_labels).to(device)

            optimizer.zero_grad()
            results, costs, log_probs, tasks_probs = router.forward(queries, tasks, llms, reasonings, task_labels)
            task_loss = F.cross_entropy(tasks_probs, tasks_y)

            answers_loss = []
            for result, true_answer, log_prob, cost in zip(results, answers, log_probs, costs):
                is_solved = daily_dialogue_get_predict(result, true_answer)
                total_solved += is_solved
                total_executed += 1
                utility = is_solved - cost * 5
                answers_loss.append(-log_prob * utility)

            answer_loss = sum(answers_loss) / len(answers_loss)
            loss = task_loss + answer_loss
            loss.backward()
            optimizer.step()

        torch.save(router.state_dict(), f"Models/virtual_gf/virtualgf_router_epoch{epoch}.pth")

        # 测试阶段
        logger.info("Start testing...")
        test_batches = int(len(test_dataset) / args.batch_size)
        total_solved, total_executed = 0, 0
        PromptTokens.instance().reset()
        CompletionTokens.instance().reset()

        for i_batch in range(test_batches):
            current_batch = dataloader(test_dataset, args.batch_size, i_batch)
            queries = [item['task'] for item in current_batch]
            answers = [item['answer'] for item in current_batch]
            task_labels = [0 for _ in current_batch]
            results, costs, log_probs, tasks_probs = router.forward(queries, tasks, llms, reasonings, task_labels)

            for result, true_answer in zip(results, answers):
                is_solved = daily_dialogue_get_predict(result, true_answer)
                total_solved += is_solved
                total_executed += 1

        accuracy = total_solved / total_executed
        logger.info(f"Epoch {epoch} Test Accuracy: {accuracy:.4f}")
        logger.info(f"PromptTokens {PromptTokens.instance().value}")
        logger.info(f"CompletionTokens {CompletionTokens.instance().value}")

        # 记录测试结果
        test_results.append({
            "epoch": epoch,
            "accuracy": accuracy,
            "prompt_tokens": PromptTokens.instance().value,
            "completion_tokens": CompletionTokens.instance().value
        })

    # 写入 Markdown 表格
    os.makedirs("Results", exist_ok=True)
    with open("Results/virtualgf_results.md", "w") as f:
        f.write("| Epoch | Accuracy | Prompt Tokens | Completion Tokens |\n")
        f.write("|-------|----------|---------------|-------------------|\n")
        for res in test_results:
            f.write(f"| {res['epoch']} | {res['accuracy']:.4f} | {res['prompt_tokens']} | {res['completion_tokens']} |\n")

    logger.info("Finish training and testing...")
