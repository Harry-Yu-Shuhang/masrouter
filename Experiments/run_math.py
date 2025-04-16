import sys
import os
import argparse
import yaml
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
from MAR.Utils.utils import fix_random_seed, split_list
from MAR.Utils.const import MAR_ROOT
from MAR.Utils.globals import Cost, PromptTokens, CompletionTokens
from MAR.Utils.log import configure_logging
from Datasets.math_dataset import load_math_dataset, MATH_get_predict, MATH_is_correct

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def parse_args():
    parser = argparse.ArgumentParser(description="AgentPrune Experiments on MATH")
    parser.add_argument("--data_path", type=str, default="Datasets/MATH")
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--num_rounds', type=int, default=1)
    parser.add_argument('--domain', type=str, default="math")
    parser.add_argument('--decision_method', type=str, default='FinalRefer')
    parser.add_argument('--prompt_file', type=str, default='MAR/Roles/FinalNode/math.json')
    parser.add_argument('--start_epoch', type=int, default=0)
    return parser.parse_args()

def dataloader(data_list, batch_size, i_batch):
    return data_list[i_batch * batch_size : i_batch * batch_size + batch_size]

if __name__ == '__main__':
    args = parse_args()
    dataset = load_math_dataset(args.data_path, split='train') + load_math_dataset(args.data_path, split='test')
    train_dataset, test_dataset = split_list(dataset, 0.2)

    current_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    log_file = f"math_{current_time}.txt"
    fix_random_seed(1234)
    configure_logging(log_name=log_file)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    router = MasRouter().to(device)
    optimizer = torch.optim.Adam(router.parameters(), lr=args.lr)
    tasks = tasks_profile
    llms = llm_profile
    reasonings = reasoning_profile

    logger.info("Start training...")
    train_batches = int(len(train_dataset) / args.batch_size)
    for epoch in range(args.epochs):
        logger.info(f"Epoch {epoch}", 80 * '-')
        total_solved, total_executed = (0, 0)
        if epoch < args.start_epoch:
            router.load_state_dict(torch.load(f"math_router_epoch{epoch}.pth", map_location=torch.device('cuda')))
            continue

        for i_batch in range(train_batches):
            logger.info(f"Batch {i_batch}", 80 * '-')
            start_ts = time.time()
            current_batch = dataloader(train_dataset, args.batch_size, i_batch)
            queries = [item['problem'] for item in current_batch]
            answers = [item['solution'] for item in current_batch]
            task_labels = [1 for _ in current_batch]
            tasks_y = torch.tensor(task_labels).to(device)

            optimizer.zero_grad()
            results, costs, log_probs, tasks_probs = router.forward(queries, tasks, llms, reasonings, task_labels, prompt_file=args.prompt_file)

            task_loss = F.cross_entropy(tasks_probs, tasks_y)
            utilities = []
            answers_loss = []

            for result, true_answer, log_prob, cost in zip(results, answers, log_probs, costs):
                predict_answer = MATH_get_predict(result)
                is_solved = MATH_is_correct(predict_answer, true_answer)
                total_solved += int(is_solved)
                total_executed += 1
                utility = int(is_solved) - cost * 10
                answer_loss = -log_prob * utility
                utilities.append(utility)
                answers_loss.append(answer_loss)
                logger.debug(f"Raw Result: {result}")
                logger.debug(f"Predict: {predict_answer}")
                logger.debug(f"Truth: {true_answer}")
                logger.debug(f"Cost: {cost}")

            answer_loss = sum(answers_loss) / len(answers_loss)
            loss = task_loss + answer_loss
            loss.backward()
            optimizer.step()

            accuracy = total_solved / total_executed
            logger.info(f"Batch time {time.time() - start_ts:.3f}")
            logger.info(f"Accuracy: {accuracy}")
            logger.info(f"utilities: {utilities}")
            logger.info(f"task_loss: {task_loss.item()}")
            logger.info(f"answer_loss: {answer_loss.item()}")
            logger.info(f"loss: {loss.item()}")
            logger.info(f"PromptTokens {PromptTokens.instance().value}")
            logger.info(f"CompletionTokens {CompletionTokens.instance().value}")

        torch.save(router.state_dict(), f"math_router_epoch{epoch}.pth")

    logger.info("Finish training...")
    logger.info("Start testing...")

    test_batches = int(len(test_dataset) / args.batch_size)
    total_solved, total_executed = (0, 0)

    for i_batch in range(test_batches):
        logger.info(f"Batch {i_batch}", 80 * '-')
        start_ts = time.time()
        current_batch = dataloader(test_dataset, args.batch_size, i_batch)
        queries = [item['problem'] for item in current_batch]
        answers = [item['solution'] for item in current_batch]
        task_labels = [1 for _ in current_batch]
        tasks_y = torch.tensor(task_labels).to(device)

        results, costs, log_probs, tasks_probs = router.forward(queries, tasks, llms, reasonings, task_labels, prompt_file=args.prompt_file)

        for result, true_answer, log_prob, cost in zip(results, answers, log_probs, costs):
            predict_answer = MATH_get_predict(result)
            is_solved = MATH_is_correct(predict_answer, true_answer)
            total_solved += int(is_solved)
            total_executed += 1
            logger.debug(f"Raw Result: {result}")
            logger.debug(f"Predict: {predict_answer}")
            logger.debug(f"Truth: {true_answer}")

        accuracy = total_solved / total_executed
        logger.info(f"Batch time {time.time() - start_ts:.3f}")
        logger.info(f"Accuracy: {accuracy}")
        logger.info(f"PromptTokens {PromptTokens.instance().value}")
        logger.info(f"CompletionTokens {CompletionTokens.instance().value}")

    logger.info("Finish testing...")