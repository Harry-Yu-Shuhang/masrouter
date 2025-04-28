from MAR.Utils.globals import Cost, PromptTokens, CompletionTokens
import tiktoken
# GPT-4:  https://platform.openai.com/docs/models/gpt-4-and-gpt-4-turbo
# GPT3.5: https://platform.openai.com/docs/models/gpt-3-5
# DALL-E: https://openai.com/pricing

def cal_token(model:str, text:str):
    encoder = tiktoken.encoding_for_model('gpt-4o')
    num_tokens = len(encoder.encode(text))
    return num_tokens

def cost_count(prompt, response, model_name):
    branch: str
    prompt_len: int
    completion_len: int
    price: float

    prompt_len = cal_token(model_name, prompt)
    completion_len = cal_token(model_name, response)
    prompt_price = MODEL_PRICE[model_name]["input"]
    completion_price = MODEL_PRICE[model_name]["output"]
    price = prompt_len * prompt_price / 1000000 + completion_len * completion_price / 1000000

    Cost.instance().value += price
    PromptTokens.instance().value += prompt_len
    CompletionTokens.instance().value += completion_len

    # print(f"Prompt Tokens: {prompt_len}, Completion Tokens: {completion_len}")
    return price, prompt_len, completion_len

MODEL_PRICE = {
    "deepseek-ai/DeepSeek-R1":{
        "input": 0.23,  
        "output": 0.91   
    },
    "Doubao-1.5-pro-32k":{
        "input": 0.25,  
        "output": 0.62  
    },
    "Qwen/Qwen2.5-7B-Instruct":{
        "input": 0.42, 
        "output": 0.42   
    },
    "deepseek-ai/DeepSeek-V3":{
        "input": 0.11, 
        "output": 0.45   
    },
    "gemini-2.5-flash-preview-04-17":{
        "input": 0.52, 
        "output": 2.08
    },
}
