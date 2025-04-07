from typing import List, Optional
import re
import torch
from llm_trainer import GRPOTrainer, TrainerTools
from utils import init_env, get_grpo_config
from constant import reasoning_system_prompt


# todo 提取<answer></answer>
def extract_answer_from_completion(completion_text: str)-> str:
    # <reasoning>思考</reasoning><answer>回答</answer></s>
    parts = completion_text.split("<answer>")
    if len(parts) < 2:
        return ''

    # 回答</answer></s>
    last_part = parts[-1]

    # Extract content up to </answer>
    if "</answer>" not in last_part:
        return ''

    # 回答
    answer = last_part.split("</answer>")[0].strip()
    return '' if answer == "..." else answer


def get_last_number(response_answer: str)-> Optional[str]:
    numbers = re.findall(r'-?\d+\.?\d*', response_answer)
    if numbers:
        last_num = numbers[-1]
        return last_num

    return None


def get_reward(completion_text: str, correct_answer: str)-> float:
    reward = 0.0
    response_answer = extract_answer_from_completion(completion_text)
    response_last_number = get_last_number(response_answer)

    #  正确答案奖励: 0.0 ~ 2.0
    if response_last_number == correct_answer:
        reward += 2.0 # 答案相同，奖励2
    elif correct_answer in response_answer:
        reward += 1.5 # 正确答案在回答中，奖励1.5

    #  回答格式奖励: 0.0 ~ 1.0
    if TrainerTools().tokenizer.text_reasoning_start in completion_text:
        reward += 0.2 # 答案包含<reasoning>，奖励0.2

    if TrainerTools().tokenizer.text_reasoning_end in completion_text:
        reward += 0.2 # 答案包含</reasoning>，奖励0.2

    if TrainerTools().tokenizer.text_answer_start in completion_text:
        reward += 0.2 # 答案包含<answer>，奖励0.2

    if TrainerTools().tokenizer.text_answer_end in completion_text:
        reward += 0.2 # 答案包含</answer>，奖励0.2

    if TrainerTools().tokenizer.text_end in completion_text:
        reward += 0.2 # 答案包含</s>，奖励0.2

    # 总奖励：0.0 ~ 3.0
    return reward


def reward_func(prompt_ids: torch.Tensor, completion_ids: torch.Tensor, answers: torch.Tensor) -> List[float]:
    # 1. 如果回答包含思考部分，则奖励1.25分
    # 2. 如果正确答案相同，则奖励1分
    # 3. 如果正确答案在回答中，则奖励0.5分

    rewards = []
    for completion_id, answer in zip(completion_ids, answers):
        completion_text = TrainerTools().tokenizer.decode_to_text(completion_id.unsqueeze(0))
        completion_text = completion_text.replace('<pad>', '').strip()
        correct_answer = TrainerTools().tokenizer.decode_to_text(answer.unsqueeze(0))

        rewards.append(get_reward(completion_text, correct_answer))

    return rewards


if __name__ == '__main__':
    init_env()

    eval_prompts = [
        f'{reasoning_system_prompt}<user>朱莉正在读一本 120 页的书。昨天，她能读12页，今天，她读的页数是昨天的两倍。如果她明天想读剩下的一半页，她应该读多少页？</s><assistant>',
        f'{reasoning_system_prompt}<user>詹姆斯从事教学工作 40 年。他的搭档教书的时间比他少了10年。他们的综合经验有多长？</s><assistant>',
        f'{reasoning_system_prompt}<user>赫克托买了一盒口香糖。他给了托德 4 个，然后他给了艾丽莎的是托德的两倍，然后他给了鲍比 5 个，比他给艾丽莎的四倍还少。如果赫克托还剩下 6 个口香糖，那么赫克托总共购买了多少个口香糖？</s><assistant>',
        f'{reasoning_system_prompt}<user>如果艾琳每周工作 40 小时，她将赚取 500 美元，并且每加班一小时即可额外获得 20 美元。如果她上周工作了 50 小时，请计算她的总收入。</s><assistant>'
    ]

    trainer = GRPOTrainer(
        train_config=get_grpo_config(),
        reward_func=reward_func,
        eval_prompts=eval_prompts
    )

    trainer.train()