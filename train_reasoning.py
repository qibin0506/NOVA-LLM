from llm_trainer import SFTTrainer
from utils import init_env, get_reasoning_config
from constant import reasoning_system_prompt

# def kd_teacher_logits_provider(inputs, attention_mask):
#     pass

# torchrun --standalone --nproc_per_node=gpu pretrain.py
if __name__ == '__main__':
    init_env()

    # <system>{reasoning_system_prompt}</s><user>你好</s><assistant><reasoning>思考</reasoning><answer>回答</answer></s>

    eval_prompts = [
        f'{reasoning_system_prompt}<user>告诉我世界上最大的湖是哪个？</s><assistant>',
        f'{reasoning_system_prompt}<user>请问今天北京天气如何？</s><assistant>',
        f'{reasoning_system_prompt}<user>哪吒和孙悟空谁更厉害？</s><assistant>',
        f'{reasoning_system_prompt}<user>保持健康的三个提示是什么？</s><assistant>',
        f'{reasoning_system_prompt}<user>你是谁？</s><assistant>'
    ]

    trainer = SFTTrainer(
        train_config=get_reasoning_config(),
        eval_prompts=eval_prompts
    )

    trainer.train()