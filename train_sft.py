from llm_trainer import SFTTrainer
from utils import init_env, get_sft_config
from constant import system_prompt

# def kd_teacher_logits_provider(inputs, attention_mask):
#     pass

# torchrun --standalone --nproc_per_node=gpu pretrain.py
if __name__ == '__main__':
    init_env()

    # <system>{system_prompt}</s><user>你好</s><assistant>我好</s><user>很好</s><assistant>不好</s>

    eval_prompts = [
        f'{system_prompt}<user>告诉我世界上最大的湖是哪个？</s><assistant>',
        f'{system_prompt}<user>请问今天北京天气如何？</s><assistant>',
        f'{system_prompt}<user>哪吒和孙悟空谁更厉害？</s><assistant>',
        f'{system_prompt}<user>保持健康的三个提示是什么？</s><assistant>',
        f'{system_prompt}<user>你是谁？</s><assistant>'
    ]

    trainer = SFTTrainer(
        train_config=get_sft_config(),
        eval_prompts=eval_prompts
    )

    trainer.train()