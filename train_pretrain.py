from llm_trainer import Trainer
from utils import init_env, get_pretrain_config

# def kd_teacher_logits_provider(inputs, attention_mask):
#     pass

# torchrun --standalone --nproc_per_node=gpu pretrain.py
if __name__ == '__main__':
    init_env()
    eval_prompts = [
        '请问今天北京天气如何？',
        '告诉我世界上最大的湖是哪个？',
        '介绍一下上海'
    ]

    trainer = Trainer(
        train_config=get_pretrain_config(),
        eval_prompts=eval_prompts
    )

    trainer.train()