import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from llm_trainer import SFTTrainer
from utils import init_env, get_vlm_train_config
from constant import system_prompt

# 使用sft的方式预训练，不mask prompt部分
if __name__ == '__main__':
    init_env()

    eval_prompts = [
        f'{system_prompt}<user>简要, 清晰地说明所显示的图片.<image></s>', # 球迷在比赛的第八局击球时,因运动队得分的次数而干扰球的次数
        f'{system_prompt}<user>总结图像的视觉内容。<image></s>', # 传统装饰的花卉帕斯利班达纳。
        f'{system_prompt}<user>分享提供图像的简要解释。<image></s>', # 它涉及到船只和时间表
        f'{system_prompt}<user>绘制清晰简洁的图片摘要.<image></s>', # 如何用人画墙壁画!
        f'{system_prompt}<user><image>写一篇简短但有益的图片摘要.</s>' # 那天有很多文化活动
    ]

    eval_image_tags = [3, 132, 277, 568, 657]

    trainer = SFTTrainer(
        train_config=get_vlm_train_config(is_sft=False),
        eval_prompts=eval_prompts,
        eval_image_tags=eval_image_tags
    )

    trainer.train()