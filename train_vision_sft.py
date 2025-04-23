import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from llm_trainer import SFTTrainer
from utils import init_env, get_vlm_train_config
from constant import system_prompt

if __name__ == '__main__':
    init_env()
    # {system_prompt}<user>这是什么图？<image>1</s><assistant>我是{assistant_name}，是一款AI大模型</s>
    eval_prompts = [
        # 把数字闹钟放在床头柜或光线充足的木桌上，会在几个方面影响睡眠质量。红色发光屏幕上显示着时间（晚上10点），发出的光可能让人分心或者引起不适，使人难以入睡。夜间暴露于哪怕少量的光线中都能干扰褪黑素的产生，而褪黑素有调节睡眠周期的作用。为了确保更好的睡眠质量，可以将时钟屏幕调暗、把它远离床摆放，或是移到房间里视线较不那么好的地方。此外，睡觉前让房间保持明亮也会扰乱自然的睡眠循环，所以关掉或调低任何不必要的灯光能够改善整体的睡眠环境。
        f'{system_prompt}<user>闹钟的位置对睡眠质量有什么影响？<image></s>',
        # 图中的男子正在进行发球，他将网球抛向空中并挥动球拍准备击球。
        f'{system_prompt}<user>照片中的男子在网球场上做什么动作？<image></s><assistant>',
        # 两个男人带着笔记本电脑坐在小隔间的餐桌旁，他们可能是在工作、学习或合作完成一个项目。他们也许正在用笔记本电脑做研究、处理文件，甚至与队友进行远程交流。这个场景暗示这里可以是咖啡馆、餐厅或者共享办公场所，而他们选择在此一起坐下来工作。当他们专注于自己的任务或是讨论有关工作和学习的话题时，这种地点的选择有助于营造一种舒适放松的环境。
        f'{system_prompt}<user>在这种情况下，这两个人使用笔记本电脑的目的是什么？<image></s>',
        # 照片中的年轻女子正在享用必胜客比萨饼和百事可乐，一边享受她的美餐并摆出各种姿势。
        f'{system_prompt}<user>照片中的年轻女子在做什么？<image></s><assistant>',
        # 厨房的设计以中央岛台为中心，周围设有水槽和椅子/凳子。这种布局允许人们在烹饪时进行社交互动，因为它提供了供人们聚集并与正在准备食物的人交谈的座位。此外，宽敞的布局、优雅的木制橱柜以及各种盆栽植物营造出一种舒适和欢迎的气氛，进一步鼓励交流和互动。靠近岛台的餐桌也确保人们可以轻松地从厨房移动到用餐区进餐并继续他们的谈话。
        f'{system_prompt}<user><image>厨房有哪些特点可以促进烹饪时的社交互动？</s>'
    ]

    eval_image_tags = [0, 3, 19, 38, 52]

    trainer = SFTTrainer(
        train_config=get_vlm_train_config(is_sft=True),
        eval_prompts=eval_prompts,
        eval_image_tags=eval_image_tags
    )

    trainer.train()