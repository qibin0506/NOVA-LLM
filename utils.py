import torch
from llm_trainer import TrainerTools, FileDataset, train_configs
from llm_model import ModelConfig, VLMConfig, RoPEConfig, MoEConfig
import os
from PIL import Image
import csv
from constant import *

from transformers import AutoProcessor, SiglipVisionModel


class ListFileDataset(FileDataset):
    def __init__(self, files):
        self.files = files

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx) -> str:
        return self.files[idx]


def init_env():
    #  Of the allocated memory 33.98 GiB is allocated by PyTorch,
    #  and 8.89 GiB is reserved by PyTorch but unallocated.
    #  If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.
    #  See documentation for Memory Management
    #  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    os.environ['TOKENIZERS_TYPE'] = 'zh_llama'  # or qwen
    os.environ['TOKEN_DIR'] = './tokens/'

    os.environ['LOG_DIR'] = './log/'

    os.environ['ENABLE_DCP'] = '1'
    os.environ['DIST_CHECKPOINT_DIR'] = 'ckpt_dir'
    os.environ['CHECKPOINT_NAME'] = 'ckpt.pth'
    os.environ['EVAL_CHECKPOINT_NAME'] = 'eval_ckpt.pth'

    # os.environ['DTYPE'] = 'float32'


def get_model_config():
    return ModelConfig(
        vocab_size=TrainerTools().tokenizer.vocab_size,
        hidden_size=768,
        intermediate_size=2048,
        moe_intermediate_size=1024,
        moe_n_dense_layer=1,
        num_hidden_layers=24,
        num_attention_heads=12,
        num_key_value_heads=2,
        max_position_embeddings=1024,
        attention_implementation='auto',
        rope_config=RoPEConfig(
            rope_theta=1e6
        ),
        moe_config=MoEConfig(
            num_experts_per_tok=2,
            n_routed_experts=8,
            n_shared_experts=1,
            aux_loss_alpha=0.1,
            seq_aux=True,
            norm_topk_prob=True
        )
    )


def get_vision_tower():
    model: torch.nn.Module = SiglipVisionModel.from_pretrained("google/siglip-base-patch16-224")
    model.to(device=TrainerTools().parallel.device, dtype=TrainerTools().dtype)

    for param in model.parameters():
        param.requires_grad = False
    model.eval()

    def vision_tower(pixel_values: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            outputs = model(pixel_values=pixel_values)
            # (1, 196, 768)
            last_hidden_state = outputs.last_hidden_state.to(pixel_values.dtype)

        return last_hidden_state

    return vision_tower


def get_vlm_config():
    assert int(image_size // patch_size) == 14
    assert int(image_size // patch_size) // int(tokens_per_image**0.5) > 0

    return VLMConfig(
        image_tok=TrainerTools().tokenizer.image,
        image_size=image_size,
        patch_size=patch_size,
        tokens_per_image=tokens_per_image,
        vision_hidden_size=768,
        vision_tower=get_vision_tower(),
        vocab_size=TrainerTools().tokenizer.vocab_size,
        hidden_size=768,
        intermediate_size=2048,
        moe_intermediate_size=1024,
        moe_n_dense_layer=1,
        num_hidden_layers=24,
        num_attention_heads=12,
        num_key_value_heads=2,
        max_position_embeddings=1024,
        attention_implementation='auto',
        rope_config=RoPEConfig(
            rope_theta=1e6
        ),
        moe_config=MoEConfig(
            num_experts_per_tok=2,
            n_routed_experts=8,
            n_shared_experts=1,
            aux_loss_alpha=0.1,
            seq_aux=True,
            norm_topk_prob=True
        )
    )

def _get_train_config(
        n_epochs: int,
        train_reasoning_model: bool,
        is_sft: bool,
        is_dpo: bool,
        is_grpo: bool,
        real_batch_size: int,
        file_dataset: FileDataset,
        model_config: ModelConfig
):
    desire_batch_size = real_batch_size * 3
    gradient_accumulation_steps = desire_batch_size // real_batch_size
    eval_batch_interval = 10 if is_grpo else 100

    ds_config = train_configs.DsConfig(zero_config=train_configs.DsZero3Config())

    loss_config = train_configs.LossConfig(
        critical_tokens=[
            TrainerTools().tokenizer.reasoning_start,
            TrainerTools().tokenizer.reasoning_end,
            TrainerTools().tokenizer.answer_start,
            TrainerTools().tokenizer.answer_end
        ],
        critical_alpha=10.0
    ) if train_reasoning_model else train_configs.LossConfig()

    dpo_config = train_configs.DPOConfig(
        loss_beta=0.1,
        loss_label_smoothing=0.0,
        nll_loss_coef=0.2
    ) if is_dpo else None

    grpo_config = train_configs.GRPOConfig(
        grpo_steps=1,
        clip_eps=0.1,
        kl_weight=0.04,
        group_size=16,
        gen_max_new_tokens=500,
        gen_temperature=0.7,
        gen_k=10,
        gen_p=0.5,
        gen_suppress_tokens=None,
    ) if is_grpo else None

    lr_mul = TrainerTools().parallel.world_size
    min_lr_ratio = 0.1

    if is_grpo:
        # grpo all_data_size=8792
        #   train_batch_per_world=epochs*(all_data_size/batch_size/world_size)*grpo_steps
        #       =1*(8792/2/4)*1=1099
        initial_lr = 5e-6 * lr_mul
        max_lr = 1e-5 * lr_mul
        min_lr = initial_lr * min_lr_ratio
        period = 100
        period_mul = 1
        warmup_iters = 100
    elif is_dpo:
        # dpo all_data_size=207339
        # train_batch_per_world=epochs*all_data_size/batch_size/world_size/gradient_accumulation_steps
        #   =2*207339/24/4/3=1439
        initial_lr = 1e-8 * lr_mul
        max_lr = 5e-8 * lr_mul
        min_lr = initial_lr * min_lr_ratio
        period = 100
        period_mul = 1
        warmup_iters = 200
    elif train_reasoning_model:
        # reasoning_size=191917
        # train_batch_per_world=epochs*all_data_size/batch_size/world_size/gradient_accumulation_steps
        #   =2*191917/24/4/3=1332
        initial_lr = 1e-5 * lr_mul
        max_lr = 5e-5 * lr_mul
        min_lr = initial_lr * min_lr_ratio
        period = 200
        period_mul = 1
        warmup_iters = 100
    elif is_sft:
        # sft_1024_size=2274622
        # train_batch_per_world=epochs*all_data_size/batch_size/world_size/gradient_accumulation_steps
        #   =5*2274622/24/4/3=39489
        initial_lr = 1e-5 * lr_mul
        max_lr = 5e-5 * lr_mul
        min_lr = initial_lr * min_lr_ratio
        period = 1000
        period_mul = 1
        warmup_iters = 1000
    else:
        initial_lr = 1e-4 * lr_mul
        max_lr = 5e-4 * lr_mul
        min_lr = initial_lr * min_lr_ratio
        period = 5000
        period_mul = 1
        warmup_iters = 3000

    lr_config = train_configs.LrConfig(
        enable_lr_scheduler=True,
        initial_lr=initial_lr,
        max_lr=max_lr,
        min_lr=min_lr,
        period=period,
        period_mul=period_mul,
        warmup_iters=warmup_iters
    )

    data_loader_config = train_configs.DataLoaderConfig(
        data_loader_pin_memory=True,
        data_loader_num_workers=0,
        data_loader_shuffle=False,
        data_loader_drop_last=True
    )

    train_config = train_configs.TrainConfig(
        n_epochs=n_epochs,
        batch_size=real_batch_size,
        model_config=model_config,
        file_dataset=file_dataset,
        gradient_accumulation_steps=gradient_accumulation_steps,
        eval_batch_interval=eval_batch_interval,
        loss_config=loss_config,
        dpo_config=dpo_config,
        grpo_config=grpo_config,
        lr_config=lr_config,
        ds_config=ds_config,
        data_loader_config=data_loader_config,
        kd_config=None
    )

    return train_config


def get_pretrain_config():
    pretrain_data_list = [
        './data/deepctrl_long_0.pkl',
        './data/deepctrl_long_1.pkl',
        './data/deepctrl_long_2.pkl',
        './data/deepctrl_long_3.pkl',
        './data/deepctrl_long_4.pkl',
        './data/deepctrl_long_final.pkl',
        './data/deepctrl_short_0.pkl',
        './data/deepctrl_short_1.pkl',
        './data/deepctrl_short_final.pkl',
    ]

    return _get_train_config(
        n_epochs=1,
        train_reasoning_model=False,
        is_sft=False,
        is_dpo=False,
        is_grpo=False,
        real_batch_size=14,
        file_dataset=ListFileDataset(pretrain_data_list),
        model_config=get_model_config()
    )


def get_sft_config():
    return _get_train_config(
        n_epochs=2,
        train_reasoning_model=False,
        is_sft=True,
        is_dpo=False,
        is_grpo=False,
        real_batch_size=12,
        file_dataset=ListFileDataset(['./data/sft_deepctrl_short.pkl']),
        model_config=get_model_config()
    )


def get_dpo_config():
    return _get_train_config(
        n_epochs=2,
        train_reasoning_model=False,
        is_sft=False,
        is_dpo=True,
        is_grpo=False,
        real_batch_size=6,
        file_dataset=ListFileDataset(['./data/dpo.pkl']),
        model_config=get_model_config()
    )


def get_reasoning_config():
    return _get_train_config(
        n_epochs=2,
        train_reasoning_model=True,
        is_dpo=False,
        is_sft=True,
        is_grpo=False,
        real_batch_size=12,
        file_dataset=ListFileDataset(['./data/r1_mix_1024.pkl']),
        model_config=get_model_config()
    )


def get_grpo_config():
    return _get_train_config(
        n_epochs=2,
        train_reasoning_model=False,
        is_dpo=False,
        is_sft=False,
        is_grpo=True,
        real_batch_size=4,
        file_dataset=ListFileDataset(['./data/grpo.pkl']),
        model_config=get_model_config()
    )


def get_pixel_values_provider(is_sft: bool):
    processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-224")

    image_path = './data/sft_images/' if is_sft else './data/pretrain_images/'
    image_csv_file = './data/vlm_sft_images.csv' if is_sft else './data/vlm_pretrain_images.csv'

    image_names = {}
    with open(image_csv_file, 'r') as f:
        reader = csv.reader(f)
        for line in reader:
            image_names[int(line[0])] = line[1]

    def pixel_values_provider(image_tags: list[int]) -> torch.Tensor:
        images = []
        for image_tag in image_tags:
            images.append(Image.open(f'{image_path}{image_names[image_tag]}'))

        inputs = processor(images=images, return_tensors="pt").pixel_values
        return inputs

    return pixel_values_provider


def get_vlm_train_config(is_sft: bool):
    n_epochs = 4 if is_sft else 3
    real_batch_size = 20 if is_sft else 24
    all_files = ['./data/vlm_sft_data.pkl'] if is_sft else ['./data/vlm_pretrain_data.pkl']
    model_config = get_vlm_config()
    desire_batch_size = real_batch_size * 3
    gradient_accumulation_steps = desire_batch_size // real_batch_size
    eval_batch_interval = 100

    ds_config = train_configs.DsConfig(zero_config=train_configs.DsZero3Config())
    loss_config = train_configs.LossConfig()

    lr_mul = TrainerTools().parallel.world_size
    min_lr_ratio = 0.1

    initial_lr = 1e-5 * lr_mul
    max_lr = 5e-5 * lr_mul
    min_lr = initial_lr * min_lr_ratio
    period = 500
    period_mul = 1
    warmup_iters = 500

    lr_config = train_configs.LrConfig(
        enable_lr_scheduler=True,
        initial_lr=initial_lr,
        max_lr=max_lr,
        min_lr=min_lr,
        period=period,
        period_mul=period_mul,
        warmup_iters=warmup_iters
    )

    data_loader_config = train_configs.DataLoaderConfig(
        data_loader_pin_memory=True,
        data_loader_num_workers=0,
        data_loader_shuffle=False,
        data_loader_drop_last=True
    )

    train_config = train_configs.TrainConfig(
        n_epochs=n_epochs,
        batch_size=real_batch_size,
        model_config=model_config,
        file_dataset=ListFileDataset(all_files),
        mask_prompt=is_sft,
        gradient_accumulation_steps=gradient_accumulation_steps,
        eval_batch_interval=eval_batch_interval,
        loss_config=loss_config,
        dpo_config=None,
        grpo_config=None,
        lr_config=lr_config,
        ds_config=ds_config,
        data_loader_config=data_loader_config,
        kd_config=None,
        pixel_values_provider=get_pixel_values_provider(is_sft),
        init_state_dict=torch.load('dpo.bin', weights_only=True)
    )

    return train_config


# 训练过程 pretrain->sft->dpo->reasoning->grpo