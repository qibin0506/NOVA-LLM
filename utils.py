import torch
from llm_trainer import TrainerTools, train_configs
from llm_model import ModelConfig, RoPEConfig, MoEConfig
import os

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

def _get_train_config(
        n_epochs: int,
        train_reasoning_model: bool,
        is_sft: bool,
        is_dpo: bool,
        is_grpo: bool,
        real_batch_size: int,
        all_files: list[str],
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

    lr_scheduler_config = train_configs.LrSchedulerConfig(
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
        all_files=all_files,
        gradient_accumulation_steps=gradient_accumulation_steps,
        eval_batch_interval=eval_batch_interval,
        loss_config=loss_config,
        dpo_config=dpo_config,
        grpo_config=grpo_config,
        lr_scheduler_config=lr_scheduler_config,
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
        all_files=pretrain_data_list,
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
        all_files=['./data/sft_deepctrl_short.pkl'],
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
        all_files=['./data/dpo.pkl'],
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
        all_files=['./data/r1_mix_1024.pkl'],
        model_config=get_model_config()
    )


def get_grpo_config():
    return _get_train_config(
        n_epochs=1,
        train_reasoning_model=False,
        is_dpo=False,
        is_sft=False,
        is_grpo=True,
        real_batch_size=4,
        all_files=['./data/grpo.pkl'],
        model_config=get_model_config()
    )


# 训练过程 pretrain->sft->dpo->reasoning->grpo

