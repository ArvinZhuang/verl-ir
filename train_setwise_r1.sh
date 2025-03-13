#!/bin/bash
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128g
#SBATCH --account=OD-228997
#SBATCH --gres=gpu:2
#SBATCH --job-name=verl-R1-7B
#SBATCH --output=logs/print-train_grpo_7B_setwise.txt
#SBATCH --error=logs/error-train_grpo_7B_setwise.txt
#SBATCH --qos=express


module load miniconda3
module load cuda
source activate verl-ir

export HF_HOME=cache
export HF_DATASETS_CACHE=cache
export PYSERINI_CACHE=cache/pyserini
export IR_DATASETS_HOME=cache/ir_datasets
export WANDB_ARTIFACT_LOCATION=cache/wandb
export WANDB_ARTIFACT_DIR=cache/wandb
export WANDB_CACHE_DIR=cache/wandb
export WANDB_CONFIG_DIR=cache/wandb
export WANDB_PROJECT=verl-ir
export VLLM_ATTENTION_BACKEND=XFORMERS

CUDA_VISIBLE_DEVICES=0,1

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=train_data/setwise-r1/train.parquet \
    data.val_files=train_data/setwise-r1/test.parquet \
    data.train_batch_size=256 \
    data.max_prompt_length=4096 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-7B-Instruct \
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.2 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.temperature=1.2 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='verl_grpo_setwise' \
    trainer.experiment_name='qwen2.5_7b_setwise' \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=10 \
    trainer.total_epochs=1