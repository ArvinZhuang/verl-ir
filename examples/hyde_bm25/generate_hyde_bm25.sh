set -x

export VLLM_ATTENTION_BACKEND=XFORMERS

for dataset in dl19 dl20; do
CUDA_VISIBLE_DEVICES=1 python3 -m verl.trainer.main_generation \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=1 \
    data.path=./train_data/hyde-bm25/${dataset}.parquet \
    data.prompt_key=prompt \
    data.n_samples=1 \
    data.batch_size=16 \
    data.output_path=./generate_data/hyde-bm25/${dataset}.parquet \
    model.path=/scratch3/zhu042/verl-ir/checkpoints/verl_grpo_hyde_bm25/qwen2.5_7b_hyde_bm25/global_step_140/actor/huggingface \
    +model.trust_remote_code=True \
    rollout.dtype=bfloat16 \
    rollout.temperature=0.0 \
    rollout.do_sample=False \
    rollout.prompt_length=2048 \
    rollout.response_length=1024 \
    rollout.tensor_model_parallel_size=1 \
    rollout.gpu_memory_utilization=0.2
done