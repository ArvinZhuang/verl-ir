import os
import datasets
import random

from tests.e2e.arithmetic_sequence.model.create_model_tokenizer import tokenizer
from verl.utils.hdfs_io import copy, makedirs
import argparse
from transformers import AutoTokenizer
import toml

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt_file')
    parser.add_argument('--local_dir', default='hyde-bm25')
    parser.add_argument('--hf_dataset', default='Tevatron/msmarco-passage')
    parser.add_argument('--hdfs_dir', default=None)

    args = parser.parse_args()

    prompt = toml.load(args.prompt_file)

    data_source = args.hf_dataset
    print(f"Loading the {data_source} dataset from huggingface...", flush=True)
    dataset = datasets.load_dataset(data_source, trust_remote_code=True)['train'].shuffle(seed=42)

    # select first 100k as training data
    train_dataset = dataset.select(range(100000))
    # select last 1k as test data
    test_dataset = dataset.select(range(len(dataset) - 1000, len(dataset)))

    # dl dataset
    dl19_dataset = datasets.load_dataset(data_source, trust_remote_code=True)['dl19']
    dl20_dataset = datasets.load_dataset(data_source, trust_remote_code=True)['dl20']

    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-3B')
    # add a row to each data item that represents a unique id
    def make_map_fn(split):

        def process_fn(example, idx):

            query = example['query']
            if len(example['positive_passages']) == 0:
                pos_docid = 'nan'
            else:
                pos_docid = example['positive_passages'][0]['docid']

            data = {
                "data_source": 'hyde-bm25',
                "prompt": [{
                    'role': 'system',
                    'content': prompt['prompt_system']},
                    {
                        "role": "user",
                        "content": prompt['prompt_user'].format(query=query),
                    }],
                "ability": "query_expansion",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": pos_docid
                },
                "extra_info": {
                    'split': split,
                    'index': idx
                }
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True, num_proc=12)
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True, num_proc=12)
    dl19_dataset = dl19_dataset.map(function=make_map_fn('dl19'), with_indices=True, num_proc=12)
    dl20_dataset = dl20_dataset.map(function=make_map_fn('dl20'), with_indices=True, num_proc=12)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))
    dl19_dataset.to_parquet(os.path.join(local_dir, 'dl19.parquet'))
    dl20_dataset.to_parquet(os.path.join(local_dir, 'dl20.parquet'))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)