import os
import datasets
import random

from verl.utils.hdfs_io import copy, makedirs
import argparse
from transformers import AutoTokenizer
import toml

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='setwise-r1')
    parser.add_argument('--hf_dataset', default='Tevatron/msmarco-passage')
    parser.add_argument('--prompt_path')
    parser.add_argument('--hdfs_dir', default=None)

    args = parser.parse_args()
    prompt = toml.load(args.prompt_path)

    data_source = args.hf_dataset
    print(f"Loading the {data_source} dataset from huggingface...", flush=True)
    dataset = datasets.load_dataset(data_source, trust_remote_code=True)['train'].shuffle(seed=42)

    # select first 100k as training data
    train_dataset = dataset.select(range(100000))
    # select last 1k as test data
    test_dataset = dataset.select(range(len(dataset) - 1000, len(dataset)))
    instruction_following = "Look into each documents carefully. Rank the documents based on their relevance to the user query. Then, pick the most relevant document identifier within \\boxed{}."

    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-3B')
    # add a row to each data item that represents a unique id
    def make_map_fn(split, prompt):

        def process_fn(example, idx):

            query = example['query']
            rel_docs = example['positive_passages']
            # ramdomly select one relevant document
            rel_doc = random.choice(rel_docs)
            rel_doc = f"{rel_doc['text']}".strip()
            random.shuffle(example['negative_passages'])
            neg_docs = example['negative_passages']

            # maximun 14 neg_docs
            neg_docs = neg_docs[:14]
            # random sample num negatives, larger number has higher probability
            nums = list(range(1, len(neg_docs) + 1))
            num = random.choices(nums, weights=nums, k=1)[0]
            neg_docs = neg_docs[:num]
            neg_docs = [f"{doc['text']}" for doc in neg_docs]
            docs = [rel_doc] + neg_docs

            # truncate documents to 256 tokens
            docs = [tokenizer.tokenize(doc)[:256] for doc in docs]
            docs = [tokenizer.convert_tokens_to_string(doc) for doc in docs]
            labels = [1] + [0] * len(neg_docs)
            indices = list(range(len(labels)))
            random.shuffle(indices)
            docs = [docs[i] for i in indices]
            labels = [labels[i] for i in indices]
            docs = [f"{prompt['doc_prefix'].format(num=i + 1)}{doc}" for i, doc in enumerate(docs)]
            docs_text = prompt['doc_separator'].join(docs)
            ground_truth = prompt['ground_truth'].format(num=labels.index(1) + 1)

            data = {
                "data_source": 'setwise-r1',
                "prompt": [
                    {'role': 'system',
                     'content': prompt['prompt_system']},
                    {
                        "role": "user",
                        "content": prompt['prompt_user'].format(query=query, docs=docs_text)
                    }
                ],
                "ability": "ranking",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": ground_truth
                },
                "extra_info": {
                    'split': split,
                    'index': idx
                }
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn('train',prompt), with_indices=True, num_proc=12)
    test_dataset = test_dataset.map(function=make_map_fn('test',prompt), with_indices=True, num_proc=12)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)