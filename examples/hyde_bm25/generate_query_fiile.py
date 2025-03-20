from datasets import load_dataset
import argparse

import toml
import re

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt_file')
    parser.add_argument('--dataset_file')
    parser.add_argument('--output_path')

    args = parser.parse_args()

    prompt = toml.load(args.prompt_file)
    pattern = prompt['pattern']

    data = load_dataset('parquet', data_files = args.dataset_file)['train']

    # write query file
    with open(args.output_path, 'w') as f:
        for example in data:
            response = example['responses'][0]
            match = re.search(pattern, response, re.DOTALL)
            if match:
                query = match.group(1).strip()
                query = query.replace('\n', ' ')
            else:
                print(f'fail to match {example}')
                query = example['query']
            f.write(f"{example['query_id']}\t{query}\n")


