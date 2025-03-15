import re
import random
from pyserini.search.lucene import LuceneSearcher

searcher = LuceneSearcher.from_prebuilt_index('msmarco-passage')

def extract_solution(solution_str, method='strict'):
    pattern = r'Keywords: (.*)'
    match = re.search(pattern, solution_str, re.DOTALL)
    if match:
        return match.group(1).strip()
    else:
        return None


def compute_score(solution_str, ground_truth, method='strict', format_score=0., score=1.):
    answer = extract_solution(solution_str=solution_str, method=method)
    if answer is None:
        answer = "Dummy query"
    answer = answer.replace("<|endoftext|>", "").strip().replace('<|im_end|>', '')
    search_results = searcher.search(answer, k=100)
    search_results = [result.docid for result in search_results]
    mrr = 0
    for i, result in enumerate(search_results):
        if result == ground_truth:
            mrr = 1 / (i + 1)
            break
    score = mrr
    # in random 5% of the cases, print the solution and the ground truth
    if random.random() < 0.05:
        print(f"###[Solution]\n{solution_str}\n\nGround Truth: {ground_truth}\n\n[MRR]: {mrr}\n\n")
    return score