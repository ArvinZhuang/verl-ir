import re
import random


# def extract_solution(solution_str, method='strict'):
#     # define re to extract content in the last boxed of \\boxed{...}
#     boxed_re = re.compile(r"\\boxed{([^}]*)}")
#     boxed_str = boxed_re.findall(solution_str)
#     if len(boxed_str) > 0:
#         return boxed_str[-1]
#     else:
#         return None

def extract_solution(solution_str, method='strict'):
    pattern = r'<think>.*?</think>\s*<answer>(.*?)</answer>'
    match = re.search(pattern, solution_str, re.DOTALL)
    if match:
        return match.group(1).strip()
    else:
        return None


def compute_score(solution_str, ground_truth, method='strict', format_score=0., score=1.):
    answer = extract_solution(solution_str=solution_str, method=method)
    if answer == ground_truth:
        score = 1
    else:
        score = 0
    # in random 5% of the cases, print the solution and the ground truth
    if random.random() < 0.05:
        print(f"###[Solution]\n{solution_str}\n\nGround Truth: {ground_truth}\n###")
    return score