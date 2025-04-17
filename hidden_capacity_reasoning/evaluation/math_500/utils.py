from lm_eval.tasks.hendrycks_math.utils import strip_string, remove_boxed, is_equiv


def dataset_answer_filter(answer):
    answer = strip_string(answer)
    replace_items = []
    for item in replace_items:
        answer = answer.replace(item, "")

    answer = "".join(answer.split(" "))

    return answer


import re


def model_answer_filter(answer):
    try:
        if "</think>" in answer:
            answer = answer.split("</think>")[1]
            # answer = re.search(r"\\boxed\{.*\}", answer)
            answer = re.search(r"\\boxed\{.*\}\}|\\boxed\{.*\}", answer)
            answer = remove_boxed(answer.group(0))
            answer = dataset_answer_filter(answer)
        else:
            answer = "error"
    except Exception as e:
        print(e)
        answer = "error"
    return answer


# gold_answer = dataset_answer_filter(dataset["test"][test_num]["answer"])
# model_answer = model_answer_filter(model_responce)
# print(gold_answer, model_answer, is_equiv(gold_answer, model_answer))