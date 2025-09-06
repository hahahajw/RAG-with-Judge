"""
em_f1 - 实现 EM 和 token level F1 score

Author - hahahajw
Date - 2025-07-30 
"""
import os
import ujson as json
import re
import string
from collections import Counter
from loguru import logger as log
from tqdm import tqdm
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from typing import List


def normalize_answer(s) -> str:
    def remove_brackets_content(text):
        # 去掉 [xxx] 格式的内容，包括方括号本身
        return re.sub(r'\[.*?\]', '', text)

    def lower(text):
        return text.lower()

    def remove_punc(text):
        # 去掉所有标点符号
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def remove_articles(text):
        # 将 a、an、the 替换为空格
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        # 修复单词间存在多个空格 和 字符串首尾的空格
        return ' '.join(text.split())

    # return white_space_fix(remove_articles(remove_punc(lower(s))))
    return white_space_fix(remove_articles(remove_punc(lower(remove_brackets_content(s)))))


def get_tokens(normalized_s: str) -> List[str]:
    """
    获取规范化后的答案中的token列表
    Args:
        normalized_s: 规范化后的字符串

    Returns:
        List[str]: token列表
    """
    # 下载一些必须的资源
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
        nltk.download('punkt_tab')
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')

    # 分词
    tokens = word_tokenize(normalized_s)

    # 移除停用词
    stop_words = set(stopwords.words('english'))
    stop_words.remove('no')  # no 不属于停用词
    tokens = [token for token in tokens if token not in stop_words]

    return tokens


def exact_match_score(prediction, ground_truth):
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def f1_score(prediction, ground_truth):
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    ZERO_METRIC = (0, 0, 0)

    if normalized_prediction in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC
    if normalized_ground_truth in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC

    # prediction_tokens = normalized_prediction.split()
    # ground_truth_tokens = normalized_ground_truth.split()
    prediction_tokens = get_tokens(normalized_prediction)
    ground_truth_tokens = get_tokens(normalized_ground_truth)
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return ZERO_METRIC
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall


def evaluation(
        qa_res_path: str,
        eval_res_path: str
):
    """
    评估问答的结果，将结果写入 eval_res_path 中
    Args:
        qa_res_path (str): 存放问答对文件
        eval_res_path (str): 保存评估结果的文件
    """
    # qa_pairs 的类型为 List[Dict[str, str]]，其结构如下
    # qa_pairs = [
    #     {
    #         'question': '1 + 1 = ?',
    #         'ground_truth': '2',
    #         'prediction': '3'
    #     },
    #     ...
    # ]

    with open(qa_res_path, 'r', encoding='utf-8') as f:
        qa_pairs = json.load(f)
    log.info(f'加载了 {len(qa_pairs)} 个问答对')

    eval_res = []
    em, f1, precision, recall = [], [], [], []
    for qa_pair in tqdm(qa_pairs, desc="处理问答对", total=len(qa_pairs)):
        # 提取预测和真实答案
        prec = qa_pair['prediction']
        gold = qa_pair['ground_truth']

        # 计算指标
        cur_em = exact_match_score(prec, gold)
        cur_f1 = f1_score(prec, gold)

        # 将结果关联到当前问答对
        qa_pair['em'] = cur_em
        qa_pair['f1'] = cur_f1
        eval_res.append(qa_pair)

        # 汇总当前结果
        em.append(cur_em)
        f1.append(cur_f1[0])
        precision.append(cur_f1[1])
        recall.append(cur_f1[2])

    # 计算最后结果
    res = {
        'em': sum(em) / len(em) if em else 0.0,
        'f1': sum(f1) / len(f1) if f1 else 0.0,
        'precision': sum(precision) / len(precision) if precision else 0.0,
        'recall': sum(recall) / len(recall) if recall else 0.0,
        'individual_em': em,
        'individual_f1': f1,
        'individual_precision': precision,
        'individual_recall': recall
    }
    eval_res.append(res)

    # 将最后结果写入文件中
    os.makedirs(os.path.dirname(eval_res_path), exist_ok=True)
    with open(eval_res_path, 'w', encoding='utf-8') as f:
        json.dump(eval_res, f, ensure_ascii=False, indent=4)

    log.info(f"\n EM: {res['em']} \n f1: {res['f1']} \n precision: {res['precision']} \n recall: {res['recall']}")

    return
