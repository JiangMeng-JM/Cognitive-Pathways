# -*- coding: utf-8 -*-

from rouge import Rouge
from paddlenlp.metrics import BLEU
import numpy as np


def compute_metrics(preds, targets):
    assert len(preds) == len(targets), (
        'The length of preds should be equal to the length of '
        'targets. But received {} and {}.'.format(len(preds), len(targets)))

    rouge = Rouge()
    bleu4 = BLEU(n_size=4)
    scores = []

    for pred, target in zip(preds, targets):
        try:
            score = rouge.get_scores(' '.join(pred), ' '.join(target))
            scores.append([
                score[0]['rouge-1']['f'], score[0]['rouge-2']['f'],
                score[0]['rouge-l']['f']
            ])
        except ValueError:
            scores.append([0, 0, 0])
        bleu4.add_inst(pred, [target])

    rouge1 = np.mean([i[0] for i in scores])
    rouge2 = np.mean([i[1] for i in scores])
    rougel = np.mean([i[2] for i in scores])
    bleu4 = bleu4.score()

    print('\n' + '*' * 15)
    print('The auto evaluation result is:')
    print('rouge-1:', round(rouge1 * 100, 2))
    print('rouge-2:', round(rouge2 * 100, 2))
    print('rouge-L:', round(rougel * 100, 2))
    print('BLEU-4:', round(bleu4 * 100, 2))

    return rouge1, rouge2, rougel, bleu4


import json

def load_data_from_json(file_path):
    with open(file_path, 'r', encoding='utf-8-sig') as file:
        data = json.load(file)
    return data

def evaluate_from_json(file_path):
    data = load_data_from_json(file_path)

    preds = [item["title"] for item in data]
    targets = [item["content"] for item in data]

    rouge1_test, rouge2_test, rougel_test, bleu4_test = compute_metrics(preds, targets)
    return rouge1_test, rouge2_test, rougel_test, bleu4_test

# Replace 'json_file_path' with the actual path to your JSON file
json_file_path = ''

rouge1_test, rouge2_test, rougel_test, bleu4_test = evaluate_from_json(json_file_path)

