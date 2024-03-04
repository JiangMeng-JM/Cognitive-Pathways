import argparse
import functools
import os

import numpy as np
import paddle
import paddle.nn.functional as F
from paddle.io import BatchSampler, DataLoader
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.metrics import recall_score
from paddlenlp.data import DataCollatorWithPadding
from paddlenlp.datasets import load_dataset
from paddlenlp.transformers import AutoModelForSequenceClassification, AutoTokenizer
from paddlenlp.utils.log import logger

# yapf: disable
parser = argparse.ArgumentParser()
parser.add_argument('--device', default="gpu", help="Select which device to evaluate model, defaults to gpu.")
parser.add_argument("--dataset_dir", required=True, type=str, help="Local dataset directory should include test.txt and label.txt")
parser.add_argument("--params_path", default="./checkpoint/", type=str, help="The path to model parameters to be loaded.")
parser.add_argument("--max_seq_length", default=128, type=int, help="The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded.")
parser.add_argument("--batch_size", default=32, type=int, help="Batch size per GPU/CPU for evaluation.")
parser.add_argument("--test_file", type=str, default="test.txt", help="test dataset file name")
parser.add_argument("--label_file", type=str, default="label.txt", help="Label file name")
parser.add_argument("--bad_case_file", type=str, default="./checkpoint/bad_case.txt", help="Bad case saving file path")
args = parser.parse_args()
# yapf: enable


def preprocess_function(examples, tokenizer, max_seq_length, label_nums, is_test=False):
    """
    Preprocess dataset
    """
    result = tokenizer(text=examples["text"], max_seq_len=max_seq_length)
    if not is_test:
        result["labels"] = [float(1) if i in examples["label"] else float(0) for i in range(label_nums)]
    return result


def read_local_dataset(path, label_list):
    """
    Read dataset file
    """
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            items = line.strip().split("\t")
            if len(items) == 0:
                continue
            elif len(items) == 1:
                sentence = items[0]
                labels = []
                label = ""
            else:
                sentence = "".join(items[:-1])
                label = items[-1]
                labels = [label_list[l] for l in label.split(",")]

            # Add print statements to debug
            print("Sentence:", sentence)
            print("Labels:", labels)

            yield {"text": sentence, "label": labels, "label_n": label}


@paddle.no_grad()
def evaluate():
    """
    Evaluate the model performance
    """
    paddle.set_device(args.device)
    # Define model & tokenizer
    if os.path.exists(args.params_path):
        model = AutoModelForSequenceClassification.from_pretrained(args.params_path)
        tokenizer = AutoTokenizer.from_pretrained(args.params_path)
    else:
        raise ValueError("The {} should exist.".format(args.params_path))

    # load and preprocess dataset
    label_path = os.path.join(args.dataset_dir, args.label_file)
    test_path = os.path.join(args.dataset_dir, args.test_file)

    label_list = {}
    label_map = {}
    label_map_dict = {}
    with open(label_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            l = line.strip()
            label_list[l] = i
            label_map[i] = l
            for ii, ll in enumerate(l.split("##")):
                if ii not in label_map_dict:
                    label_map_dict[ii] = {}
                if ll not in label_map_dict[ii]:
                    iii = len(label_map_dict[ii])
                    label_map_dict[ii][ll] = iii

    # Add print statements to debug
    print("Label List:", label_list)
    print("Label Map:", label_map)
    print("Label Map Dict:", label_map_dict)

    test_ds = load_dataset(read_local_dataset, path=test_path, label_list=label_list, lazy=False)
    trans_func = functools.partial(
        preprocess_function, tokenizer=tokenizer, max_seq_length=args.max_seq_length, label_nums=len(label_list)
    )
    test_ds = test_ds.map(trans_func)

    # batchify dataset
    collate_fn = DataCollatorWithPadding(tokenizer)
    test_batch_sampler = BatchSampler(test_ds, batch_size=args.batch_size, shuffle=False)
    test_data_loader = DataLoader(dataset=test_ds, batch_sampler=test_batch_sampler, collate_fn=collate_fn)

    model.eval()
    probs = []
    labels = []
    for batch in test_data_loader:
        label = batch.pop("labels")
        logits = model(**batch)
        labels.extend(label.numpy())
        probs.extend(F.sigmoid(logits).numpy())
    probs = np.array(probs)
    labels = np.array(labels)
    preds = probs > 0.5
    report = classification_report(labels, preds, digits=4, output_dict=True)
    accuracy = accuracy_score(labels, preds)

    labels_dict = {ii: [] for ii in range(len(label_map_dict))}
    preds_dict = {ii: [] for ii in range(len(label_map_dict))}
    for i in range(len(preds)):
        for ii in range(len(label_map_dict)):
            labels_dict[ii].append([0] * len(label_map_dict[ii]))
            preds_dict[ii].append([0] * len(label_map_dict[ii]))
        for l in test_ds.data[i]["label_n"].split(","):
            for ii, sub_l in enumerate(l.split("##")):
                labels_dict[ii][-1][label_map_dict[ii][sub_l]] = 1

        pred_n = [label_map[i] for i, pp in enumerate(preds[i]) if pp]

        for l in pred_n:
            for ii, sub_l in enumerate(l.split("##")):
                preds_dict[ii][-1][label_map_dict[ii][sub_l]] = 1

    logger.info("-----Evaluate model-------")
    logger.info("test dataset size: {}".format(len(test_ds)))
    logger.info("Accuracy in test dataset: {:.2f}%".format(accuracy * 100))
    logger.info(
        "Micro avg in test dataset: precision: {:.2f} | recall: {:.2f} | F1 score {:.2f}".format(
            report["micro avg"]["precision"] * 100,
            report["micro avg"]["recall"] * 100,
            report["micro avg"]["f1-score"] * 100,
        )
    )
    logger.info(
        "Macro avg in test dataset: precision: {:.2f} | recall: {:.2f} | F1 score {:.2f}".format(
            report["macro avg"]["precision"] * 100,
            report["macro avg"]["recall"] * 100,
            report["macro avg"]["f1-score"] * 100,
        )
    )
    for ii in range(len(label_map_dict)):
        macro_f1_score = f1_score(labels_dict[ii], preds_dict[ii], average="macro")
        micro_f1_score = f1_score(labels_dict[ii], preds_dict[ii], average="micro")
        accuracy = accuracy_score(labels_dict[ii], preds_dict[ii])
        recall = recall_score(labels_dict[ii], preds_dict[ii], average='micro')
        logger.info(
            "Level {} Label Performance: Macro F1 score: {:.2f} | Micro F1 score: {:.2f}  | Recall: {:.2f} | Accuracy: {:.2f}".format(
                ii + 1, macro_f1_score * 100, micro_f1_score * 100, recall * 100, accuracy * 100
            )
        )

    for i in label_map:
        logger.info("Class name: {}".format(label_map[i]))
        logger.info(
            "Evaluation examples in test dataset: {}({:.1f}%) | precision: {:.2f} | recall: {:.2f} | F1 score {:.2f}".format(
                report[str(i)]["support"],
                100 * report[str(i)]["support"] / len(test_ds),
                report[str(i)]["precision"] * 100,
                report[str(i)]["recall"] * 100,
                report[str(i)]["f1-score"] * 100,
            )
        )
        logger.info("----------------------------")
    bad_case_path = os.path.join(args.dataset_dir, args.bad_case_file)
    with open(bad_case_path, "w", encoding="utf-8") as f:
        f.write("Text\tLabel\tPrediction\n")
        for i in range(len(preds)):
            for p, l in zip(preds[i], labels[i]):
                if (p and l == 0) or (not p and l == 1):
                    pred_n = [label_map[i] for i, pp in enumerate(preds[i]) if pp]
                    f.write(test_ds.data[i]["text"] + "\t" + test_ds.data[i]["label_n"] + "\t" + ",".join(pred_n) + "\n")
                    break

    f.close()
    logger.info("Bad case in test dataset saved in {}".format(bad_case_path))

    return


if __name__ == "__main__":
    evaluate()
