import numpy as np
from sklearn.metrics import precision_score, recall_score

import paddle
import paddle.nn.functional as F
from paddlenlp.utils.log import logger


@paddle.no_grad()
def evaluate(model, criterion, metric, data_loader):

    model.eval()
    metric.reset()
    losses = []
    y_true = []
    y_pred = []

    for batch in data_loader:
        labels = batch.pop("labels")
        logits = model(**batch)
        loss = criterion(logits, labels)
        probs = F.sigmoid(logits)
        losses.append(loss.numpy())
        metric.update(probs, labels)

        # 获取真实标签和预测标签
        y_true.extend(labels.numpy())
        y_pred.extend((probs.numpy() > 0.5).astype(int))

    f1_score = metric.accumulate()
    precision = precision_score(y_true, y_pred, average='micro')
    recall = recall_score(y_true, y_pred, average='micro')

    logger.info(
        "eval loss: %.5f, micro f1 score: %.5f, macro f1 score: %.5f, precision: %.5f, recall: %.5f"
        % (np.mean(losses), f1_score, precision, recall)
    )
    model.train()
    metric.reset()

    return f1_score, precision, recall


def preprocess_function(examples, tokenizer, max_seq_length, label_nums, is_test=False):
    
    result = tokenizer(text=examples["sentence"], max_seq_len=max_seq_length)
    # One-Hot label
    if not is_test:
        result["labels"] = [float(1) if i in examples["label"] else float(0) for i in range(label_nums)]
    return result


def read_local_dataset(path, label_list=None, is_test=False):
    """
    Read dataset
    """
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if is_test:
                items = line.strip().split("\t")
                sentence = "".join(items)
                yield {"sentence": sentence}
            else:
                items = line.strip().split("\t")
                if len(items) == 0:
                    continue
                elif len(items) == 1:
                    sentence = items[0]
                    labels = []
                else:
                    sentence = "".join(items[:-1])
                    label = items[-1]
                    labels = [label_list[l] for l in label.split(",")]
                yield {"sentence": sentence, "label": labels}
