# -*- coding: utf-8 -*-
import os
import json
import argparse
import random
import time
import distutils.util
from pprint import pprint
from functools import partial
from tqdm import tqdm
import numpy as np
import math
from datasets import load_dataset
import contextlib
from rouge import Rouge
from visualdl import LogWriter

import paddle
import paddle.nn as nn
from paddle.io import BatchSampler, DistributedBatchSampler, DataLoader
from paddlenlp.transformers import PegasusForConditionalGeneration, PegasusChineseTokenizer
from paddlenlp.transformers import LinearDecayWithWarmup
from paddlenlp.utils.log import logger
from paddlenlp.metrics import BLEU
from paddlenlp.data import DataCollatorForSeq2Seq
import matplotlib.pyplot as plt

# 通过load_dataset读取本地数据集：train.json和valid.json
train_dataset = load_dataset("json", data_files='D:\Programs\Depression\Abstract-Summarization\\Data\\train.json', split="train", encoding='utf-8-sig')
dev_dataset = load_dataset("json", data_files='D:\Programs\Depression\Abstract-Summarization\\Data\\valid.json', split="train", encoding='utf-8-sig')
test_dataset = load_dataset("json", data_files='D:\Programs\Depression\Abstract-Summarization\\Data\\test.json', split="train", encoding='utf-8-sig')
# 初始化分词器  Randeng-Pegasus-523M-Summary-Chinese-SSTIA
tokenizer = PegasusChineseTokenizer.from_pretrained('Randeng-Pegasus-523M-Summary-Chinese-SSTIA')
def convert_example(example, text_column, summary_column, tokenizer,
                    max_source_length, max_target_length):
    """
    构造模型的输入.
    """
    inputs = example[text_column]
    targets = example[summary_column]
    # 分词
    model_inputs = tokenizer(inputs,
                             max_length=max_source_length,
                             padding=False,
                             truncation=True,
                             return_attention_mask=True)
    labels = tokenizer(targets,
                       max_length=max_target_length,
                       padding=False,
                       truncation=True)
    # 得到labels，后续通过DataCollatorForSeq2Seq进行移位
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def convert_example(example, text_column, summary_column, tokenizer,
                    max_source_length, max_target_length):
    """
    构造模型的输入.
    """
    inputs = example[text_column]
    targets = example[summary_column]

    # 分词
    model_inputs = tokenizer(inputs,
                             max_length=max_source_length,
                             padding=False,
                             truncation=True,
                             return_attention_mask=True)
    labels = tokenizer(targets,
                       max_length=max_target_length,
                       padding=False,
                       truncation=True)
    # 得到labels，后续通过DataCollatorForSeq2Seq进行移位
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


# 原始字段需要移除
remove_columns = ['content', 'title']
# 文本的最大长度
max_source_length = 256
# 摘要的最大长度
max_target_length = 64
# 定义转换器
trans_func = partial(convert_example,
                     text_column='content',
                     summary_column='title',
                     tokenizer=tokenizer,
                     max_source_length=max_source_length,
                     max_target_length=max_target_length)

# train_dataset和dev_dataset分别转换
train_dataset = train_dataset.map(trans_func,
                                  batched=True,
                                  load_from_cache_file=True,
                                  remove_columns=remove_columns)
dev_dataset = dev_dataset.map(trans_func,
                              batched=True,
                              load_from_cache_file=True,
                              remove_columns=remove_columns)
test_dataset = test_dataset.map(trans_func,
                                batched=True,
                                load_from_cache_file=True,
                                remove_columns=remove_columns)


# 输出训练集的前 3 条样本
for idx, example in enumerate(dev_dataset):
    if idx < 3:
        print(example)


# Function to plot summary length distribution
import seaborn as sns
train_summary_lengths = [len(tokenizer.decode(example['labels'], skip_special_tokens=True)) for example in train_dataset]
dev_summary_lengths = [len(tokenizer.decode(example['labels'], skip_special_tokens=True)) for example in dev_dataset]
test_summary_lengths = [len(tokenizer.decode(example['labels'], skip_special_tokens=True)) for example in test_dataset]
def plot_summary_length_distribution(train_lengths, dev_lengths, test_lengths):
    plt.figure(figsize=(10, 6))
    sns.histplot(train_lengths, bins=50, kde=True, label='Train')
    sns.histplot(dev_lengths, bins=50, kde=True, label='Dev')
    sns.histplot(test_lengths, bins=50, kde=True, label='Test')
    plt.title('Summary Length Distribution')
    plt.xlabel('Summary Length')
    plt.ylabel('Count')
    plt.legend()
    plt.savefig('D:\Programs\Depression\Abstract-Summarization\\ALLbiaozhuData\summary_length_distribution.png')
    plt.show()
plot_summary_length_distribution(train_summary_lengths, dev_summary_lengths, test_summary_lengths)

# 初始化模型，也可以选择IDEA-CCNL/Randeng-Pegasus-523M-Summary-Chinese
model = PegasusForConditionalGeneration.from_pretrained('Randeng-Pegasus-523M-Summary-Chinese-SSTIA')
# 组装 Batch 数据 & Padding
batchify_fn = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

# 分布式批采样器，用于多卡分布式训练
train_batch_sampler = DistributedBatchSampler(
    train_dataset, batch_size=12, shuffle=True)

# 构造训练Dataloader
train_data_loader = DataLoader(dataset=train_dataset,
                               batch_sampler=train_batch_sampler,
                               num_workers=0,
                               collate_fn=batchify_fn,
                               return_list=True)

dev_batch_sampler = BatchSampler(dev_dataset,
                                 batch_size=12,
                                 shuffle=False)
# 构造验证Dataloader
dev_data_loader = DataLoader(dataset=dev_dataset,
                             batch_sampler=dev_batch_sampler,
                             num_workers=0,
                             collate_fn=batchify_fn,
                             return_list=True)

test_batch_sampler = BatchSampler(test_dataset,
                                 batch_size=12,
                                 shuffle=False)
# 构造验证Dataloader
test_data_loader = DataLoader(dataset=test_dataset,
                             batch_sampler=test_batch_sampler,
                             num_workers=0,
                             collate_fn=batchify_fn,
                             return_list=True)


# 学习率预热比例
warmup = 0.02
# 学习率
learning_rate = 5e-5
# 训练轮次
num_epochs = 10
# 训练总步数
num_training_steps = len(train_data_loader) * num_epochs
# AdamW优化器参数epsilon
adam_epsilon = 1e-6
# AdamW优化器参数weight_decay
weight_decay=0.01
# 训练中，每个log_steps打印一次日志
log_steps = 1
# 训练中，每隔eval_steps进行一次模型评估
eval_steps = 1000
# 摘要的最小长度
min_target_length = 0
# 训练模型保存路径
output_dir = 'D:\Programs\Depression\Abstract-Summarization\\ALLbiaozhuData\checkpoints'
# 解码beam size
num_beams = 4

log_writer = LogWriter('D:\\Programs\\Depression\\Abstract-Summarization\\ALLbiaozhuData\\visualdl_log_dir')
lr_scheduler = LinearDecayWithWarmup(learning_rate, num_training_steps, warmup)

# LayerNorm参数不参与weight_decay
decay_params = [
    p.name for n, p in model.named_parameters()
    if not any(nd in n for nd in ["bias", "norm"])
]
# 优化器AdamW
optimizer = paddle.optimizer.AdamW(
    learning_rate=lr_scheduler,
    beta1=0.9,
    beta2=0.999,
    epsilon=adam_epsilon,
    parameters=model.parameters(),
    weight_decay=weight_decay,
    apply_decay_param_fun=lambda x: x in decay_params)

# 计算训练评估参数Rouge-1，Rouge-2，Rouge-L，BLEU-4
def compute_metrics(preds, targets):
    assert len(preds) == len(targets), (
        'The length of pred_responses should be equal to the length of '
        'target_responses. But received {} and {}.'.format(
            len(preds), len(targets)))
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
    print('rouge-1:', round(rouge1*100, 2))
    print('rouge-2:', round(rouge2*100, 2))
    print('rouge-L:', round(rougel*100, 2))
    print('BLEU-4:', round(bleu4*100, 2))
    return rouge1, rouge2, rougel, bleu4

# 模型评估函数
@paddle.no_grad()
def evaluate(model, data_loader, tokenizer, min_target_length,
             max_target_length):
    model.eval()
    all_preds = []
    all_labels = []
    model = model._layers if isinstance(model, paddle.DataParallel) else model
    for batch in tqdm(data_loader, total=len(data_loader), desc="Eval step"):
        labels = batch.pop('labels').numpy()
        # 模型生成
        preds = model.generate(input_ids=batch['input_ids'],
                               attention_mask=batch['attention_mask'],
                               min_length=min_target_length,
                               max_length=max_target_length,
                               diversity_rate='beam_search',
                               num_beams=num_beams,
                               use_cache=True)[0]
        # tokenizer将id转为string
        all_preds.extend(
            tokenizer.batch_decode(preds.numpy(),
                                   skip_special_tokens=True,
                                   clean_up_tokenization_spaces=False))
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        all_labels.extend(
            tokenizer.batch_decode(labels,
                                   skip_special_tokens=True,
                                   clean_up_tokenization_spaces=False))
    rouge1, rouge2, rougel, bleu4 = compute_metrics(all_preds, all_labels)
    model.train()
    return rouge1, rouge2, rougel, bleu4


# 存储每个周期的验证集评估指标
rouge1_scores = []
rouge2_scores = []
rougel_scores = []
bleu4_scores = []
# 存储每个周期的测试集评估指标
rouge1_test_scores = []
rouge2_test_scores = []
rougel_test_scores = []
bleu4_test_scores = []

def train(model, train_data_loader, dev_data_loader, test_data_loader):
    global_step = 0
    best_rougel = 0
    tic_train = time.time()

    for epoch in range(num_epochs):
        for step, batch in enumerate(train_data_loader):
            global_step += 1
            # 模型前向训练，计算loss
            _, _, loss = model(**batch)
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.clear_grad()

            if global_step % log_steps == 0:
                logger.info(
                    "global step %d/%d, epoch: %d, batch: %d, rank_id: %s, loss: %f, lr: %.10f, speed: %.4f step/s"
                    % (global_step, num_training_steps, epoch, step,
                       paddle.distributed.get_rank(), loss, optimizer.get_lr(),
                       log_steps / (time.time() - tic_train)))
                log_writer.add_scalar("train_loss", loss.numpy(), global_step)
                tic_train = time.time()

        # 每个epoch结束后执行评估
        rouge1, rouge2, rougel, bleu4 = evaluate(model, dev_data_loader, tokenizer,
                                                 min_target_length, max_target_length)

        # 存储评估指标
        rouge1_scores.append(rouge1)
        rouge2_scores.append(rouge2)
        rougel_scores.append(rougel)
        bleu4_scores.append(bleu4)

        logger.info(f"Epoch {epoch + 1} evaluation results:")
        logger.info("rouge-1: %.2f, rouge-2: %.2f, rouge-L: %.2f, BLEU-4: %.2f" % (
        rouge1 * 100, rouge2 * 100, rougel * 100, bleu4 * 100))

        rouge1_test, rouge2_test, rougel_test, bleu4_test = evaluate(model, test_data_loader, tokenizer,
                                                                     min_target_length, max_target_length)
        # 存储评估指标
        rouge1_test_scores.append(rouge1_test)
        rouge2_test_scores.append(rouge2_test)
        rougel_test_scores.append(rougel_test)
        bleu4_test_scores.append(bleu4_test)
        print("\nTest set evaluation results:")
        print('rouge-1:', round(rouge1_test * 100, 2))
        print('rouge-2:', round(rouge2_test * 100, 2))
        print('rouge-L:', round(rougel_test * 100, 2))
        print('BLEU-4:', round(bleu4_test * 100, 2))

        if best_rougel < rougel:
            best_rougel = rougel
            if paddle.distributed.get_rank() == 0:
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                # Need better way to get inner model of DataParallel
                model_to_save = model._layers if isinstance(
                    model, paddle.DataParallel) else model
                model_to_save.save_pretrained(output_dir)
                tokenizer.save_pretrained(output_dir)
    # Evaluate on the test set
    # rouge1_test, rouge2_test, rougel_test, bleu4_test = evaluate(model, test_data_loader, tokenizer,min_target_length, max_target_length)
    # print("\nTest set evaluation results:")
    # print('rouge-1:', round(rouge1_test * 100, 2))
    # print('rouge-2:', round(rouge2_test * 100, 2))
    # print('rouge-L:', round(rougel_test * 100, 2))
    # print('BLEU-4:', round(bleu4_test * 100, 2))

# 调用模型训练
train(model, train_data_loader, dev_data_loader, test_data_loader)




