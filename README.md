# Cognitive-Pathways---Deep-Learning
This is the repository for the paper: Exploring Cognitive Pathways through Deep Learning: A Study on Chinese Social Media. For more information, please see the paper. 
* Paper link: https://arxiv.org/abs/***


## Introduction:
This repository contains all the data and code necessary for replicating the experiments in the paper. Additionally, it provides a set of tools for implementing automatic extraction of cognitive pathways. You can package it into an executable (.exe) for use by non-professionals.
The directory structure is as follows:
* **Data**:
**Source data: This folder contains the source of our dataset.
**Labeled data: After filtering and processing the source data, it is imported into the Doccano annotation tool for labeling. The folder contains the exported JSON files after annotation.
**Abstractive summarization: This folder contains the data required for the Abstractive Summarization model in deep learning, which is obtained by further processing labeled data.
**Hierarchical text classification: This folder contains the data required for the Hierarchical text classification model in deep learning, which is obtained by further processing labeled data.
**LLMEvaluate: 
  code for LLM test


* **Deep_learning**:



* **LLMEvaluate**:



* **Web_Tool**:





* **Dependency**:
```
PyTorch==1.4.0, sklearn, tqdm, transformers  
```

## Questions:
If you have any questions, comments, suggestions, or issues with using this repository, please send a note to **palpitatejiang@163.com** . 

## Citation:
If you use this material, we would appreciate if you could cite the following reference:
