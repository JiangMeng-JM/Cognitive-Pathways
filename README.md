# AI-Enhanced Cognitive Behavioral Therapy: Deep Learning and Large Language Models for Extracting Cognitive Pathways from Social Media Texts


This repository contains material associated to this [paper](#Citation).

It contains:
- The pre-trained models Ernie and Pegasus used in the paper ([link](#Pre-trained-models))
- The deep learning models for extracting cognitive pathways from social media texts ([link](#Deep_learning))
- The code for extracting cognitive pathways using LLM from social media texts ([link](#LLMEvaluate))


If you use this material, we would appreciate if you could cite the following reference:
* Paper link: https://arxiv.org/abs/***
## Citation
```bibtex
Jiang M, Yu Y J, Zhao Q, et al. AI-Enhanced Cognitive Behavioral Therapy: Deep Learning and Large Language Models for Extracting Cognitive Pathways from Social Media Texts[J]. arXiv preprint arXiv:2404.11449, 2024.
```

## Pre-trained-Model:
The pre-trained Models Ernie and Pegasus used in the paper have been uploaded to the folder **`Model`**, you can also download them from the following links:
  * **Ernie**: [https://huggingface.co/nghuyong/ernie-3.0-medium-zh](https://huggingface.co/nghuyong/ernie-3.0-medium-zh)
  * **Pegasus**: [https://huggingface.co/IDEA-CCNL/Randeng-Pegasus-523M-Summary-Chinese](https://huggingface.co/IDEA-CCNL/Randeng-Pegasus-523M-Summary-Chinese)

## Deep_learning:
### Abstractive summarization
* `train.py`: Use this file to train the model.
* `predict.py`: You can use this file to load the trained model and extract the summary directly.
* `checkpoints`: This folder is my trained model for text summarization.

### Hierarchical text classification
* `train.py`: Use this file to train the model.
* `utils.py`: This file contains some utility functions.
* `evaluate.py`: You can use this python file to evaluate the performance of the model.
* `predict.py`: You can use this file to load the trained model and perform classification prediction directly.
* `checkpoints`: This folder is my trained model for hierarchical text classification.


## LLMEvaluate:
* `Cognitive_Pathways_prompt.py`: This python file is the prompt provided to the LLM to enable it to extract the cognitive pathway based on this file.
* `Classification_performance_Evaluation.py`: This python file is used to evaulate the performance of the LLM in extracting cognitive pathway from statements.
* `Summarization_performance_Evaluation.py`: This python file is used to evaulate the performance of the LLM for generating summaries.


## Related codes:
1. **PaddleNLP** :   [https://github.com/PaddlePaddle/PaddleNLP](https://github.com/PaddlePaddle/PaddleNLP)

## References:
1. Sun Y, Wang S, Feng S, et al. Ernie 3.0: Large-scale knowledge enhanced pre-training for language understanding and generation[J]. arXiv preprint arXiv:2107.02137, 2021.
2. Zhang J, Zhao Y, Saleh M, et al. Pegasus: Pre-training with extracted gap-sentences for abstractive summarization[C]//International conference on machine learning. PMLR, 2020: 11328-11339.
