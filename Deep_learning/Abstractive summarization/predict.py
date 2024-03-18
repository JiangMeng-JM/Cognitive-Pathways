# -*- coding: utf-8 -*-
import argparse

# 模型推理，针对单条文本，生成摘要
from paddlenlp.transformers import PegasusForConditionalGeneration, PegasusChineseTokenizer

# 文本的最大长度
max_source_length = 512
# 摘要的最大长度
max_target_length = 128
# 摘要的最小长度
min_target_length = 0
# 解码beam size
num_beams = 4

def infer(text, model, tokenizer):
    tokenized = tokenizer(text,
                          truncation=True,
                          max_length=max_source_length,
                          return_tensors='pd')
    preds, _ = model.generate(input_ids=tokenized['input_ids'],
                              max_length=max_target_length,
                              min_length=min_target_length,
                              decode_strategy='beam_search',
                              num_beams=num_beams)
    # print(tokenizer.decode(preds[0], skip_special_tokens=True, clean_up_tokenization_spaces=False))
    return tokenizer.decode(preds[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)


# # 加载训练好的模型
# model = PegasusForConditionalGeneration.from_pretrained('./checkpoints')    #在此替换为你的训练好的文件位置
# model.eval()
# tokenizer = PegasusChineseTokenizer.from_pretrained('./checkpoints')      #在此替换为你的训练好的文件位置
#
# # 推理
# text = ''
# infer(text, model, tokenizer)
def main():
    parser = argparse.ArgumentParser(description="Generate summary for given text")
    parser.add_argument("--text", type=str, required=True, help="Text for summarization")
    args = parser.parse_args()

    model = PegasusForConditionalGeneration.from_pretrained('./checkpoints')     #在此替换为你的训练好的文件位置
    model.eval()
    tokenizer = PegasusChineseTokenizer.from_pretrained('./checkpoints')     #在此替换为你的训练好的文件位置

    summary_text = infer(args.text, model, tokenizer)
    print(summary_text)

if __name__ == "__main__":
    main()
