# coding: utf-8
import os

import pandas as pd
import openai
openai.api_key = 'your api_key'

def generate_cognitive_path(prompt):
    prompt1 = '认知路径可概括为以下几个部分：' \
              '1、触发性事件：是某个被个体感知到的经验事件，它不包含个体的感知活动，是一个纯粹客观的事件。' \
              '2、认知歪曲：指的是人们在面对问题、情境或事件时，由于主观体验、个人观念和思维方式的影响，产生了对现实的扭曲和失真的认知方式。这种认知歪曲可能导致情绪的负面变化和不良的心理反应。包括非此即彼、以偏概全、心理过滤、否定正面思考、妄下结论、放大和缩小、情绪化推理、应该句式、乱贴标签以及罪责归己/他等。' \
              '3、情绪反应：分为情感效应和行为效应。情感效应：指个体在面对某种情境、事件或刺激时，内心情感状态发生的变化或产生的情感体验，可以是瞬时的，也可以持续一段时间。行为效应：个体在面对特定情感、思维或认知过程时可能表现出的具体行为或行动。这些行为通常反映了个体的情感状态、应对机制和应对策略，主要为冲动攻击性行为和自伤自杀等，也可能是逃避行为和药物或滥用物质；' \
              '4、驳斥：分为习惯性驳斥和有效驳斥。习惯性驳斥是指个体在面对某种观念、信念或思维时，几乎自动地、不经思考地予以否定或拒绝的心理过程。这种反应是一种固定化的、反射性的思维模式，通常不受合理思考或证据的影响。有效驳斥是指个体对某种观念、信念或思维进行有意识、深思熟虑的反驳，通过合理的思考、分析和证据来产生积极的结果和行为的过程。' \

    role_define1 = '您现在是一个在患者自杀认知方面颇有建树的心理学家，您的任务是充分理解患者的认知路径，然后从患者的陈述中摘取到患者的认知路径。主要有以下两个步骤：' \
                   'step1：你需要先将患者的陈述分句，然后判断这一个个句子都属于认知路径中的哪个部分。如果你觉得有些句子是不属于认知路径，那你就忽略即可，你只需要将你觉得属于认知路径的句子分类到这几个部分中。' \
                   '每部分输出的都是患者陈述中的原本的句子。' \
                   '如果认知路径的某个部分在患者的陈述中没有体现，那就输出"无"即可' \
                   '请以下述格式进行回复：' \
                   '触发性事件的句子：' \
                   '认知歪曲的句子：' \
                   '情绪反应的句子：' \
                   '驳斥的句子：' \
                   'step2：请将step1的回复中属于每个部分的几个句子以患者的角度都以生成式摘要的方式生成一句话，请注意是每部分输出一句话，比如属于触发性事件的几个句子提摘要成一句，属于认知歪曲的几个句子提摘要成一句。' \
                   '如果认知路径的某个部分在患者的陈述中没有体现，那就输出"无"即可' \
                   '请以下述格式进行回复：' \
                   '触发性事件：' \
                   '认知歪曲：' \
                   '情绪反应：' \
                   '驳斥：' \

    role_define2 = '另外，在step1中对于触发性事件的分类，如果句子属于触发性事件，那我们在该句子前面用{}：标注上它属于哪一类触发性事件类型（疾病症状、社会关系、生活、学习工作、情感）' \
                   '另外，在step1中对于认知歪曲的分类，如果句子属于认知歪曲，那我们在该句子前面用{}：标注上它属于哪一类认知歪曲类型（非此即彼、以偏概全、心理过滤、否定正面思考、妄下结论、放大和缩小、情绪化推理、应该句式、乱贴标签以及罪责归己/他等。）' \
                   '另外，在step1中对于情绪反应的分类，如果句子属于情绪反应，那我们在该句子前面用{}：标注上它属于哪一类情绪反应类型（情感效应、行为效应）' \
                   '另外，在step1中对于驳斥的分类，如果句子属于驳斥，那我们在该句子前面用{}：标注上它属于哪一类驳斥类型（习惯性驳斥、有效驳斥）' \
                   '请注意，我的任务分为两个step，上述两个步骤都要按照指定格式输出，step1每部分输出的都是患者陈述中的原句子，各句子之间换行输出即可，不需要标注序号。step2：以患者的角度对每一部分的句子生成摘要（尽量简洁），然后按照指定格式输出' \

    response = openai.ChatCompletion.create(
        model="gpt-4",
        temperature=0.8,
        max_tokens=1000,
        messages=[
            {"role": "user", "content": prompt1},
            {"role": "user", "content": role_define1},
            {"role": "user", "content": role_define2},
            {"role": "user", "content": prompt}
        ]
    )

    generated_text = response['choices'][0]['message']['content']

    return generated_text

def main():
    # 读取包含患者陈述的Excel文件
    df = pd.read_excel('D:\Programs\Depression\APITest\Test4\Reddit2-output_folder\\redditTwo31.xlsx')  # 替换为您的Excel文件路径
    # 用于生成唯一文件名的计数器
    file_count =1
    output_folder = 'D:\Programs\Depression\APITest\Test4\Reddit2-CepingData'
    # 遍历每一行，提取患者陈述并生成认知路径
    for index, row in df.iterrows():
        patient_statement = row['text']
        # 打印当前进度信息
        print(f"Processing row {index + 1}/{len(df)}")
        cognitive_path = generate_cognitive_path(patient_statement)

        # 生成文件名
        output_filename = os.path.join(output_folder, f"redditTwo{file_count}.txt")

        # 保存认知路径到文件中
        with open(output_filename, 'w', encoding='utf-8') as file:
            file.write(cognitive_path)

        # 增加文件计数器
        file_count += 1


if __name__ == '__main__':
    main()
