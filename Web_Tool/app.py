# -*- coding: utf-8 -*-
import re
import os
import subprocess
from collections import OrderedDict

from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for, Response

app = Flask(__name__, static_folder='static')

def categorize_results(prediction_results):
    categories = {}
    for result in prediction_results:
        sentences = result.split('\n')
        for sentence in sentences:
            if sentence.startswith('level 1 :'):
                level_1 = sentence.split(':')[1].strip()
                if level_1 not in categories:
                    categories[level_1] = []
                categories[level_1].append(sentences[0].split(': ')[1])  # 将 text 作为分类的关键信息
        # 对分类结果进行排序
    sorted_categories = OrderedDict()
    desired_order = ["触发性事件", "认知歪曲", "情绪反应", "驳斥"]
    for category in desired_order:
        if category in categories:
            sorted_categories[category] = categories[category]
    # 将剩余的分类结果按原顺序添加到排序后的结果中
    for category, sentences in categories.items():
        if category not in sorted_categories:
            sorted_categories[category] = sentences

    return sorted_categories
    # return categories

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    uploaded_file = request.files['file']
    if uploaded_file.filename != '':
        # 保存上传的文件
        file_path = os.path.join('data', uploaded_file.filename)
        uploaded_file.save(file_path)

        # 分句处理
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        sentences = re.split(r'(?<=[。？！])', text)

        # 保存处理后的文件
        processed_file_path = os.path.join('data', 'processed_' + uploaded_file.filename)
        with open(processed_file_path, 'w', encoding='utf-8') as file:
            for sentence in sentences:
                file.write(sentence.strip() + '\n')

        # 进行预测
        result = subprocess.run([
            "python", "Classification.py",
            "--device", "cpu",
            "--dataset_dir", "data",
            "--params_path", "./biaozhuNewDataset/checkpoint/",
            "--max_seq_length", "128",
            "--batch_size", "32",
            "--data_file", 'processed_' + uploaded_file.filename  # 使用处理后的文件进行预测
        ], capture_output=True, text=True)

        if result.stdout:
            print("子进程的标准输出:")
            print(result.stdout)

            # 使用正则表达式去除 ANSI 转义序列
            clean_result = re.sub(r'\x1b\[[0-9;]*m', '', result.stdout)

            if clean_result.endswith('\n'):
                clean_result = clean_result[:-1]

            # 保存预测结果
            prediction_result_path = os.path.join('data', 'prediction_result.txt')
            with open(prediction_result_path, 'w', encoding='utf-8') as file:
                file.write(clean_result)

            # 对预测结果进行分类
            categorized_results = categorize_results(clean_result.split('--------------------\n'))

            # 将分类后的结果保存到 categorized_result.txt 文件中
            categorized_result_path = 'data/categorized_result.txt'
            with open(categorized_result_path, 'w', encoding='utf-8') as file:
                for category, sentences in categorized_results.items():
                    file.write(f"{category}:\n")
                    file.write('\n'.join(sentences))
                    file.write('\n--------------------\n')
            print("Categorized results:", categorized_results)  # 添加打印语句，输出分类后的结果

            # 生成摘要并保存到文件
            summary_file_path = 'data/summary.txt'
            with open(summary_file_path, 'w', encoding='utf-8') as summary_file:
                for category, sentences in categorized_results.items():
                    # 将类别中的所有文本合并成一个字符串
                    text = ' '.join(sentences)
                    print(text)
                    # 生成摘要
                    # 调用 summary.py 脚本生成摘要
                    summary_process = subprocess.run([
                        "python", "summary.py",
                        "--text", text  # 传递文本作为参数给 summary.py 脚本
                    ], capture_output=True, text=True)
                    # 将摘要写入文件
                    print(summary_process.stdout)
                    if summary_process.stdout:
                        # 使用正则表达式去除 ANSI 转义序列
                        summary = re.sub(r'\x1b\[[0-9;]*m', '', summary_process.stdout)
                        # 如果去除 ANSI 转义序列后为空，则不写入文件
                        if summary.strip():
                            # 将摘要写入文件
                            summary_file.write(f"{category}:\n")
                            summary_file.write(summary)
                            # summary_file.write('\n--------------------\n')

            return clean_result
        else:
            return jsonify({'error': 'No prediction result'})
    else:
        return jsonify({'error': 'No file uploaded'})


@app.route('/result')
def view_result():
    # 读取 prediction_result.txt 文件内容
    prediction_result_path = 'data/prediction_result.txt'
    with open(prediction_result_path, 'r', encoding='utf-8') as file:
        prediction_results = file.read().strip().split('--------------------\n')

    # 读取 categorized_result.txt 文件内容
    categorized_results = categorize_results(prediction_results)
    print(categorized_results)

    # 读取 summary.txt 文件内容
    summary_results = {}
    summary_file_path = 'data/summary.txt'
    with open(summary_file_path, 'r', encoding='utf-8') as summary_file:
        categories = ['触发性事件:', '认知歪曲:', '情绪反应:', '驳斥:']  # 类别列表
        for line in summary_file:
            line = line.strip()
            if line in categories:  # 如果行是类别名称之一
                category = line
                summary_results[category] = []  # 初始化空列表
            elif category is not None:  # 如果已经确定了类别
                summary_results[category].append(line)

    print("summary_results的结果是：",summary_results)

    return render_template('result.html', prediction_results=prediction_results,
                           categorized_results=categorized_results, summary_results=summary_results)

@app.route('/download')
def download_file():
    # 读取原始预测结果
    prediction_result_path = 'data/prediction_result.txt'
    with open(prediction_result_path, 'r', encoding='utf-8') as file:
        prediction_result = file.read().strip()

    categorized_result_path = 'data/categorized_result.txt'
    with open(categorized_result_path, 'r', encoding='utf-8') as file:
        categorized_results = file.read()

    print('AAA是',categorized_results)

    summary_file_path = 'data/summary.txt'
    with open(summary_file_path, 'r', encoding='utf-8') as file:
        summary_results = file.read()

    # 合并原始预测结果和分类后的结果以及摘要后的结果
    combined_result = f"原始预测结果:\n{prediction_result}\n{'*' * 100}\n根据Level分类为ABCD的结果:\n{categorized_results}\n{'*' * 100}\n患者认知路径抽取结果:\n{summary_results}"
    # 将合并后的结果写入临时文件
    temp_file_path = 'data/combined_result.txt'
    with open(temp_file_path, 'w', encoding='utf-8') as file:
        file.write(combined_result)
    return send_file(temp_file_path, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)