import pandas as pd

true_df = pd.read_excel("D:\Programs\Depression\APITest\\result_biaozhu.xlsx")
pred_df = pd.read_excel("D:\Programs\Depression\APITest\ENDresult_GPT-4.xlsx")


# Define the hierarchy levels
level_1_labels = ["触发性事件", "认知歪曲", "情绪反应", "驳斥"]
level_2_labels = ["疾病症状","社会关系","生活","学习工作","情感","非此即彼", "以偏概全", "心理过滤", "否定正面思考", "妄下结论", "放大和缩小", "情绪化推理", "应该句式", "乱贴标签", "罪责归己", "情感效应","行为效应","习惯性驳斥","有效驳斥"]

# Sort dataframes by id
true_df = true_df.sort_values(by="id")
pred_df = pred_df.sort_values(by="id")

# Extract true labels
true_labels_level_1 = true_df[level_1_labels].values
true_labels_level_2 = true_df[level_2_labels].values

# Extract predicted labels
pred_labels_level_1 = pred_df[level_1_labels].values
pred_labels_level_2 = pred_df[level_2_labels].values

# Calculate True Positives, False Positives, False Negatives, True Negatives for Level 1
TP_level_1 = ((true_labels_level_1 == 1) & (pred_labels_level_1 == 1)).sum()
FP_level_1 = ((true_labels_level_1 == 0) & (pred_labels_level_1 == 1)).sum()
FN_level_1 = ((true_labels_level_1 == 1) & (pred_labels_level_1 == 0)).sum()
TN_level_1 = ((true_labels_level_1 == 0) & (pred_labels_level_1 == 0)).sum()

# Calculate True Positives, False Positives, False Negatives, True Negatives for Level 2
TP_level_2 = ((true_labels_level_2 == 1) & (pred_labels_level_2 == 1)).sum()
FP_level_2 = ((true_labels_level_2 == 0) & (pred_labels_level_2 == 1)).sum()
FN_level_2 = ((true_labels_level_2 == 1) & (pred_labels_level_2 == 0)).sum()
TN_level_2 = ((true_labels_level_2 == 0) & (pred_labels_level_2 == 0)).sum()

# Calculate precision, recall, and micro F1 score for Level 1
precision_level_1 = TP_level_1 / (TP_level_1 + FP_level_1)
recall_level_1 = TP_level_1 / (TP_level_1 + FN_level_1)
micro_f1_level_1 = 2 * precision_level_1 * recall_level_1 / (precision_level_1 + recall_level_1)

# Calculate precision, recall, and micro F1 score for Level 2
precision_level_2 = TP_level_2 / (TP_level_2 + FP_level_2)
recall_level_2 = TP_level_2 / (TP_level_2 + FN_level_2)
micro_f1_level_2 = 2 * precision_level_2 * recall_level_2 / (precision_level_2 + recall_level_2)

print("Level 1 Metrics:")
print(f"Precision: {precision_level_1:.4f}")
print(f"Recall: {recall_level_1:.4f}")
print(f"Micro F1: {micro_f1_level_1:.4f}")

print("\nLevel 2 Metrics:")
print(f"Precision: {precision_level_2:.4f}")
print(f"Recall: {recall_level_2:.4f}")
print(f"Micro F1: {micro_f1_level_2:.4f}")

# Aggregate TP, FP, FN, TN across both levels
TP_total = TP_level_1 + TP_level_2
FP_total = FP_level_1 + FP_level_2
FN_total = FN_level_1 + FN_level_2
TN_total = TN_level_1 + TN_level_2

# Calculate precision, recall, and micro F1 score for total
precision_total = TP_total / (TP_total + FP_total)
recall_total = TP_total / (TP_total + FN_total)
micro_f1_total = 2 * precision_total * recall_total / (precision_total + recall_total)

print("Overall Metrics:")
print(f"Precision: {precision_total:.4f}")
print(f"Recall: {recall_total:.4f}")
print(f"Micro F1: {micro_f1_total:.4f}")
