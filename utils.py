import torch
import os

def load_model(milnet, dataset, model, loss=None, model_weight=None):
    model_dir = 'weights'
    model_name = None
    if model == 'dsmil':    
        if dataset=='C16_dataset_c16_low99_v0': 
            weight_name = 'C16_dataset_c16_low99_v0_dsmil_1class'  
            weight_time = '20220929_183051'     # 0.86, 0.9332
            model_name = 'best.pth'
    
    elif model == 'frmil':
        if dataset=='C16_dataset_c16_low99_v0':                                          
            if loss=='FrmilLoss' or loss=='FrmilLoss2':                 
                weight_name = 'C16_dataset_c16_low99_v0_frmil_FrmilLoss_1class_thres10.17_dropout_0.2'
                weight_time = '20221103_165624'   # ACC05 0.8915 | AUC 0.9457  
    
    if model_weight is not None:
        model_path = os.path.join(model_dir, model_weight, 'best.pth')
    elif model_name is None:
        model_path = os.path.join(model_dir, weight_name, weight_time, 'best.pth')
    else:
        model_path = os.path.join(model_dir, weight_name, weight_time, model_name) 
    print("Loading the pretrain model from:", model_path)
    state_dict_weights = torch.load(model_path)    
    milnet.load_state_dict(state_dict_weights)
    return milnet

def print_result(avg_score, avg_05_score, aucs, praucs, result_type='', dataset=None): 
    if dataset=='Lung': 
        print(f"The {result_type} Model: ACC {avg_score:.4f} | ACC05 {avg_05_score:.4f} |auc_LUAD: {aucs[0]:.4f}/{praucs[0]:.4f}, auc_LUSC: {aucs[1]:.4f}/{praucs[1]:.4f}")        
    else: 
        print(f"The {result_type} Model: ACC {avg_score:.4f} | ACC05 {avg_05_score:.4f} | AUC {aucs[0]:.4f}/{praucs[0]:.4f}")
        
        
from sklearn.metrics import accuracy_score, precision_score, roc_auc_score
from sklearn.metrics import f1_score, recall_score, roc_curve
from prettytable import PrettyTable

def multi_class_scores_mtl(gt_logtis, pred_logits, class_labels, threshold):
    """
    参数：
        gt_logtis (list): [N, num_class], 真实标签
        pred_logits (tensor): [N, num_class], 每个样本的类别概率
        class_labels (list): ['NILM', 'ASC-US', 'LSIL', 'ASC-H', 'HSIL', 'AGC', 'BV', 'M', 'T'],  类别标签, 
        wsi_names (list): N, WSI名称,方便打印错误信息
    返回：
        roc_auc_macro (ndarray): 多类别ROC_AUC
        accuracy (float): Micro 准确率
        recall (ndarray): Macro 阳性召回率 len = 8 or 4 
        precision (ndarray): Macro 阳性精确率
        fscore (ndarray): Macro F1分数
    TODO 目前对于多类别任务，只考虑了1,5,3的多分类划分方式以及5分类的单任务模式
    """
    # 对于多类别样本 拆分成多个样本，预测概率将正确的其他类别概率设为0
    
    assert len(class_labels) == 5 or len(class_labels) == 10 or len(class_labels) == 9
    bag_labels = []
    new_pred_logits = []
    
    for i, gt_logit in enumerate(gt_logtis):
        gt_labels = np.where(gt_logit == 1)[0]
        if len(gt_labels) > 1:
            for gt_label in gt_labels:
                pred_logit = copy.deepcopy(pred_logits[i])
                pred_logit[gt_logit == 1] = 0
                bag_labels.append(gt_label)
                pred_logit[gt_label] = pred_logits[i][gt_label]
                new_pred_logits.append(pred_logit)
        else:
            bag_labels.append(gt_labels[0])
            new_pred_logits.append(pred_logits[i])
            
    bag_labels = np.array(bag_labels)
    bag_logits = np.array(new_pred_logits)
    
    # 对于宫颈癌症风险和微生物感染任务 分开计算指标
    if len(class_labels) in [9, 10]:
        bag_labels_cancer, bag_labels_microbial = bag_labels[bag_labels < 6], bag_labels[bag_labels >= 6]
        bag_logits_cancer, bag_logtis_microbial = bag_logits[bag_labels < 6, :6], bag_logits[bag_labels >= 6, 6:]
        class_labels_cancer, class_labels_microbial = class_labels[:6], class_labels[6:]
    else:
        bag_labels_cancer, bag_logits_cancer, class_labels_cancer = bag_labels, bag_logits, class_labels
 
    roc_auc = []
    # 首先评估宫颈癌症风险
    n_cancer_class = len(class_labels_cancer)
    n_cancer_sample = bag_labels_cancer.shape[0]
    bag_labels_cancer_onehot = np.eye(n_cancer_class)[bag_labels_cancer]
    bag_pred_cancer_onehot = np.zeros_like(bag_logits_cancer)
    for i in range(1, n_cancer_class):
        roc_auc.append(roc_auc_score(bag_labels_cancer_onehot[:, i], bag_logits_cancer[:, i]))
        bag_pred_cancer_onehot[:, i] = bag_logits_cancer[:, i] >= threshold
    # print(bag_pred_cancer_onehot.shape)
    for j in range(bag_pred_cancer_onehot.shape[0]):
        if np.sum(bag_pred_cancer_onehot[j]) == 0:
            bag_pred_cancer_onehot[j, 0] = 1
        elif np.sum(bag_pred_cancer_onehot[j]) > 1:
            # 多个类别都大于阈值，则保留最大风险的类别
            bag_pred_cancer_onehot[j] = 0
            bag_pred_cancer_onehot[j, np.argmax(bag_logits_cancer[j, 1:])+1] = 1
            # 如果该类别被判定为NILM的概率过高 也输出错误信息
            # if bag_logits_cancer[j, 0] > 0.95:
            #     print(f'[ERROR] {wsi_names[j]} risk prediction is wrong: {[round(risk, 4) for risk in bag_logits_cancer[j]]}')
    
    bag_pred_cancer = np.argmax(bag_pred_cancer_onehot, axis=-1) # [N_cancer,]
    accuracy = accuracy_score(bag_labels_cancer, bag_pred_cancer)
    recalls = recall_score(bag_labels_cancer, bag_pred_cancer, average=None, labels=list(range(1,n_cancer_class)))
    precisions = precision_score(bag_labels_cancer, bag_pred_cancer, average=None, labels=list(range(1,n_cancer_class)))
    fscores = f1_score(bag_labels_cancer, bag_pred_cancer, average=None, labels=list(range(1,n_cancer_class)))
    print('[INFO] confusion matrix for cancer labels:')
    cancer_matrix = confusion_matrix(bag_labels_cancer, bag_pred_cancer, class_labels_cancer)
    print('fscores len' + str(len(fscores)))
    
    # 评估微生物感染
    microbial_matrix = None
    if len(class_labels) in [9, 10]:
        n_microbial_class = len(class_labels_microbial)
        n_microbial_sample = bag_labels_microbial.shape[0]
        bag_labels_microbial = bag_labels_microbial - 6
        bag_labels_microbial_onehot = np.eye(n_microbial_class)[bag_labels_microbial]
        bag_pred_microbial_onehot = np.zeros_like(bag_logtis_microbial)
        for i in range(n_microbial_class):
            roc_auc.append(roc_auc_score(bag_labels_microbial_onehot[:, i], bag_logtis_microbial[:, i]))
            bag_pred_microbial_onehot[:, i] = bag_logtis_microbial[:, i] >= threshold
        for j in range(bag_pred_microbial_onehot.shape[0]):
            if np.sum(bag_pred_microbial_onehot[j]) == 0:
                bag_pred_microbial_onehot[j, 0] = 1
            elif np.sum(bag_pred_microbial_onehot[j]) > 1:
                # 多个类别都大于阈值，则保留最大风险的类别
                bag_pred_microbial_onehot[j] = 0
                bag_pred_microbial_onehot[j, np.argmax(bag_logtis_microbial[j,])] = 1
        bag_pred_microbial = np.argmax(bag_pred_microbial_onehot, axis=-1) # [N,]
        # print(recalls, recalls.shape, type(recalls))
        recalls2 = recall_score(bag_labels_microbial, bag_pred_microbial, average=None, labels=list(range(n_microbial_class)))
        precisions2 = precision_score(bag_labels_microbial, bag_pred_microbial, average=None, labels=list(range(n_microbial_class)))
        fscores2 = f1_score(bag_labels_microbial, bag_pred_microbial, average=None, labels=list(range(n_microbial_class)))
        # print(recalls2, recalls2.shape, type(recalls2))
        recalls, precisions, fscores = np.concatenate((recalls, recalls2)), np.concatenate((precisions, precisions2)), np.concatenate((fscores, fscores2))
        print('[INFO] confusion matrix for microbial labels:')
        microbial_matrix = confusion_matrix(bag_labels_microbial, bag_pred_microbial, class_labels_microbial)
        accuracy_2 = accuracy_score(bag_labels_microbial, bag_pred_microbial)
        accuracy_all = (accuracy * n_cancer_sample + accuracy_2 * n_microbial_sample) / (n_cancer_sample + n_microbial_sample)
        accuracys = [accuracy, accuracy_2, accuracy_all]
    print('Recalls: ' + str(recalls))
    print('roc', 'acc', 'recall', 'prec', 'fs')
    print(roc_auc, accuracys, recalls, precisions, fscores)
    return roc_auc, accuracys, recalls, precisions, fscores, cancer_matrix, microbial_matrix
    # return roc_auc_macro, accuracy, recall, precision, fscore
    
    
def confusion_matrix(bag_labels, bag_pred, class_labels):
    """
    混淆矩阵生成：
    参数：
        bag_labels (ndarray): [N] 真实标签
        bag_pred (ndarray): [N] 预测标签
        class_labels (list): n_class 标签名称
    """
    if len(class_labels) == 2:
        y_true, y_pred = [1 if i != 0 else 0 for i in bag_labels], [1 if i != 0 else 0 for i in bag_pred]
        # if isinstance(bag_logits[0], np.ndarray):
        #     y_true, y_pred = bag_labels, np.argmax(np.array(bag_logits), axis=-1)
        # else:
        #     y_true, y_pred = bag_labels, np.array([1 if x > 0.5 else 0 for x in bag_logits])
            
    y_true, y_pred = bag_labels, bag_pred
    num_classes = len(class_labels)
    print(max(y_true), max(y_pred), num_classes)

    # 初始化混淆矩阵
    cm_manual = np.zeros((num_classes, num_classes), dtype=int)

    # 遍历数据，填充混淆矩阵
    for true, pred in zip(y_true, y_pred):
        cm_manual[true][pred] += 1

    row_totals = [sum(row) for row in cm_manual]
    col_totals = [sum(col) for col in zip(*cm_manual)]
    total = sum(row_totals)

    # 重新格式化混淆矩阵，确保第一行包含类别名称
    print(f"Confusion Matrix for {len(bag_labels)} data")
    table = PrettyTable()
    table.field_names = ["实际\预测"] + class_labels + ["总计"]
    for i, label in enumerate(class_labels):
        table.add_row([label] + list(map(str, cm_manual[i])) + [row_totals[i]])
    table.add_row(["总计"] + list(map(str, col_totals)) + [total])
    print(table)
    return table

def prettytable_to_dataframe(pt):
    """
    将 PrettyTable 转换为 pandas DataFrame。
    
    参数：
        pt (PrettyTable): PrettyTable 对象。
    返回：
        pd.DataFrame: 转换后的 DataFrame。
    """
    # 获取表头和行数据
    headers = pt.field_names
    rows = pt._rows

    # 转换为 DataFrame
    df = pd.DataFrame(rows, columns=headers)
    return df


import pandas as pd
import numpy as np

def save_metrics_to_excel(roc_auc, accuracies, recalls, precisions, fscores, confusion_matrix_cancer_pt, confusion_matrix_microbial_pt, class_labels, output_excel_path):
    """
    将每个类别的AUC、召回率、精确率、F1分数，以及宫颈癌类别和微生物感染类别的平均指标存储到Excel表格中。
    所有指标以百分比形式显示，并保留两位小数。同时保存混淆矩阵（PrettyTable 格式）。
    
    参数：
        roc_auc (list): 每个类别的AUC值。
        accuracies (list): 三个准确率值，分别是宫颈癌类别、微生物感染类别和所有类别的准确率。
        recalls (list): 每个类别的召回率。
        precisions (list): 每个类别的精确率。
        fscores (list): 每个类别的F1分数。
        confusion_matrix_cancer_pt (PrettyTable): 宫颈癌类别的混淆矩阵（PrettyTable 格式）。
        confusion_matrix_microbial_pt (PrettyTable): 微生物感染类别的混淆矩阵（PrettyTable 格式）。
        class_labels (list): 类别标签。
        output_excel_path (str): 输出Excel文件的路径。
    """
    # 将指标转换为百分比形式，并保留两位小数
    roc_auc = [round(auc * 100, 2) for auc in roc_auc]
    recalls = [round(recall * 100, 2) for recall in recalls]
    precisions = [round(precision * 100, 2) for precision in precisions]
    fscores = [round(fscore * 100, 2) for fscore in fscores]
    accuracies = [round(acc * 100, 2) for acc in accuracies]

    # 创建一个DataFrame存储每个类别的指标
    results = {
        "Class": class_labels[1:],
        "AUC (%)": roc_auc,
        "Recall (%)": recalls,
        "Precision (%)": precisions,
        "F1 Score (%)": fscores
    }
    df = pd.DataFrame(results)

    # 计算宫颈癌类别和微生物感染类别的平均指标
    cancer_avg_auc = round(np.mean(roc_auc[:5]), 2)  # 前五个类别为宫颈癌
    cancer_avg_recall = round(np.mean(recalls[:5]), 2)
    cancer_avg_precision = round(np.mean(precisions[:5]), 2)
    cancer_avg_fscore = round(np.mean(fscores[:5]), 2)
    cancer_accuracy = accuracies[0]  # 宫颈癌类别的准确率

    microbial_avg_auc = round(np.mean(roc_auc[5:]), 2)  # 后面几个类别为微生物感染
    microbial_avg_recall = round(np.mean(recalls[5:]), 2)
    microbial_avg_precision = round(np.mean(precisions[5:]), 2)
    microbial_avg_fscore = round(np.mean(fscores[5:]), 2)
    microbial_accuracy = accuracies[1]  # 微生物感染类别的准确率

    # 计算所有类别的平均指标
    all_avg_auc = round(np.mean(roc_auc), 2)
    all_avg_recall = round(np.mean(recalls), 2)
    all_avg_precision = round(np.mean(precisions), 2)
    all_avg_fscore = round(np.mean(fscores), 2)
    all_accuracy = accuracies[2]  # 所有类别的准确率

    # 将平均指标添加到DataFrame中
    df_avg = pd.DataFrame({
        "Class": ["Cervical Cancer Average", "Microbial Infection Average", "All Classes Average"],
        "AUC (%)": [cancer_avg_auc, microbial_avg_auc, all_avg_auc],
        "Recall (%)": [cancer_avg_recall, microbial_avg_recall, all_avg_recall],
        "Precision (%)": [cancer_avg_precision, microbial_avg_precision, all_avg_precision],
        "F1 Score (%)": [cancer_avg_fscore, microbial_avg_fscore, all_avg_fscore],
        "Accuracy (%)": [cancer_accuracy, microbial_accuracy, all_accuracy]
    })

    # 合并结果
    df_final = pd.concat([df, df_avg], ignore_index=True)

    # 将 PrettyTable 转换为 DataFrame
    confusion_matrix_cancer_df = prettytable_to_dataframe(confusion_matrix_cancer_pt)
    confusion_matrix_microbial_df = prettytable_to_dataframe(confusion_matrix_microbial_pt)

    # 将混淆矩阵保存到Excel的不同Sheet中
    with pd.ExcelWriter(output_excel_path) as writer:
        df_final.to_excel(writer, sheet_name="Metrics", index=False)
        confusion_matrix_cancer_df.to_excel(writer, sheet_name="Confusion Matrix (Cancer)", index=False)
        confusion_matrix_microbial_df.to_excel(writer, sheet_name="Confusion Matrix (Microbial)", index=False)

    print(f"Metrics and confusion matrices saved to {output_excel_path}")
