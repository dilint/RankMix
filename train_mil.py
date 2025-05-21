import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.transforms.functional as VF
from torchvision import transforms

import sys, argparse, os, copy, itertools, glob, datetime
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_fscore_support
from sklearn.metrics import auc, precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_svmlight_file
from collections import OrderedDict
import time

from dataset import ClassDataset, C16Dataset
from dataset_frmil import FrmilClassDataset
from samplers import CategoriesSampler
from utils import multi_class_scores_mtl, save_metrics_to_excel

class FeatMag(nn.Module):
    
    def __init__(self, margin):
        super().__init__()
        self.margin = margin
        
    def forward(self, feat_pos, feat_neg, w_scale=1.0, zero_idx=None):        
        if zero_idx is not None:
            loss_act = self.margin - torch.norm(torch.mean(feat_pos[:,:zero_idx[0],:], dim=1), p=2, dim=1)
            loss_act[loss_act < 0] = 0
            loss_bkg = torch.norm(torch.mean(feat_neg[:,:zero_idx[1],:], dim=1), p=2, dim=1)
            
            loss_um = torch.mean((loss_act + loss_bkg) ** 2)        
            return loss_um/w_scale
        else:
            loss_act = self.margin - torch.norm(torch.mean(feat_pos, dim=1), p=2, dim=1)
            loss_act[loss_act < 0] = 0
            loss_bkg = torch.norm(torch.mean(feat_neg, dim=1), p=2, dim=1)
            
            loss_um = torch.mean((loss_act + loss_bkg) ** 2)        
            return loss_um/w_scale

def get_bag_feats(csv_file_df, args):    
    feats_csv_path = csv_file_df.iloc[0]
    df = pd.read_csv(feats_csv_path)
    feats = shuffle(df).reset_index(drop=True)
    feats = feats.to_numpy()
    label = np.zeros(args.num_classes)
    if args.num_classes==1:
        label[0] = csv_file_df.iloc[1]
    else:
        if int(csv_file_df.iloc[1])<=(len(label)-1):
            label[int(csv_file_df.iloc[1])] = 1
        
    return label, feats

def train(train_loader, milnet, criterion, optimizer, args):
    milnet.train()    
    total_loss = 0    
    Tensor = torch.cuda.FloatTensor
     
    for i, batch in enumerate(train_loader):
        optimizer.zero_grad()
        
        label, feats = batch        
        label, feats = label.cuda(), feats.view(-1, args.feats_size).cuda()
        
        feats = F.dropout(feats,p=args.dropout_input)
        ins_prediction, bag_prediction, _, _ = milnet(feats)        
        max_prediction, _ = torch.max(ins_prediction, 0)
        bag_loss = criterion(bag_prediction.view(1, -1), label.view(1, -1))
        max_loss = criterion(max_prediction.view(1, -1), label.view(1, -1))
        loss = 0.5*bag_loss + 0.5*max_loss
        loss.backward()
        optimizer.step()
        total_loss = total_loss + loss.item()        
        sys.stdout.write('\r Training bag [%d/%d] bag loss: %.4f' % (i, len(train_loader.dataset), loss.item()))
    
    return total_loss / len(train_loader.dataset)


def train_frmil(train_loader, milnet, criterion, optimizer, args):
    milnet.train()
    
    total_loss = 0
    if args.loss=='FrmilLoss':
        ce_weight  = [i for i in train_loader.dataset.count_dict.values()]
        ce_weight  = 1. / torch.tensor(ce_weight, dtype=torch.float)
        ce_weight  = ce_weight.cuda()
        bce_weight = train_loader.dataset.pos_weight.cuda()
    mag_loss = FeatMag(margin=args.mag).cuda()
    
    for i, (feats, label, _, zero_idx) in enumerate(train_loader):
        optimizer.zero_grad()        

        norm_idx = torch.where(label == 0)[0].numpy()[0]
        ano_idx  = 1 - norm_idx        
        label, feats = label.cuda().float(), feats.view(2, -1, args.feats_size).cuda()        
       
        feats = F.dropout(feats,p=args.dropout_input)
        bag_prediction, query, ins_prediction = milnet(feats)  # (2,C), (2,N,D), (2,N)
        
        max_prediction, _ = torch.max(ins_prediction, 1) # (2)

        if args.loss=='FrmilLoss':
            if args.num_classes==1:
                max_loss = F.binary_cross_entropy(max_prediction, label.squeeze()) # (2), (2)                
                bag_loss = criterion(bag_prediction, label.view(2, -1)) 
            elif args.num_classes==2:                
                max_loss = F.binary_cross_entropy(max_prediction, label[:, 1], weight=bce_weight)
                bag_loss = F.cross_entropy(bag_prediction, label[:, 1].long(), weight=ce_weight)            
            if args.dataset.startswith("Lung"):
                ano_idx, norm_idx = norm_idx, ano_idx                
            loss_ft  = mag_loss(query[ano_idx,:,:].unsqueeze(0),query[norm_idx,:,:].unsqueeze(0), w_scale=query.shape[1])
            loss = (bag_loss + max_loss + loss_ft) * (1./3)
        elif args.loss=='DsmilLoss':     
            # ablation study of loss_ft       
            bag_loss = criterion(bag_prediction.view(2, -1), label.view(2, -1))
            if args.num_classes==1:
                max_loss = criterion(max_prediction.view(2, -1), label.view(2, -1)) # (2,1), (2,1)      
            elif args.num_classes==2:   
                max_loss = criterion(max_prediction.view(2, -1), label[:, 1].view(2, -1)) # (2,1), (2,2)
            loss = 0.5*bag_loss + 0.5*max_loss
        loss.backward()
        optimizer.step()
        total_loss = total_loss + loss.item()        
        sys.stdout.write('\r Training bag [%d/%d] bag loss: %.4f' % (i, len(train_loader.dataset), loss.item()))
    
    return total_loss / len(train_loader.dataset)

def dropout_patches(feats, p):
    idx = np.random.choice(np.arange(feats.shape[0]), int(feats.shape[0]*(1-p)), replace=False)
    sampled_feats = np.take(feats, idx, axis=0)
    pad_idx = np.random.choice(np.arange(sampled_feats.shape[0]), int(feats.shape[0]*p), replace=False)
    pad_feats = np.take(sampled_feats, pad_idx, axis=0)
    sampled_feats = np.concatenate((sampled_feats, pad_feats), axis=0)
    return sampled_feats


def test(test_loader, milnet, criterion, optimizer, args):
    milnet.eval()    
    total_loss = 0
    test_labels = []
    test_predictions = []
    Tensor = torch.cuda.FloatTensor
    with torch.no_grad():        
        for i, batch in enumerate(test_loader):
            if args.model=='frmil':
                feats, label, _  = batch
                feats, label = feats.cuda(), label.cuda().float()           
                bag_prediction, ins_prediction  = milnet(feats)
                ins_prediction = ins_prediction.squeeze(0)  
            else:
                label, feats = batch
                label, feats = label.cuda().float(), feats.view(-1, args.feats_size).cuda()            
                ins_prediction, bag_prediction, _, _ = milnet(feats)
            
            bag_loss = criterion(bag_prediction.view(1, -1), label.view(1, -1))
            max_loss = torch.tensor(0)
            max_prediction, _ = torch.max(ins_prediction, 0)
            if args.num_classes!=1 and args.model=='frmil':   
                max_loss = criterion(max_prediction.view(1, -1), label[:, 1].view(1, -1))     
            else:
                max_loss = criterion(max_prediction.view(1, -1), label.view(1, -1))  
            loss = 0.5*bag_loss + 0.5*max_loss
            total_loss = total_loss + loss.item()            
            sys.stdout.write('\r Testing bag [%d/%d] bag loss: %.4f' % (i, len(test_loader.dataset), loss.item()))            
            test_labels.append([label.squeeze().cpu().numpy()])
            test_predictions.extend([1.0*torch.sigmoid(bag_prediction).squeeze().cpu().numpy()])
            ori_test_predictions = test_predictions.copy()
    test_labels = np.array(test_labels)
    test_predictions = np.array(test_predictions)
    if len(test_labels.shape)==3:
        test_labels = test_labels.squeeze(1)
    auc_value, _, thresholds_optimal = multi_label_roc(test_labels, test_predictions, args.num_classes, pos_label=1)
    prauc_value = multi_label_prauc(test_labels, test_predictions, args.num_classes)
    if args.num_classes==1:
        class_prediction_bag = copy.deepcopy(test_predictions)
        class_prediction_bag[test_predictions>=thresholds_optimal[0]] = 1
        class_prediction_bag[test_predictions<thresholds_optimal[0]] = 0
        test_predictions = class_prediction_bag
        test_labels = np.squeeze(test_labels)
    else:        
        for i in range(args.num_classes):
            class_prediction_bag = copy.deepcopy(test_predictions[:, i])
            class_prediction_bag[test_predictions[:, i]>=thresholds_optimal[i]] = 1
            class_prediction_bag[test_predictions[:, i]<thresholds_optimal[i]] = 0
            test_predictions[:, i] = class_prediction_bag
    bag_score = 0    
    for i in range(0, len(test_loader.dataset)):
        bag_score = np.array_equal(test_labels[i], test_predictions[i]) + bag_score         
    avg_score = bag_score / len(test_loader.dataset)    
    
    ## Cal avg score with threshold=0.5
    """
    optimal threshold is used in DSMIL https://github.com/binli123/dsmil-wsi
    However, we only use threshold=0.5 instead of overfitting to different datasets.
    """
    if args.num_classes==1:
        class_prediction_bag = copy.deepcopy(test_predictions)
        class_prediction_bag[test_predictions>=0.5] = 1
        class_prediction_bag[test_predictions<0.5] = 0
        test_predictions = class_prediction_bag
        test_labels = np.squeeze(test_labels)
    else:        
        for i in range(args.num_classes):
            class_prediction_bag = copy.deepcopy(test_predictions[:, i])
            class_prediction_bag[test_predictions[:, i]>=0.5] = 1
            class_prediction_bag[test_predictions[:, i]<0.5] = 0
            test_predictions[:, i] = class_prediction_bag
    bag_score = 0
    for i in range(0, len(test_loader.dataset)):
        bag_score = np.array_equal(test_labels[i], test_predictions[i]) + bag_score         
    avg_05_score = bag_score / len(test_loader.dataset)

    ######################### custome metrics #########################
    # ori_test_predictions = torch.sigmoid(torch.tensor(ori_test_predictions)).cpu().numpy()
    roc_auc, accuracies, recalls, precisions, fscores, cancer_matrix, microbial_matrix = multi_class_scores_mtl(test_labels, ori_test_predictions, args.class_labels, threshold=0.5)
    output_excel_path = os.path.join(args.save_path, 'metrics.xlsx')
    save_metrics_to_excel(roc_auc, accuracies, recalls, precisions, fscores, cancer_matrix, microbial_matrix, args.class_labels, output_excel_path)
    ######################### end #########################

    return total_loss / len(test_loader.dataset), avg_score, auc_value, thresholds_optimal, avg_05_score, prauc_value

def multi_label_roc(labels, predictions, num_classes, pos_label=1):
    fprs = []
    tprs = []
    thresholds = []
    thresholds_optimal = []
    aucs = []
    
    if len(predictions.shape)==1:
        predictions = predictions[:, None]
    
    for c in range(0, num_classes):
        label = labels[:, c]
        prediction = predictions[:, c]        
        fpr, tpr, threshold = roc_curve(label, prediction, pos_label=1)
        fpr_optimal, tpr_optimal, threshold_optimal = optimal_thresh(fpr, tpr, threshold)
        c_auc = roc_auc_score(label, prediction)
        aucs.append(c_auc)
        thresholds.append(threshold)
        thresholds_optimal.append(threshold_optimal)
    return aucs, thresholds, thresholds_optimal

def multi_label_prauc(labels, predictions, num_classes):
    praucs = []
    if len(predictions.shape)==1:
        predictions = predictions[:, None]
    for c in range(0, num_classes):
        label = labels[:, c]
        prediction = predictions[:, c]
        precision, recall, thresholds = precision_recall_curve(label, prediction)        
        auc_precision_recall = auc(recall, precision)   
        praucs.append(auc_precision_recall)        
    return praucs

def optimal_thresh(fpr, tpr, thresholds, p=0):
    loss = (fpr - tpr) - p * tpr / (fpr + tpr + 1)
    idx = np.argmin(loss, axis=0)
    return fpr[idx], tpr[idx], thresholds[idx]

def main():
    parser = argparse.ArgumentParser(description='Train DSMIL on 20x patch features learned by SimCLR')
    parser.add_argument('--num_classes', default=9, type=int, help='Number of output classes [2]')
    parser.add_argument('--feats_size', default=1536, type=int, help='Dimension of the feature size [512]')
    parser.add_argument('--n_heads', default=8, type=int, help='the number of heads for self-attention')
    parser.add_argument('--lr', default=0.0002, type=float, help='Initial learning rate [0.0002]')
    parser.add_argument('--num_epochs', default=40, type=int, help='Number of total training epochs [40|200]')
    parser.add_argument('--gpu_index', type=int, nargs='+', default=(0,), help='GPU ID(s) [0]')
    parser.add_argument('--weight_decay', default=1e-5, type=float, help='Weight decay [5e-3]')
    parser.add_argument('--dataset', default='gc_10k', type=str, help='Dataset folder name')
    parser.add_argument('--dataset_dir', type=str, default='dataset', help='Dataset directory')
    parser.add_argument('--split', default=0.2, type=float, help='Training/Validation split [0.2]')
    parser.add_argument('--model', default='abmil', type=str, help='MIL model [dsmil, frmil]')
    parser.add_argument('--loss', default='FrmilLoss', type=str, help='loss for FRMIL')
    parser.add_argument("--mag", type=float, default=8.48, help='margin used in the feature loss (cm16)')
    parser.add_argument('--dropout_patch', default=0, type=float, help='Patch dropout rate [0]')
    parser.add_argument('--dropout_node', default=0, type=float, help='Bag classifier dropout rate [0]')
    parser.add_argument('--dropout_input', default=0.2, type=float, help='The ratio of dropout')
    parser.add_argument('--non_linearity', default=1, type=float, help='Additional nonlinear operation [0]')
    parser.add_argument('--num_workers', default=4, type=int, help='num_workers for dataloader')  
    parser.add_argument('--save_all', action='store_true', help='save all models for every epoch')        
    args = parser.parse_args()
    gpu_ids = tuple(args.gpu_index)
    os.environ['CUDA_VISIBLE_DEVICES']=','.join(str(x) for x in gpu_ids)
    
    if args.model == 'frmil':
        from frmil import FRMIL
        assert 'Not support frmil'
        milnet = FRMIL(args).cuda()
        print("Using FRMIL model!!")
    elif args.model == 'dsmil':       
        import dsmil as mil
        print("Using DSMIL model!!")

        i_classifier = mil.FCLayer(in_size=args.feats_size, out_size=args.num_classes).cuda()
        b_classifier = mil.BClassifier(input_size=args.feats_size, output_class=args.num_classes, dropout_v=args.dropout_node, nonlinear=args.non_linearity).cuda()
        milnet = mil.MILNet(i_classifier, b_classifier).cuda()
    else:
        from model import MIL
        import dsmil as dsmil
        i_classifier = dsmil.FCLayer(in_size=args.feats_size, out_size=args.num_classes).cuda()
        b_classifier = MIL(input_dim=args.feats_size,
            mlp_dim=512,
            n_classes=args.num_classes,
            mil=args.model,
            dropout=0.25).cuda()
        milnet = dsmil.MILNetv2(i_classifier, b_classifier).cuda()
        
    """
    This model weight('init.pth') is downloaded from DSMIL https://github.com/binli123/dsmil-wsi
    However, we don't use this weight in the paper for the fairness because we don't know the source of this weight.
    """
    # if args.model == 'dsmil':
    #     state_dict_weights = torch.load('init.pth')
    #     milnet.load_state_dict(state_dict_weights, strict=False)
    #     print("Loading checkpoint: init.pth")
    
    criterion = nn.BCEWithLogitsLoss()    
    # optimizer = torch.optim.Adam(milnet.parameters(), lr=args.lr, betas=(0.5, 0.9), weight_decay=args.weight_decay)
    optimizer = torch.optim.Adam(milnet.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.num_epochs, 0.000005)
    
    if args.model=='dsmil':
        save_path = os.path.join('weights', f"{args.dataset}_{args.model}_{args.num_classes}class_dropout_{args.dropout_input}", datetime.datetime.today().strftime("%Y%m%d_%H%M%S"))
    elif args.model=='frmil':
        save_path = os.path.join('weights', f"{args.dataset}_{args.model}_{args.loss}_{args.num_classes}class_thres{args.mag}_dropout_{args.dropout_input}", datetime.datetime.today().strftime("%Y%m%d_%H%M%S"))
    else:
        save_path = os.path.join('weights', f"{args.dataset}_{args.model}_{args.num_classes}class_dropout_{args.dropout_input}", datetime.datetime.today().strftime("%Y%m%d_%H%M%S"))
    args.save_path = save_path
    os.makedirs(save_path, exist_ok=True)
    log_filename = os.path.join(save_path, 'log.txt')    
    print(f"log_filename: {log_filename}")
    
    ################# custome dataset #################
    id2label = {
    0: 'nilm',
    1: 'ascus',
    2: 'asch',
    3: 'lsil',
    4: 'hsil',
    5: 'agc',
    6: 't',
    7: 'm',
    8: 'bv',}
    label2id = {v: k for k, v in id2label.items()}
    if args.dataset == 'gc_10k':
        args.train_label_path = '/data/wsi/TCTGC10k-labels/9_labels/TCTGC10k-v15-train.csv'
        args.test_label_path = '/data/wsi/TCTGC10k-labels/9_labels/TCTGC10k-v15-test.csv'
        # args.dataset_root = '/data/wsi/TCTGC10k-features/gigapath-coarse'
        args.dataset_root = '/data/wsi/TCTGC10k-features/gigapath-1000'
        args.num_classes = 9
        args.class_labels = ['nilm', 'ascus', 'asch', 'lsil', 'hsil', 'agc', 't', 'm', 'bv']
        df_train = pd.read_csv(args.train_label_path)
        train_wsi_names = df_train['wsi_name'].values
        train_wsi_labels = df_train['wsi_label'].map(label2id).values
        df_test = pd.read_csv(args.test_label_path)
        test_wsi_names = df_test['wsi_name'].values
        test_wsi_labels = df_test['wsi_label'].map(label2id).values
        train_dataset = C16Dataset(train_wsi_names, train_wsi_labels,root=args.dataset_root)
        val_dataset = C16Dataset(test_wsi_names,test_wsi_labels,root=args.dataset_root)
        test_dataset = C16Dataset(test_wsi_names,test_wsi_labels,root=args.dataset_root)
    ################# end #################
    
    # if args.model=="dsmil":
    #     train_dataset = ClassDataset(train_path, args)
    #     val_dataset = train_dataset
    #     test_dataset = ClassDataset(test_path, args)

    # elif args.model=="frmil":
    #     train_dataset = FrmilClassDataset(df=train_path, mode='train', num_classes=args.num_classes, batch=True)    
    #     val_dataset   = FrmilClassDataset(df=train_path, mode='train', num_classes=args.num_classes)  
    #     test_dataset  = FrmilClassDataset(df=test_path, mode='test', num_classes=args.num_classes) 
    #     ## for Debug
    #     # train_dataset = FrmilClassDataset(df=train_path[:3], mode='train', num_classes=args.num_classes, batch=True)
    #     # test_dataset  = FrmilClassDataset(df=test_path[70:80], mode='test', num_classes=args.num_classes)            
    #     # val_dataset   = FrmilClassDataset(df=train_path[:3], mode='train', num_classes=args.num_classes) 
        
    #     train_sampler = CategoriesSampler(train_dataset.labels, n_batch=len(train_dataset), n_cls=2, n_per=1)  

    train_loader = DataLoader(dataset=train_dataset, batch_size=1, num_workers=args.num_workers, pin_memory=torch.cuda.is_available(), shuffle=True, drop_last=False)
    # elif args.model=="frmil":
    #     train_loader = DataLoader(dataset=train_dataset, batch_sampler=train_sampler, num_workers=args.num_workers, pin_memory=torch.cuda.is_available(), drop_last=False)
    val_loader = DataLoader(dataset=val_dataset, batch_size=1, num_workers=args.num_workers, pin_memory=torch.cuda.is_available(), drop_last=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, num_workers=args.num_workers, pin_memory=torch.cuda.is_available(), drop_last=False)
    
    best_val_score = 0      
    for epoch in range(args.num_epochs):
        start = time.time()
        if args.model == 'frmil':
            train_loss_bag = train_frmil(train_loader, milnet, criterion, optimizer, args) # iterate all bags            
        else:
            train_loss_bag = train(train_loader, milnet, criterion, optimizer, args) # iterate all bags
            
        # val_loss_bag, val_avg_score, val_aucs, val_thresholds_optimal, val_avg_05_score, val_praucs = test(val_loader, milnet, criterion, optimizer, args)
        test_loss_bag, avg_score, aucs, thresholds_optimal, avg_05_score, praucs = test(test_loader, milnet, criterion, optimizer, args)
        
        if args.dataset=='Lung':
            output_string = (
                '\r Epoch [%d/%d] time: %.4f train loss: %.4f , test loss: %.4f, test acc: %.4f, test 0.5 acc: %.4f, auc_LUAD: %.4f, auc_LUSC: %.4f, prauc_LUAD: %.4f, prauc_LUSC: %.4f' % 
                (epoch, args.num_epochs, time.time()-start, train_loss_bag, test_loss_bag, avg_score, avg_05_score, aucs[0], aucs[1], praucs[0], praucs[1])
            )
        else:
            output_string = (
                '\r Epoch [%d/%d] time: %.4f train loss: %.4f , test loss: %.4f, test acc: %.4f, test 0.5 acc: %.4f' % 
                (epoch, args.num_epochs, time.time()-start, train_loss_bag, test_loss_bag, avg_score, avg_05_score) 
                + ', AUC: '
                + '|'.join('class-{}>>{:.4f}'.format(*k) for k in enumerate(aucs))
                + ', PRAUC: '
                + '|'.join('class-{}>>{:.4f}'.format(*k) for k in enumerate(praucs))
            )

        print(output_string)
        with open(log_filename, "a") as log_file:
            log_file.write(output_string + '\n')

        scheduler.step()        
        current_score = (sum(aucs) + avg_score + sum(praucs))/3         
        
        if args.save_all or epoch==args.num_epochs-1:
            torch.save(milnet.state_dict(), os.path.join(save_path, f"checkpoint_{epoch}.pth"))    
        torch.save(milnet.state_dict(), os.path.join(save_path, f"temp.pth"))
            
        save_name = os.path.join(save_path, 'best.pth')
        torch.save(milnet.state_dict(), save_name)
            
        if args.dataset=='Lung':
            print('Best model saved at: ' + save_name + ' Best thresholds: LUAD %.4f, LUSC %.4f' % (thresholds_optimal[0], thresholds_optimal[1]))
            with open(log_filename, "a") as log_file:
                log_file.write('Best model saved at: ' + save_name + ' Best thresholds: LUAD %.4f, LUSC %.4f\n' % (thresholds_optimal[0], thresholds_optimal[1]))
        else:
            print('Best model saved at: ' + save_name)
            print('Best thresholds ===>>> '+ '|'.join('class-{}>>{}'.format(*k) for k in enumerate(thresholds_optimal)))
            with open(log_filename, "a") as log_file:
                log_file.write('Best model saved at: ' + save_name)
                log_file.write('Best thresholds ===>>> '+ '|'.join('class-{}>>{}'.format(*k) for k in enumerate(thresholds_optimal)) + '\n')
    
    test_loss_bag, avg_score, aucs, thresholds_optimal, avg_05_score, praucs = test(test_loader, milnet, criterion, optimizer, args)
    # if args.dataset=='Lung':         
    #     result_string2 = f"The Last Model: ACC {avg_score:.4f} | ACC05 {avg_05_score:.4f} |auc_LUAD: {aucs[0]:.4f}/{praucs[0]:.4f}, auc_LUSC: {aucs[1]:.4f}/{praucs[1]:.4f}"
    # else:         
    #     result_string2 = f"The Last Model: ACC {avg_score:.4f} | ACC05 {avg_05_score:.4f} | AUC {aucs[0]:.4f}/{praucs[0]:.4f}"
    # print(result_string2)    
    # with open(log_filename, "a") as log_file:
    #     log_file.write(result_string2 + '\n')        

    print()    

if __name__ == '__main__':
    main()