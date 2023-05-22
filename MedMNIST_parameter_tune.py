import torch
import torchvision
import torchvision.transforms as transforms
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import glob
import argparse
import copy
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score
import optuna
from optuna.visualization import *

def parse_args(script):
    parser = argparse.ArgumentParser(description= 'few-shot script %s' %(script))
    parser.add_argument('--seed', default=53, help='Seed for Numpy and pyTorch. Default: 0 (None)', type=int)
    # parser.add_argument('--dataset', default='MedMNIST', help='MedMNIST/MNIST')
    parser.add_argument('--method', default='MTMK L1', help='MTMK mix/MTMK L1/MTMK L2/LR/SVM/STMK/STSK/MTSK')
    parser.add_argument('--imagefile', default='./organcmnist.npz', help='file link of selected dataset')
    parser.add_argument('--textfile', default='./phecode_final.tsv', help='file link of selected dataset')
    parser.add_argument('--savedir', default='./models', help='dir for saved models')
    parser.add_argument('--resdir', default='./result', help='dir for results')
    parser.add_argument('--savemodel', default=True, help='save model or not', type=bool)
    parser.add_argument('--folds', default=5, help='folds of cross validation', type=int)
    parser.add_argument('--samples', default=100, help='postive sample num of images', type=int)
    parser.add_argument('--texts', default=10, help='postive sample num of texts', type=int)
    parser.add_argument('--lmd2', default=0.05, help='L1 penalization term', type=float)
    parser.add_argument('--lmd3', default=0., help='L2 penalization term', type=float)
    parser.add_argument('--device', default='cuda', help='cpu or gpu')
    parser.add_argument('--textonly', default=False, help='only train on text', type=bool)
    parser.add_argument('--imageonly', default=False, help='only train on images', type=bool)
    parser.add_argument('--onefold', default=False, help='only run fold 0', type=bool)
    parser.add_argument('--start',default=0.05, help='start of lmd', type=float)
    parser.add_argument('--end',default=0.5, help='end of lmd', type=float)
    parser.add_argument('--step',default=0.05, help='step of lmd', type=float)
    return parser.parse_args()

def _set_seed(seed, verbose=True):
    if(seed!=0):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        if(verbose): print("[INFO] Setting SEED: " + str(seed))
    else:
        if(verbose): print("[INFO] Setting SEED: None")

if __name__ == '__main__':
    params = parse_args('MedMNIST')
    data = np.load(params.imagefile)
    _set_seed(params.seed)
    
    # split train_images
    TASK_NUM = 11
    task_data = []
    for i in range(TASK_NUM):
        task_data.append([])
    for i in range(data['train_images'].shape[0]):
        task_data[data['train_labels'][i].item()].append(i)
        
    # sample
    SAMPLES = params.samples
    for i in range(TASK_NUM):
        task_data[i] = random.sample(task_data[i], 2 * SAMPLES)
        
    # get image data and labels
    MT_training_Data = []
    MT_training_Label = []
    # skip 1, 4, 7, 9, 10
    CLASS_LIST = [0, 2, 3, 5, 6, 8]
    TASK_NUM = len(CLASS_LIST)
    for i in CLASS_LIST:
        # positive images
        train_t = random.sample(task_data[i], SAMPLES)
        # negative images
        # draw SAMPLES(default 100) negative samples
        # if SAMPLES % (TASK_NUM-1) !=0, we draw more/less samples from the last class
        SUBSAMPLES = int(SAMPLES/(TASK_NUM-1))
        idx = 1
        for j in CLASS_LIST:
            if j != i:
                if not idx == TASK_NUM-1:
                    train_t += random.sample(task_data[j], SUBSAMPLES)
                    idx += 1
                else:
                    train_t += random.sample(task_data[j], SAMPLES - (idx - 1) * SUBSAMPLES)

        label = np.array([1] * SAMPLES + [-1] * SAMPLES, dtype=int)
        # shuffle label and train
        # index = [i for i in range(2*SAMPLES)]
        # random.shuffle(index)
        # train_t = np.array(train_t)[index]
        # label = label[index]
        train_t = np.array(train_t)
    
        # preprocess by dividing 255.
        MT_training_Data.append((data['train_images'][train_t] / 255.0).reshape((2 * SAMPLES, 28 * 28)))
        MT_training_Label.append(label)
        
    # get text data and labels
    text_embedding = []
    text_label = []
    phecode_text = []
    
    phecode = pd.read_csv(params.textfile, sep='\t')
    TEXT_NUM = params.texts
    # CLASS_LIST and CATEGORY_LIST is one to one
    CLASS_LIST = [0, 2, 3, 5, 6, 8]
    CATEOGRY_LIST = ['bladder', 'femur', 'heart', 'kidney', 'liver', 'lung']
    TASK_NUM = len(CLASS_LIST)
    
    # We select TEXT_NUM postive samples from each category
    # We will first use MIKGI's gold similarity, and then use cosine similarity score
    for category in CATEOGRY_LIST:
        phecode_cat = phecode[phecode['category'] == category].sort_values(by=['MIKGI', 'cossim'], ascending=[False, False])
        embeddings = [eval(emb) for emb in phecode_cat.head(TEXT_NUM)['Embedding'].to_list()]
        text_embedding.append(embeddings)
        phecode_text.append(phecode_cat.head(TEXT_NUM)['CUI'].to_list())
    
    # Now we sample negative texts in the same way as images
    text_data = copy.deepcopy(text_embedding)
    phecodes = copy.deepcopy(phecode_text)
    SUBSAMPLES = int(TEXT_NUM/(TASK_NUM-1))
    for i in range(TASK_NUM):
        idx = 1
        for j in range(TASK_NUM):
            if j != i:
                if not idx == TASK_NUM-1:
                    samples = random.sample(range(len(text_embedding[j])), SUBSAMPLES)
                    for sample_idx in samples:
                        text_data[i].append(text_embedding[j][sample_idx])
                        phecodes[i].append(phecode_text[j][sample_idx])
                    idx += 1
                else:
                    samples = random.sample(range(len(text_embedding[j])), TEXT_NUM - (idx - 1) * SUBSAMPLES)
                    for sample_idx in samples:
                        text_data[i].append(text_embedding[j][sample_idx])
                        phecodes[i].append(phecode_text[j][sample_idx])
    
    text_label = [np.array([1]*TEXT_NUM + [-1]*TEXT_NUM, dtype=int) for i in range(TASK_NUM)]    
    
    # add padding for texts
    for i in range(len(text_data)):
        text_data[i] = torch.tensor(text_data[i], dtype=float)
        text_data[i] = torch.nn.functional.pad(text_data[i], (0, 784 - 200), 'constant', 0)
        
    # stratifiedKFold
    data = MT_training_Data
    y = MT_training_Label
    text_x = text_data
    text_y = text_label
    skf = StratifiedKFold(n_splits=params.folds, shuffle=True, random_state=params.seed)
    train_set = []
    test_set = []
    train_label = []
    test_label = []
    train_phecode = []
    test_phecode = []

    for i in range(params.folds):
        train_set.append([])
        test_set.append([])
        train_label.append([])
        test_label.append([])
        train_phecode.append([])
        test_phecode.append([])
    
    for i in range(TASK_NUM):
        fold= 0
        for train_index, test_index in skf.split(text_x[i], text_y[i]):           
            train_set[fold].append(text_x[i][train_index].to(params.device))
            train_label[fold].append(torch.tensor(text_y[i],dtype=int)[train_index].to(params.device))
            test_set[fold].append(text_x[i][test_index].to(params.device))
            test_label[fold].append(torch.tensor(text_y[i],dtype=int)[test_index].to(params.device))
            train_phecode[fold].append(np.array(phecodes[i])[train_index])
            test_phecode[fold].append(np.array(phecodes[i])[test_index])
            fold += 1
            
    for i in range(TASK_NUM):
        fold= 0
        for train_index, test_index in skf.split(data[i], y[i]):
            train_set[fold][i] = torch.cat((train_set[fold][i], torch.tensor(data[i],dtype=float)[train_index].to(params.device)), dim=0)
            train_label[fold][i] = torch.cat((train_label[fold][i], torch.tensor(y[i],dtype=int)[train_index].to(params.device)), dim=0)
            test_set[fold][i] = torch.cat((test_set[fold][i], torch.tensor(data[i],dtype=float)[test_index].to(params.device)), dim=0)
            test_label[fold][i] = torch.cat((test_label[fold][i], torch.tensor(y[i],dtype=int)[test_index].to(params.device)), dim=0)
            fold += 1
    # now we get train_set, test_set, train_label, test_label
    
    # only MTMK is implemented now
    if params.method in ['MTMK mix', 'MTMK L1', 'MTMK L2','MTSK','STMK','STSK']:
        from mt_model_gpu_grad import MtModel, SingleModel
        from kernels_gpu_grad import *

        gamma_list = [10 ** i for i in range(-3, 3)]
        kernel_list = []
        kernel_list.extend([Gaussian_kernel(gamma) for gamma in gamma_list])
        kernel_list.extend([linear(), poly_kernel(2), poly_kernel(3), poly_kernel(4), poly_kernel(5)])
        kernel_list.extend([poly_kernel(i) for i in range(6, 11)])

    # ********************
    # 调参:
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    if params.method == 'MTMK L1':
        def objective(trial):
            lambda_2 = trial.suggest_float('lambda_2',params.start,params.end)

            ret_auc = np.zeros((TASK_NUM),dtype=float)
            # ret1_auc = np.zeros((TASK_NUM),dtype=float)
            train_x = train_set[0]
            train_y = train_label[0]
            test_x = test_set[0]
            test_y = test_label[0]
            task_num = len(train_x)

            model = MtModel(kernel_list, lambda_2=lambda_2, lambda_3=0., h=1e-6, epsilon=1e-3, k_0=1, sigma=0.1, delta=0.5, device=params.device)
            model.fit(train_x, train_y)
            for i in range(task_num):
                x_test_k = test_x[i]
                y_test_k = test_y[i]

                # x_train_k = train_x[i]
                y_train_k = train_y[i]

                y_pred_k = torch.zeros_like(y_test_k,dtype=float)

                for m in range(x_test_k.shape[0]):
                    y_pred_k[m] = model.high_dim_predict(x_test_k[m],i)
                

                y_pred_k = y_pred_k.to('cpu').detach().numpy()
                y_pred_k = 1 / (1 + np.exp(-y_pred_k))
                

                y_test_k = y_test_k.to('cpu').detach().numpy()
                y_train_k = y_train_k.to('cpu').detach().numpy()

                auc_ = roc_auc_score(y_test_k, y_pred_k)
                if auc_ < 0.5:
                    auc_ = 1 - auc_ 
                ret_auc[i] = ret_auc[i] + auc_
                # ret1_auc[i] = ret1_auc[i] + auc1_
            print(f"lambda2: {lambda_2}; value:{np.mean(ret_auc)}")
            return np.mean(ret_auc)
        grids = int((params.end-params.start)/params.step)
        search_space = {"lambda_2": [params.start + i*params.step for i in range(grids+1)]}
        study_MTMK_l1 = optuna.create_study(sampler=optuna.samplers.GridSampler(search_space), study_name='MTMK L1',direction='maximize')
        study_MTMK_l1.optimize(objective)
        print("MTMK_L1 optimal parameters:")
        print(study_MTMK_l1.best_params)
        print("MTMK_L1 optimal value:")
        print(study_MTMK_l1.best_value)
        lmd2 = study_MTMK_l1.best_params['lambda_2']
        lmd3 = 0.
        
    if params.method == 'MTMK mix':
        def objective(trial):
            lambda_2 = trial.suggest_float('lambda_2',params.start,params.end)
            lambda_3 = trial.suggest_float('lambda_3',params.start,params.end)
            ret_auc = np.zeros((TASK_NUM),dtype=float)

            train_x = train_set[0]
            train_y = train_label[0]
            test_x = test_set[0]
            test_y = test_label[0]
            task_num = len(train_x)

            model = MtModel(kernel_list, lambda_2=lambda_2, lambda_3=lambda_3, h=1e-6, epsilon=1e-3, k_0=1, sigma=0.1, delta=0.5, device=params.device)
            model.fit(train_x, train_y)

            for i in range(task_num):
                x_test_k = test_x[i]
                y_test_k = test_y[i]

                # x_train_k = train_x[i]
                y_train_k = train_y[i]

                y_pred_k = torch.zeros_like(y_test_k,dtype=float)

                for m in range(x_test_k.shape[0]):
                    y_pred_k[m] = model.high_dim_predict(x_test_k[m],i)
                

                y_pred_k = y_pred_k.to('cpu').detach().numpy()
                y_pred_k = 1 / (1 + np.exp(-y_pred_k))
                

                y_test_k = y_test_k.to('cpu').detach().numpy()
                y_train_k = y_train_k.to('cpu').detach().numpy()

                auc_ = roc_auc_score(y_test_k, y_pred_k)
                if auc_ < 0.5:
                    auc_ = 1 - auc_ 
                ret_auc[i] = ret_auc[i] + auc_

            print(f"lambda2: {lambda_2}; lambda3: {lambda_3} value:{np.mean(ret_auc)}")
            return np.mean(ret_auc)
        grids = int((params.end-params.start)/params.step)
        search_space = {"lambda_2": [params.start + i*params.step for i in range(grids+1)],\
                        "lambda_3": [params.start + i*params.step for i in range(grids+1)]}
        study_MTMK_l1 = optuna.create_study(sampler=optuna.samplers.GridSampler(search_space), study_name='MTMK Mix',direction='maximize')
        study_MTMK_l1.optimize(objective)
        print("MTMK_Mix optimal parameters:")
        print(study_MTMK_l1.best_params)
        print("MTMK_Mix optimal value:")
        print(study_MTMK_l1.best_value)
        lmd2 = study_MTMK_l1.best_params['lambda_2']
        lmd3 = study_MTMK_l1.best_params['lambda_3']

    if params.method == 'MTMK L2':
        def objective(trial):
            # lambda_2 = trial.suggest_float('lambda_2',params.start,params.end)
            lambda_3 = trial.suggest_float('lambda_3',params.start,params.end)
            ret_auc = np.zeros((TASK_NUM),dtype=float)

            train_x = train_set[0]
            train_y = train_label[0]
            test_x = test_set[0]
            test_y = test_label[0]
            task_num = len(train_x)

            model = MtModel(kernel_list, lambda_2=0., lambda_3=lambda_3, h=1e-6, epsilon=1e-3, k_0=1, sigma=0.1, delta=0.5, device=params.device)
            model.fit(train_x, train_y)

            for i in range(task_num):
                x_test_k = test_x[i]
                y_test_k = test_y[i]

                # x_train_k = train_x[i]
                y_train_k = train_y[i]

                y_pred_k = torch.zeros_like(y_test_k,dtype=float)

                for m in range(x_test_k.shape[0]):
                    y_pred_k[m] = model.high_dim_predict(x_test_k[m],i)
                

                y_pred_k = y_pred_k.to('cpu').detach().numpy()
                y_pred_k = 1 / (1 + np.exp(-y_pred_k))
                

                y_test_k = y_test_k.to('cpu').detach().numpy()
                y_train_k = y_train_k.to('cpu').detach().numpy()

                auc_ = roc_auc_score(y_test_k, y_pred_k)
                if auc_ < 0.5:
                    auc_ = 1 - auc_ 
                ret_auc[i] = ret_auc[i] + auc_

            print(f"lambda3: {lambda_3} value:{np.mean(ret_auc)}")
            return np.mean(ret_auc)
        grids = int((params.end-params.start)/params.step)
        search_space = {"lambda_3": [params.start + i*params.step for i in range(grids+1)]}
        study_MTMK_l1 = optuna.create_study(sampler=optuna.samplers.GridSampler(search_space), study_name='MTMK L2',direction='maximize')
        study_MTMK_l1.optimize(objective)
        print("MTMK_L2 optimal parameters:")
        print(study_MTMK_l1.best_params)
        print("MTMK_L2 optimal value:")
        print(study_MTMK_l1.best_value)
        # lmd2 = study_MTMK_l1.best_params['lambda_2']
        lmd2 = 0.
        lmd3 = study_MTMK_l1.best_params['lambda_3']
        # model = MtModel(kernel_list, lambda_2=0., lambda_3=params.lmd3, h=1e-6, epsilon=1e-3, k_0=1, sigma=0.1, delta=0.5, device=params.device)
    
    if params.method == 'MTSK':
        def objective(trial):
            # lambda_2 = trial.suggest_float('lambda_2',params.start,params.end)
            lambda_3 = trial.suggest_float('lambda_3',params.start,params.end)
            ret_auc = np.zeros((TASK_NUM),dtype=float)

            train_x = train_set[0]
            train_y = train_label[0]
            test_x = test_set[0]
            test_y = test_label[0]
            task_num = len(train_x)

            model = MtModel([linear()], lambda_2=0., lambda_3=lambda_3, h=1e-6, epsilon=1e-3, k_0=1, sigma=0.1, delta=0.5, device=params.device)
            model.fit(train_x, train_y)

            for i in range(task_num):
                x_test_k = test_x[i]
                y_test_k = test_y[i]

                # x_train_k = train_x[i]
                y_train_k = train_y[i]

                y_pred_k = torch.zeros_like(y_test_k,dtype=float)

                for m in range(x_test_k.shape[0]):
                    y_pred_k[m] = model.high_dim_predict(x_test_k[m],i)
                

                y_pred_k = y_pred_k.to('cpu').detach().numpy()
                y_pred_k = 1 / (1 + np.exp(-y_pred_k))
                

                y_test_k = y_test_k.to('cpu').detach().numpy()
                y_train_k = y_train_k.to('cpu').detach().numpy()

                auc_ = roc_auc_score(y_test_k, y_pred_k)
                if auc_ < 0.5:
                    auc_ = 1 - auc_ 
                ret_auc[i] = ret_auc[i] + auc_

            print(f"lambda3: {lambda_3} value:{np.mean(ret_auc)}")
            return np.mean(ret_auc)
        grids = int((params.end-params.start)/params.step)
        search_space = {"lambda_3": [params.start + i*params.step for i in range(grids+1)]}
        study_MTSK_L2 = optuna.create_study(sampler=optuna.samplers.GridSampler(search_space), study_name='MTSK L2',direction='maximize')
        study_MTSK_L2.optimize(objective)
        print("MTSK_L2 optimal parameters:")
        print(study_MTSK_L2.best_params)
        print("MTSK_L2 optimal value:")
        print(study_MTSK_L2.best_value)
        # lmd2 = study_MTSK_L2.best_params['lambda_2']
        lmd2 = 0.
        lmd3 = study_MTSK_L2.best_params['lambda_3']
    
    if params.method == 'STMK':
        def objective(trial):
            lambda_2 = trial.suggest_float('lambda_2',params.start,params.end)

            ret_auc = np.zeros((TASK_NUM),dtype=float)
            # ret1_auc = np.zeros((TASK_NUM),dtype=float)
            train_x = train_set[0]
            train_y = train_label[0]
            test_x = test_set[0]
            test_y = test_label[0]
            task_num = len(train_x)
            # model = SingleModel(kernel_list, lambda_2=params.lmd2, lambda_3=params.lmd3, h=1e-6, epsilon=1e-3, k_0=1, sigma=0.1, delta=0.5, device=params.device)
            model = SingleModel(kernel_list, lambda_2=lambda_2, lambda_3=0., h=1e-6, epsilon=1e-3, k_0=1, sigma=0.1, delta=0.5, device=params.device)
            model.fit(train_x, train_y)
            for i in range(task_num):
                x_test_k = test_x[i]
                y_test_k = test_y[i]

                # x_train_k = train_x[i]
                y_train_k = train_y[i]

                y_pred_k = torch.zeros_like(y_test_k,dtype=float)

                for m in range(x_test_k.shape[0]):
                    y_pred_k[m] = model.high_dim_predict(x_test_k[m],i)
                

                y_pred_k = y_pred_k.to('cpu').detach().numpy()
                y_pred_k = 1 / (1 + np.exp(-y_pred_k))
                

                y_test_k = y_test_k.to('cpu').detach().numpy()
                y_train_k = y_train_k.to('cpu').detach().numpy()

                auc_ = roc_auc_score(y_test_k, y_pred_k)
                if auc_ < 0.5:
                    auc_ = 1 - auc_ 
                ret_auc[i] = ret_auc[i] + auc_
                # ret1_auc[i] = ret1_auc[i] + auc1_
            print(f"lambda2: {lambda_2}; value:{np.mean(ret_auc)}")
            return np.mean(ret_auc)
        grids = int((params.end-params.start)/params.step)
        search_space = {"lambda_2": [params.start + i*params.step for i in range(grids+1)]}
        study_STMK_l1 = optuna.create_study(sampler=optuna.samplers.GridSampler(search_space), study_name='STMK L1',direction='maximize')
        study_STMK_l1.optimize(objective)
        print("STMK_L1 optimal parameters:")
        print(study_STMK_l1.best_params)
        print("STMK_L1 optimal value:")
        print(study_STMK_l1.best_value)
        lmd2 = study_STMK_l1.best_params['lambda_2']
        lmd3 = 0.    
        
    if params.method == 'STSK':
        model = SingleModel([linear()], lambda_2=0., lambda_3=params.lmd3, h=1e-6, epsilon=1e-3, k_0=1, sigma=0.1, delta=0.5, device=params.device)


    train_total = np.zeros((params.folds, TASK_NUM),dtype=float)
    train_text = np.zeros((params.folds, TASK_NUM),dtype=float)
    train_image = np.zeros((params.folds, TASK_NUM),dtype=float)

    test_total = np.zeros((params.folds, TASK_NUM),dtype=float)
    test_text = np.zeros((params.folds, TASK_NUM),dtype=float)
    test_image = np.zeros((params.folds, TASK_NUM),dtype=float)

    for k in range(params.folds):
        import datetime
        current_time = datetime.datetime.now().strftime("%Y/%m/%d %H:%M")
        print('Start of fold {} at '.format(k) + current_time)
        train_x = train_set[k]
        train_y = train_label[k]
        test_x = test_set[k]
        test_y = test_label[k] 

        if params.method == 'MTMK L1':
            model = MtModel(kernel_list, lambda_2=lmd2, lambda_3=0., h=1e-6, epsilon=1e-4, k_0=1, sigma=0.1, delta=0.5, device=params.device)
        if params.method == 'MTMK mix':
            model = MtModel(kernel_list, lambda_2=lmd2, lambda_3=lmd3, h=1e-6, epsilon=1e-4, k_0=1, sigma=0.1, delta=0.5, device=params.device)
        if params.method == 'MTMK L2':
            model = MtModel(kernel_list, lambda_2=0., lambda_3=lmd3, h=1e-6, epsilon=1e-4, k_0=1, sigma=0.1, delta=0.5, device=params.device)
        if params.method == 'MTSK':
            model = MtModel([linear()], lambda_2=0., lambda_3=lmd3, h=1e-6, epsilon=1e-4, k_0=1, sigma=0.1, delta=0.5, device=params.device)
        if params.method == 'STMK':
            model = SingleModel(kernel_list, lambda_2=lmd2, lambda_3=0., h=1e-6, epsilon=1e-4, k_0=1, sigma=0.1, delta=0.5, device=params.device)
        if params.method == 'STSK':
            model = SingleModel([linear()], lambda_2=0., lambda_3=lmd3, h=1e-6, epsilon=1e-4, k_0=1, sigma=0.1, delta=0.5, device=params.device)
            
        if params.method in ['MTMK mix', 'MTMK L1', 'MTMK L2','MTSK','STMK','STSK']:
            model.train_phecode = train_phecode[k]
            model.test_phecode = test_phecode[k]
        
        # only train on text                
        if params.textonly:
            train_x_text = copy.deepcopy(train_x)
            train_y_text = copy.deepcopy(train_y)
            for i in range(TASK_NUM):
                train_x_text[i] = train_x_text[i][0:int(0.8 * TEXT_NUM)]
                train_y_text[i] = train_y_text[i][0:int(0.8 * TEXT_NUM)]
            model.fit(train_x_text, train_y_text)
        
        # only train on images
        elif params.imageonly:
            train_x_image = copy.deepcopy(train_x)
            train_y_image = copy.deepcopy(train_y)
            for i in range(TASK_NUM):
                train_x_image[i] = train_x_image[i][int(0.8 * TEXT_NUM):]
                train_y_image[i] = train_y_image[i][int(0.8 * TEXT_NUM):]
            model.fit(train_x_image, train_y_image)
        # train on both text and image
        else:
            model.fit(train_x,train_y)
        
        if params.savemodel:
            if not os.path.exists(params.savedir):
                os.mkdir(params.savedir)
            
            current_time = datetime.datetime.now().strftime("%Y_%m_%d %H_%M")
            torch.save(model, params.savedir + '/' + current_time + ' ' + 'FOLD ' + str(k) + ' ' + params.method + '.pt')
            
        # ____________________________ 
        # AUC of TASKS
        for i in range(TASK_NUM):
            x_test_k = test_x[i]
            y_test_k = test_y[i]

            x_train_k = train_x[i]
            y_train_k = train_y[i]
            
            test_pred_k = torch.zeros_like(y_test_k, dtype=float)
            train_pred_k = torch.zeros_like(y_train_k, dtype=float)
            
            test_pred_k = model.high_dim_predict_nightly(x_test_k,i)
            train_pred_k = model.high_dim_predict_nightly(x_train_k,i)
            
            # for m in range(x_test_k.shape[0]):
            #     test_pred_k[m] = model.high_dim_predict(x_test_k[m],i)
            
            # for m in range(x_train_k.shape[0]):
            #     train_pred_k[m] = model.high_dim_predict(x_train_k[m],i)
                
            test_pred_k = 1 / (1 + torch.exp(- test_pred_k))
            train_pred_k = 1 / (1 + torch.exp(- train_pred_k))
            
            y_train_k = y_train_k.to('cpu').detach().numpy()
            y_test_k = y_test_k.to('cpu').detach().numpy()
            train_pred_k = train_pred_k.to('cpu').detach().numpy()
            test_pred_k = test_pred_k.to('cpu').detach().numpy()
            
            train_total_auc = roc_auc_score(y_train_k, train_pred_k)
            test_total_auc = roc_auc_score(y_test_k, test_pred_k)
            train_text_auc = roc_auc_score(y_train_k[0:int(TEXT_NUM * 2 * 0.8)], train_pred_k[0:int(TEXT_NUM * 2 * 0.8)])
            test_text_auc = roc_auc_score(y_test_k[0:int(TEXT_NUM * 2 * 0.2)], test_pred_k[0:int(TEXT_NUM * 2 * 0.2)])
            train_image_auc = roc_auc_score(y_train_k[int(TEXT_NUM * 2 * 0.8):], train_pred_k[int(TEXT_NUM * 2 * 0.8):])
            test_image_auc = roc_auc_score(y_test_k[int(TEXT_NUM * 2 * 0.2):], test_pred_k[int(TEXT_NUM * 2 * 0.2):])
            
            train_total[k][i] = train_total_auc
            train_text[k][i] = train_text_auc
            train_image[k][i] = train_image_auc
            test_total[k][i] = test_total_auc
            test_text[k][i] = test_text_auc
            test_image[k][i] = test_image_auc
        
        if params.onefold:
            break
        
    # save log 
    current_time = datetime.datetime.now().strftime("%Y_%m_%d %H_%M")
    results = []
    for k in range(params.folds):
        for i in range(TASK_NUM):
            results.append([params.method, k, i, 'train', 'total', train_total[k][i]])
            results.append([params.method, k, i, 'train', 'image', train_image[k][i]])
            results.append([params.method, k, i, 'train', 'text', train_text[k][i]])
            results.append([params.method, k, i, 'test', 'total', test_total[k][i]])
            results.append([params.method, k, i, 'test', 'image', test_image[k][i]])
            results.append([params.method, k, i, 'test', 'text', test_text[k][i]])
        # average results
        results.append([params.method, k, 'AVG', 'train', 'total', np.mean(train_total[k])])
        results.append([params.method, k, 'AVG', 'train', 'image', np.mean(train_image[k])])
        results.append([params.method, k, 'AVG', 'train', 'text', np.mean(train_text[k])])
        results.append([params.method, k, 'AVG', 'test', 'total', np.mean(test_total[k])])
        results.append([params.method, k, 'AVG', 'test', 'image', np.mean(test_image[k])])
        results.append([params.method, k, 'AVG', 'test', 'text', np.mean(test_text[k])])
    # average results of k folds
    for i in range(TASK_NUM):
        results.append([params.method, 'AVG', i, 'train', 'total', np.mean(train_total, axis=0)[i]])
        results.append([params.method, 'AVG', i, 'train', 'image', np.mean(train_image, axis=0)[i]])
        results.append([params.method, 'AVG', i, 'train', 'text', np.mean(train_text, axis=0)[i]])
        results.append([params.method, 'AVG', i, 'test', 'total', np.mean(test_total, axis=0)[i]])
        results.append([params.method, 'AVG', i, 'test', 'image', np.mean(test_image, axis=0)[i]])
        results.append([params.method, 'AVG', i, 'test', 'text', np.mean(test_text, axis=0)[i]])
    results.append([params.method, 'AVG', 'AVG', 'train', 'total', np.mean(train_total)])
    results.append([params.method, 'AVG', 'AVG', 'train', 'image', np.mean(train_image)])
    results.append([params.method, 'AVG', 'AVG', 'train', 'text', np.mean(train_text)])
    results.append([params.method, 'AVG', 'AVG', 'test', 'total', np.mean(test_total)])
    results.append([params.method, 'AVG', 'AVG', 'test', 'image', np.mean(test_image)])
    results.append([params.method, 'AVG', 'AVG', 'test', 'text', np.mean(test_text)])
    
    results = pd.DataFrame(results, columns=['Model', 'Fold', 'Task', 'Train/Test', 'Type', 'AUC'])
    
    if not os.path.exists(params.resdir):
        os.mkdir(params.resdir)
                
    results.to_csv(params.resdir + '/' + current_time + ' ' + params.method + '_' + str(params.samples) + '_' + str(params.texts) + '_' + str(lmd2) + '_' + str(lmd3) + '.csv', index=False)
    

