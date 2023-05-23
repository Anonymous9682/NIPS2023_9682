import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd

class MtModel(nn.Module):
    def __init__(self, kernel_list, lambda_2, lambda_3, h, epsilon, k_0, sigma, delta, device) -> None:
        super().__init__()
        self.kernel_list = kernel_list
        self.lambda_2 = lambda_2
        self.lambda_3 = lambda_3
        self.h = h
        self.epsilon = epsilon
        self.k_0 = k_0
        self.sigma = sigma
        self.delta = delta
        self.alpha = None
        self.X_train = None
        self.Y_train = None
        self.task_num = 0
        self.device = device

    def fit(self, mt_X:list, mt_y:list):
        for idx in range(len(mt_X)):
            mt_X[idx].to(self.device)
            mt_y[idx].to(self.device)

        mt_alpha = self.alpha_train(mt_X, mt_y, self.kernel_list,
                            self.lambda_2, self.lambda_3, self.h,
                            self.epsilon, self.k_0, self.sigma,
                            self.delta)
        self.alpha = mt_alpha
        self.X_train = mt_X
        self.Y_train = mt_y
        self.task_num = len(mt_alpha)
        # print("Successfully fitted")

    def predict(self, x_test, task):
        f_hat = 0.
        for m in range(self.alpha[task].shape[0]):
            # f_hat += torch.dot(self.alpha[task][m], self.kernel_list[m](self.X_train[task], x_test.view(1, -1)).squeeze())
            # print(x_test.shape)
            f_hat += torch.dot(self.alpha[task][m], self.kernel_list[m](self.X_train[task], x_test.squeeze()).squeeze())
            
        return f_hat
    
    def high_dim_predict(self, x_test, task):
        # 对samples递归
        f_hat = 0.
        # m: kernel
        for m in range(self.alpha[task].shape[0]):
            # i: samples
            for i in range(self.alpha[task].shape[1]):
                f_hat += self.alpha[task][m][i] * self.kernel_list[m](self.X_train[task][i].reshape(1,-1), x_test.reshape(1,-1)).squeeze().squeeze()        
        return f_hat
    
    def high_dim_predict_nightly(self, x_test, task):
        # if one test point
        if len(x_test.shape) == 1 or x_test.shape[0] == 1:
            f_hat = 0.
            for m in range(self.alpha[task].shape[0]):    
                f_hat += self.alpha[task][m].double() @ self.kernel_list[m](self.X_train[task].double(), x_test.reshape(1, -1).double()) 
        # if multiple test points, assume shape is N, D
        else:
            f_hat = torch.zeros(x_test.shape[0], device=self.device).double()
            for m in range(self.alpha[task].shape[0]):
                f_hat += self.alpha[task][m].double() @ self.kernel_list[m](self.X_train[task].double(), x_test.double())
        return f_hat
            
    def predict_proba(self, x_test, task):
        fhat = self.high_dim_predict(x_test,task)
        return torch.sigmoid(fhat).item()

    def alpha_train(self, mt_X, mt_y, kernel_list, lambda_2, lambda_3, h,
                epsilon, k_0, sigma, delta):
        # initialize mt_alpha and mt_kernel_matrix_list
        with torch.no_grad():
            mt_alpha = []
            mt_kernel_matrix_list = []
            for task in range(len(mt_y)):
                mt_alpha.append(torch.zeros(len(kernel_list), mt_y[task].shape[0], device=self.device))
                mt_kernel_matrix_list.append(torch.zeros(len(kernel_list), mt_y[task].shape[0], mt_y[task].shape[0], device=self.device))

            # compute gram matrices in mt_kernel_matrix_list
            for task in range(len(mt_y)):
                for m in range(len(kernel_list)):
                    kernel = kernel_list[m]
                    mt_kernel_matrix_list[task][m] = kernel(mt_X[task])

            # compute gamma and W
            gamma = self.compute_gamma(mt_alpha)
            # W = self.compute_W(mt_kernel_matrix_list)

            loss_prev = 2 / epsilon
            loss_now = 1 / epsilon
            while loss_prev - loss_now > epsilon:
                loss_prev = loss_now
                for m in range(len(kernel_list)):
                    # nabla l_m , K^-1 nabla l_m
                    gm, k_inv_gm = self.gradient_gamma_m(mt_alpha, mt_kernel_matrix_list, mt_y, m)
                    # q0 = quadra_calculator(gamma[m], W[m])
                    # another way to compute q0
                    q0 = 0.
                    for task in range(len(mt_alpha)):
                        q0 += self.quadra_calculator(mt_alpha[task][m], mt_kernel_matrix_list[task][m])

                    temp = gm.dot(k_inv_gm) + 2 * h * gamma[m].dot(gm) + h ** 2 * q0
                    # if temp is too small and becomes negative due to underflow
                    if temp > 0:
                        norm = torch.sqrt(temp)
                    else:
                        norm = torch.tensor(1e-6, device=self.device)

                    if torch.le(norm, lambda_2):
                        d_m = - gamma[m]
                        # if all zero
                        if len(torch.nonzero(d_m)) < 1:
                            continue
                    else:
                        temp = k_inv_gm + h * gamma[m]
                        d_m = 1. / (h + 2 * lambda_3) * (temp - lambda_2 * temp / norm) - gamma[m]

                    k = self.armijo_line_search(k_0, sigma, delta, lambda_2, lambda_3,
                                        mt_alpha, d_m, mt_kernel_matrix_list, mt_y, gm, m)
                    gamma[m] = gamma[m] + k * d_m

                    mt_alpha = self.compute_mt_alpha(gamma, mt_alpha)
                    ll_now = self.mt_log_likelihood(mt_alpha, mt_kernel_matrix_list, mt_y)
                    loss_now = self.mt_loss(ll_now, lambda_2, lambda_3, mt_kernel_matrix_list, mt_alpha)
                    # print(f"loss now:{loss_now}")              
            # print(f"loss now:{loss_now}") 
            return mt_alpha
    
    def mt_log_likelihood(self, mt_alpha, mt_kernel_matrix_list, mt_y):
        mt_ll = 0.
        for task in range(len(mt_alpha)):
            mt_ll += self.log_likelihood(mt_alpha[task], mt_kernel_matrix_list[task], mt_y[task])
        return mt_ll
    
    def log_likelihood(self, alpha, kernel_matrix_list, y):
        sum_Kalpha = torch.zeros(y.shape[0], device=self.device)
        for m in range(len(alpha)):
            sum_Kalpha += kernel_matrix_list[m].mm(alpha[m].unsqueeze(1)).squeeze()
        ll = -1. / alpha[0].shape[0] * torch.sum(torch.log(1 + torch.exp(-y * sum_Kalpha)))
        return ll
    
    def mt_loss(self, mt_ll, lambda_2, lambda_3, mt_kernel_matrix_list, mt_alpha):
        # sum of quadratic over tasks, K kernels
        sum_qt = torch.zeros(mt_alpha[0].shape[0], device=self.device)
        for j in range(mt_alpha[0].shape[0]):
            for task in range(len(mt_alpha)):
                sum_qt[j] += self.quadra_calculator(mt_alpha[task][j], mt_kernel_matrix_list[task][j])
        l1 = lambda_2 * torch.sum(torch.sqrt(sum_qt))
        l2 = lambda_3 * torch.sum(sum_qt)
        return - mt_ll + l1 + l2
    
    def gradient_gamma_m(self, mt_alpha, mt_kernel_matrix_list, mt_y, m):
        g_gamma_m = []
        kg_gamma_m = []
        for task in range(len(mt_alpha)):
            g_task, kg_task = self.gradient_alpha_m(mt_alpha[task], mt_kernel_matrix_list[task], mt_y[task], m)
            g_gamma_m.append(g_task.squeeze())
            kg_gamma_m.append(kg_task.squeeze())
        g_gamma_m = torch.cat(g_gamma_m)
        kg_gamma_m = torch.cat(kg_gamma_m)
        return g_gamma_m, kg_gamma_m
    
    def compute_gamma(self, mt_alpha):
        return torch.cat(mt_alpha, dim=1)
    
    def compute_mt_alpha(self, gamma, mt_alpha_prev):
        mt_alpha = []
        idx = 0
        for task in range(len(mt_alpha_prev)):
            alpha_task = gamma[:, idx:idx + mt_alpha_prev[task].shape[1]]
            mt_alpha.append(alpha_task)
            idx += mt_alpha_prev[task].shape[1]
        return mt_alpha

    def compute_W(self, mt_kernel_matrix_list):
        W = []
        for j in range(mt_kernel_matrix_list[0].shape[0]):
            W_j = torch.block_diag(*[mt_kernel_matrix_list[task][j] for task in range(len(mt_kernel_matrix_list))])
            W.append(W_j)
        W = torch.stack(W, dim=0)
        return W
    
    def armijo_line_search(self, k_0, sigma, delta, lambda_2, lambda_3,
                       mt_alpha, d_m, mt_kernel_matrix_list, mt_y, g_gamma_m, m):
        gamma = self.compute_gamma(mt_alpha)
        gamma_m = gamma[m]
        W_m = torch.block_diag(*[mt_kernel_matrix_list[task][m] for task in range(len(mt_kernel_matrix_list))])

        q1 = self.quadra_calculator(gamma_m + d_m, W_m)
        q0 = self.quadra_calculator(gamma_m, W_m)

        Delta = - torch.dot(d_m, g_gamma_m) + lambda_2 * (torch.sqrt(q1) - torch.sqrt(q0)) + lambda_3 * (q1 - q0)

        mt_ll = self.mt_log_likelihood(mt_alpha, mt_kernel_matrix_list, mt_y)
        loss = self.mt_loss(mt_ll, lambda_2, lambda_3, mt_kernel_matrix_list, mt_alpha)

        # search for the largest k
        l = 0
        k = 1
        while k > 1e-10:
            k = k_0 * delta ** l
            gamma_k = gamma.clone()
            gamma_k[m] += k * d_m
            mt_alpha_k = self.compute_mt_alpha(gamma_k, mt_alpha)
            mt_ll_ = self.mt_log_likelihood(mt_alpha_k, mt_kernel_matrix_list, mt_y)
            loss_ = self.mt_loss(mt_ll_, lambda_2, lambda_3, mt_kernel_matrix_list, mt_alpha_k)

            if torch.le(loss_ - loss, k * sigma * Delta):
                break
            else:
                l += 1
        k = k_0 * delta ** l
        if k < 1e-10:
            k = 0
        return k
    
    def quadra_calculator(self, alpha_m: torch.tensor, K_m: torch.tensor):
        return alpha_m.dot((K_m.mm(alpha_m.unsqueeze(1))).squeeze())
    
    def gradient_alpha_m(self, alpha, kernel_matrix_list, y, m):
        cons = torch.zeros(y.shape[0], device=self.device)
        for j in range(len(alpha)):
            cons += kernel_matrix_list[j].mm(alpha[j].unsqueeze(1)).squeeze()
        cons = torch.div(torch.exp(-y * cons), (1 + torch.exp(-y * cons))) * (-y)

        kg = - 1. / alpha[0].shape[0] * cons
        g = kernel_matrix_list[m].mm(kg.unsqueeze(1)).squeeze()
        # g = - 1. / alpha[0].shape[0] * kernel_matrix_list[m].mm(cons.unsqueeze(1)).squeeze()
        return g, kg
    

class SingleModel(nn.Module):
    def __init__(self, kernel_list, lambda_2, lambda_3, h, epsilon, k_0, sigma, delta, device) -> None:
        super().__init__()
        self.kernel_list = kernel_list
        self.lambda_2 = lambda_2
        self.lambda_3 = lambda_3
        self.h = h
        self.epsilon = epsilon
        self.k_0 = k_0
        self.sigma = sigma
        self.delta = delta
        # self.alpha = None
        # self.X_train = None
        self.task_num = 0
        self.device = device

    def fit(self, mt_X:list, mt_y:list):   
        self.task_num = len(mt_X)
        self.Models = []
        for t in range(self.task_num):
            x = mt_X[t].to('cuda')
            y = mt_y[t].to('cuda')
            model = MtModel(self.kernel_list,
                            self.lambda_2, self.lambda_3, self.h,
                            self.epsilon, self.k_0, self.sigma,
                            self.delta,'cuda')
            model.fit([x],[y])
            self.Models.append(model)
    
    def predict(self, x_test, task):
        f_hat = 0.
        model = self.Models[task]
        for m in range(model.alpha[0].shape[0]):    
            f_hat += torch.dot(model.alpha[0][m], self.kernel_list[m](model.X_train[0], x_test.view(1, -1)).squeeze())
        return f_hat

    
    def high_dim_predict(self, x_test, task):
        f_hat = 0.
        model = self.Models[task]
        # 对kernels计算
        for m in range(model.alpha[0].shape[0]):    
            for i in range(model.alpha[0].shape[1]):
                f_hat += model.alpha[0][m][i].item()* self.kernel_list[m](model.X_train[0][i].reshape(1,-1), x_test.reshape(1,-1)).squeeze()
        return f_hat
    

    def high_dim_predict_nightly(self, x_test, task):
        # if one test point
        model = self.Models[task]
        if len(x_test.shape) == 1 or x_test.shape[0] == 1:
            f_hat = 0.
            for m in range(model.alpha[0].shape[0]):    
                f_hat += model.alpha[0][m].double() @ self.kernel_list[m](model.X_train[0].double(), x_test.reshape(1, -1).double()) 
        # if multiple test points, assume shape is N, D
        else:
            f_hat = torch.zeros(x_test.shape[0], device=self.device).double()
            for m in range(model.alpha[0].shape[0]):
                f_hat += model.alpha[0][m].double() @ self.kernel_list[m](model.X_train[0].double(), x_test.double())
        return f_hat

    def predict_proba(self, x_test, task):
        fhat = self.high_dim_predict(x_test,task)
        
        return torch.sigmoid(fhat).item()

    def save_model(self, path, path1):
        # 存alpha数据
        torch.save(self.alpha, path)
        # 存训练集合数据
        torch.save(self.X_train, path1)

    def load_model(self, path, path1):
        self.alpha = torch.load(path)
        self.X_train = torch.load(path1)

