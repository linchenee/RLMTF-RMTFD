import scipy.io as sio
from argparse import ArgumentParser
from time import time
import math
import matplotlib.pyplot as plt
import torch.nn.functional as F
from models import *
from util.common_utils import *
parser = ArgumentParser(description='RMFD')
parser.add_argument('--data_dir', type=str, default='data', help='data directory')
parser.add_argument('--learning_rate', type=float, default=1e-3, help='learning rate')
parser.add_argument('--num_iter', type=int, default=1000, help='number of iterations')
parser.add_argument('--inner_iter', type=int, default=2, help='number of inner iterations used to update network parameters')
parser.add_argument('--show_iter', type=int, default=100, help='show the loss during the iteration')
parser.add_argument('--lambda1', type=float, default=2, help='regularization parameter')
parser.add_argument('--lambda2', type=float, default=1, help='regularization parameter')
parser.add_argument('--rank', type=int, default=80, help='rank of the matrix')
parser.add_argument('--p', type=float, default=0.8, help='parameter that controls the weight allocation in the reweighted nuclear norm (0<p<=1)')
parser.add_argument('--varepsilon', type=float, default=1e-8, help='small constant to avoid dividing by zero in the (re)weighted nuclear norm')
parser.add_argument('--c', type=float, default=1, help='compromising constant in the (re)weighted nuclear norm')
parser.add_argument('--mu', type=float, default=1e-3, help='penalty parameter in the ADMM (mu>0)')
parser.add_argument('--mu_max', type=float, default=10, help='maximum of the penalty parameter')
parser.add_argument('--rho', type=float, default=1.01, help='update rate of the penalty parameter')
parser.add_argument('--tol', type=float, default=1e-2, help='termination tolerance')
args = parser.parse_args()
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
learning_rate = args.learning_rate
num_iter = args.num_iter
inner_iter = args.inner_iter
show_iter = args.show_iter
lambda1 = args.lambda1
lambda2 = args.lambda2
rank = args.rank
p = args.p
varepsilon = args.varepsilon
c = args.c
mu = args.mu
mu_max = args.mu_max
rho = args.rho
tol = args.tol

num_img = 36 # number of images
n1, n2, n3 = 256, 256, 1 # size of the image
pad = 'reflection'
test_mat = sio.loadmat('./%s/%s' % (args.data_dir, 'USC-SIPI_salt_and_ pepper_noise_10dB.mat'))
test_data = test_mat['Y']
INPUT = 'noise'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def psnr(img1, img2, M_comp):
    mse = np.sum(((img1 - img2) * M_comp) ** 2) / np.sum(M_comp)
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def prox_w(A, W1, alpha, mu):
    # Proximal operator of the (re)weighted Frobenius norm
    L, S, R = torch.svd(A)
    W2 = torch.mul(S, 1 / (alpha / mu * W1 + 1))
    return [(L @ torch.diag(W2) @ torch.t(R)).detach(), W2.detach()]

def RMFD(Y, img_id, img_gt, M, M_comp, mu, net_input):
    net = get_net(n3, 'skip', pad, n_channels=n3, skip_n33d=128, skip_n33u=128, skip_n11=4, num_scales=5, upsample_mode='bilinear').type(torch.cuda.FloatTensor).to(device)
    U1 = torch.randn(n1, rank).to(device)
    V1 = torch.randn(rank, n2).to(device)
    L1, W_u, R1 = torch.svd(U1)
    L2, W_v, R2 = torch.svd(V1)
    U = U1
    V = V1
    X = U @ V
    Z1 = torch.zeros(n1, rank).to(device)
    Z2 = torch.zeros(rank, n2).to(device)
    Z3 = torch.zeros(n1, n2).to(device)

    for i in range(num_iter):
        U = ((U1 + Z1 / mu + (X - Z3 / mu) @ torch.t(V)) @ torch.pinverse(eye_rank + V @ torch.t(V))).detach()
        V = (torch.pinverse(eye_rank + torch.t(U) @ U) @ (V1 + Z2 / mu + torch.t(U) @ (X - Z3 / mu))).detach()
        weight = (c * torch.pow(torch.mul(W_u, W_v) + varepsilon, p - 1)).detach()
        [U1, W_u] = prox_w(U - Z1 / mu, weight, lambda1, mu)
        weight = (c * torch.pow(torch.mul(W_u, W_v) + varepsilon, p - 1)).detach()
        [V1, W_v] = prox_w(V - Z2 / mu, weight, lambda1, mu)
        optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
        B = (U @ V + Z3 / mu).detach()
        for j in range(inner_iter):
            optimizer.zero_grad()
            net_output = net(net_input)
            total_loss = norm2_loss(net_output - B) * mu * 0.5 + lambda2 * norm1_loss(torch.mul(M, net_output - Y))
            total_loss.backward()
            optimizer.step()
        X = net(net_input).view(n1, n2).detach()
        Z1 = (Z1 + mu * (U1 - U)).detach()
        Z2 = (Z2 + mu * (V1 - V)).detach()
        Z3 = (Z3 + mu * (U @ V - X)).detach()
        mu = min(rho * mu, mu_max)
        loss_U = torch.pow(torch.norm(U - U1, p='fro') / torch.norm(U1, p='fro'), 2)
        loss_V = torch.pow(torch.norm(V - V1, p='fro') / torch.norm(V1, p='fro'), 2)
        loss_UV = torch.pow(torch.norm(U @ V - X, p='fro') / torch.norm(U @ V, p='fro'), 2)
        sum_loss = loss_U + loss_V + loss_UV
        if sum_loss < tol:
            break
        if i % show_iter == 0:
            rec_PSNR = psnr(np.clip(X.cpu().numpy() * 255, 0, 255), img_gt * 255, M_comp.cpu().numpy())
            print("[%02d] Iter %02d, loss_U %.5f, loss_V %.5f, loss_UV %.5f, loss_Adam %.5f, PSNR %.2f, mu %.4f, sum_loss %.4f" % (img_id, i, loss_U, loss_V, loss_UV, total_loss.item()/n1/n2, rec_PSNR, mu, sum_loss))

    rec_img = np.clip(X.cpu().numpy() * 255, 0, 255)
    rec_PSNR = psnr(rec_img, img_gt * 255, M_comp.cpu().numpy())
    return rec_img, rec_PSNR

torch.manual_seed(3)
rec_PSNR = np.zeros((num_img, 1))
rec_img = np.zeros((num_img, n1, n2))
eye_rank = torch.eye(rank).to(device)
net_input = get_noise(n3, INPUT, (n1, n2)).type(torch.cuda.FloatTensor).to(device)
for i in range(num_img):
    data0 = torch.from_numpy(test_data[i, 0:n1*n2]).type(torch.FloatTensor) * (1 / 255)
    data1 = torch.from_numpy(test_data[i, n1*n2:n1*n2*2]).type(torch.FloatTensor) * (1 / 255)
    data2 = torch.from_numpy(test_data[i, n1*n2*2:n1*n2*3]).type(torch.FloatTensor)
    img_gt = data0.view(n1, n2).numpy()  # groundtruth
    img_noisy = data1.view(n1, n2).to(device)  # noisy image
    M = data2.view(n1, n2).to(device)  # binary mask that indicates the locations of missing and known entries in the tensor
    M_comp = torch.ones(n1, n2).to(device) - M
    Y = torch.mul(img_noisy, M).to(device)
    rec_img[i, :, :], rec_PSNR[i] = RMFD(Y, i, img_gt, M, M_comp, mu, net_input)
    print("[%02d] PSNR achieved by RMFD is %.4f" % (i, rec_PSNR[i]))

# sio.savemat('Result_USC-SIPI_salt_and_ pepper_noise_10dB.mat', {'Result': rec_img})
print(rec_PSNR)
print(np.mean(rec_PSNR))