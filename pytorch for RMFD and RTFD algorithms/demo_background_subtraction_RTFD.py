import scipy.io as sio
from argparse import ArgumentParser
from time import time
import math
import torch_dct as dct
import matplotlib.pyplot as plt
import torch.nn.functional as F
from models import *
from util.common_utils import *
parser = ArgumentParser(description='RTFD')
parser.add_argument('--data_dir', type=str, default='data', help='data directory')
parser.add_argument('--learning_rate', type=float, default=1e-2, help='learning rate')
parser.add_argument('--num_iter', type=int, default=1100, help='number of iterations')
parser.add_argument('--show_iter', type=int, default=100, help='show the loss during the iteration')
parser.add_argument('--lambda1', type=float, default=1e2, help='regularization parameter')
parser.add_argument('--lambda2', type=float, default=0.1, help='regularization parameter')
parser.add_argument('--p', type=float, default=0.8, help='parameter that controls the weight allocation in the reweighted nuclear norm (0<p<=1)')
parser.add_argument('--varepsilon', type=float, default=1e-8, help='small constant to avoid dividing by zero in the (re)weighted nuclear norm')
parser.add_argument('--c', type=float, default=1, help='compromising constant in the (re)weighted nuclear norm')
parser.add_argument('--mu', type=float, default=1e-3, help='penalty parameter in the ADMM (mu>0)')
parser.add_argument('--mu_max', type=float, default=10, help='maximum of the penalty parameter')
parser.add_argument('--rho', type=float, default=1.01, help='update rate of the penalty parameter')
parser.add_argument('--tol', type=float, default=5e-3, help='termination tolerance')
args = parser.parse_args()
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
learning_rate = args.learning_rate
num_iter = args.num_iter
show_iter = args.show_iter
lambda1 = args.lambda1
lambda2 = args.lambda2
p = args.p
varepsilon = args.varepsilon
c = args.c
mu = args.mu
mu_max = args.mu_max
rho = args.rho
tol = args.tol

num_img = 1  # number of images
n1, n2, n3 = 256, 256, 100 # size of the image
rank1 = 200  # tensor multi-rank = [rank1, rank2*ones(1, n3-1)]
rank2 = 1

pad = 'reflection'
test_mat = sio.loadmat('./%s/%s' % (args.data_dir, 'SBI.mat'))
test_data = test_mat['Y']
INPUT = 'noise'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
eye_rank = [torch.eye(rank1).to(device)] + [torch.eye(rank2).to(device) for _ in range(n3 - 1)]

def psnr3D(img1, img2):
    PIXEL_MAX = 255.0
    PSNR_SUM = 0
    for i in range(n3):
      mse = np.sum((img1[:, :, i] - img2[:, :, i]) ** 2) / (n1 * n2)
      PSNR_SUM = PSNR_SUM + 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
    return PSNR_SUM / n3

def prox_w(A, W1, alpha, mu):
    # Proximal operator of the (re)weighted Frobenius norm
    L, S, R = torch.svd(A)
    W2 = torch.mul(S, 1 / (alpha / mu * W1 + 1))
    return (L @ torch.diag(W2) @ torch.t(R)).detach(), W2.detach()

def RTFD(Y, img_id, img_gt, M, M_comp, mu):
    Y_4D = torch.transpose(torch.transpose(Y, 2, 1), 1, 0).view(1, n3, n1, n2)
    M_4D = torch.transpose(torch.transpose(M, 2, 1), 1, 0).view(1, n3, n1, n2)
    net_input = get_noise(n3, INPUT, (n1, n2)).type(torch.cuda.FloatTensor).detach()
    net = get_net(n3, 'skip', pad, n_channels=n3, skip_n33d=128, skip_n33u=128, skip_n11=4, num_scales=5, upsample_mode='bilinear').type(torch.cuda.FloatTensor)
    net = net.type(torch.cuda.FloatTensor).to(device)
    Xd = dct.dct(torch.mul(torch.ones(n1, n2, n3).to(device), M_comp) + torch.mul(Y, M), norm='ortho')  # X in the DCT domain
    temp = Xd
    Ud = [torch.empty(n1, rank1).to(device)] + [torch.empty(n1, rank2).to(device) for _ in range(n3 - 1)]  # U in the DCT domain
    Vd = [torch.empty(rank1, n2).to(device)] + [torch.empty(rank2, n2).to(device) for _ in range(n3 - 1)]  # V in the DCT domain
    U1d = [torch.empty(n1, rank1).to(device)] + [torch.empty(n1, rank2).to(device) for _ in range(n3 - 1)]  # U1 in the DCT domain
    V1d = [torch.empty(rank1, n2).to(device)] + [torch.empty(rank2, n2).to(device) for _ in range(n3 - 1)]  # V1 in the DCT domain
    W_u = [torch.empty(rank1).to(device)] + [torch.empty(rank2).to(device) for _ in range(n3 - 1)]  # weights in the reweighted Frobenius norm of U
    W_v = [torch.empty(rank1).to(device)] + [torch.empty(rank2).to(device) for _ in range(n3 - 1)]  # weights in the reweighted Frobenius norm of V
    for i in range(n3):
       if i == 0:
           subspace = rank1
       else:
           subspace = rank2
       L0, S0, R0 = torch.svd(Xd[:, :, i])
       W_u[i] = torch.sqrt(S0[0:subspace])
       W_v[i] = W_u[i]
       Ud[i] = L0[:, 0:subspace] @ torch.diag(W_u[i])
       U1d[i] = Ud[i]
       Vd[i] = torch.diag(W_u[i]) @ torch.t(R0[:, 0:subspace])
       V1d[i] = Vd[i]
    Z1 = [torch.zeros(n1, rank1).to(device)] + [torch.zeros(n1, rank2).to(device) for _ in range(n3 - 1)]  # Lagrange multipliers
    Z2 = [torch.zeros(rank1, n2).to(device)] + [torch.zeros(rank2, n2).to(device) for _ in range(n3 - 1)]  # Lagrange multipliers
    Z3 = torch.zeros(n1, n2, n3).to(device)  # Lagrange multipliers
    UVd = torch.empty(n1, n2, n3).to(device)
    for j in range(num_iter):
        # norms = torch.zeros(4).to(device)
        for i in range(n3):
           Ud[i] = ((U1d[i] + Z1[i] / mu + temp[:, :, i] @ torch.t(Vd[i])) @ torch.pinverse(eye_rank[i] + Vd[i] @ torch.t(Vd[i]))).detach()
           Vd[i] = (torch.pinverse(eye_rank[i] + torch.t(Ud[i]) @ Ud[i]) @ (V1d[i] + Z2[i] / mu + torch.t(Ud[i]) @ temp[:, :, i])).detach()
           weight = (c * torch.pow(torch.mul(W_u[i], W_v[i]) + varepsilon, p - 1)).detach()
           U1d[i], W_u[i] = prox_w(Ud[i] - Z1[i] / mu, weight, lambda1, mu)
           weight = (c * torch.pow(torch.mul(W_u[i], W_v[i]) + varepsilon, p - 1)).detach()
           V1d[i], W_v[i] = prox_w(Vd[i] - Z2[i] / mu, weight, lambda1, mu)
           UVd[:, :, i] = (Ud[i] @ Vd[i]).detach()
           Z1[i] = (Z1[i] + mu * (U1d[i] - Ud[i])).detach()
           Z2[i] = (Z2[i] + mu * (V1d[i] - Vd[i])).detach()
        #    norms[0] += torch.pow(torch.norm(Ud[i] - U1d[i], p='fro'), 2)
        #    norms[1] += torch.pow(torch.norm(U1d[i], p='fro'), 2)
        #    norms[2] += torch.pow(torch.norm(Vd[i] - V1d[i], p='fro'), 2)
        #    norms[3] += torch.pow(torch.norm(V1d[i], p='fro'), 2)
        # loss_U = norms[0] / norms[1]
        # loss_V = norms[2] / norms[3]
        UV = dct.idct(UVd, norm='ortho').detach()
        optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
        B = torch.transpose(torch.transpose(UV + Z3 / mu, 2, 1), 1, 0).view(1, n3, n1, n2).detach()
        if j < 500:
            inner_iter = 1
        else:
            inner_iter = int(j / 50)  # number of inner iterations used to update network parameters
        for i in range(inner_iter):
            optimizer.zero_grad()
            net_output = net(net_input)
            total_loss = norm2_loss(net_output - B) * mu * 0.5 + lambda2 * norm1_loss(torch.mul(M_4D, net_output - Y_4D))
            total_loss.backward()
            optimizer.step()
        X = torch.transpose(torch.transpose(net(net_input).view(n3, n1, n2), 0, 1), 1, 2).detach()
        Z3 = (Z3 + mu * (UV - X)).detach()
        temp = dct.dct(X - Z3 / mu, norm='ortho').detach()
        mu = min(rho * mu, mu_max)
        # loss_UV = torch.pow(torch.norm(UV - X, p='fro') / torch.norm(UV, p='fro'), 2)
        # sum_loss = loss_U + loss_V + loss_UV
        # if sum_loss < tol:
        #     break
        if j % show_iter == 0:
            rec_PSNR = psnr3D(X.detach().cpu().numpy() * 255, img_gt * 255)
            # print("[%02d] Iter %02d, loss_U %.5f, loss_V %.5f, loss_UV %.5f, loss_Adam %.5f, PSNR %.2f, mu %.4f, sum_loss %.4f" % (img_id, j, loss_U, loss_V, loss_UV, total_loss.item() / n1 / n2, rec_PSNR, mu, sum_loss))
            print("[%02d] Iter %02d, loss_Adam %.5f, PSNR %.2f, mu %.4f" % (img_id, j, total_loss.item() / n1 / n2, rec_PSNR, mu))

    rec_img = X.detach().cpu().numpy() * 255
    rec_PSNR = psnr3D(rec_img, img_gt * 255)
    return rec_img, rec_PSNR

torch.manual_seed(3)
rec_PSNR = np.zeros((num_img, 1))
rec_img = np.zeros((num_img, n1, n2, n3))
for i in range(num_img):
    data0 = torch.from_numpy(test_data[i, 0:n1*n2*n3]).type(torch.FloatTensor) * (1 / 255)
    data1 = torch.from_numpy(test_data[i, n1*n2*n3:n1*n2*n3*2]).type(torch.FloatTensor) * (1 / 255)
    data2 = torch.from_numpy(test_data[i, n1*n2*n3*2:n1*n2*n3*3]).type(torch.FloatTensor)
    img_gt = data0.view(n1, n2, n3).numpy()  # groundtruth
    img_input = data1.view(n1, n2, n3).to(device) # input image
    M = data2.view(n1, n2, n3).to(device)  # All entries are equal to 1.
    M_comp = torch.ones(n1, n2, n3).to(device) - M
    Y = torch.mul(img_input, M).to(device)
    rec_img[i, :, :, :], rec_PSNR[i] = RTFD(Y, i, img_gt, M, M_comp, mu)
    print("[%02d] PSNR achieved by RTFD is %.4f" % (i, rec_PSNR[i]))

sio.savemat('Result_SBI.mat', {'Result': rec_img})
print(rec_PSNR)
print(np.mean(rec_PSNR))