# +
import random
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

import util
from mainCNF import get_datasets, save_args
from cnf_args import DATASETS

# Learning to Adapt to Evolving Domains
# https://proceedings.neurips.cc/paper/2020/hash/fd69dbe29f156a7ef876a40a94f65599-Abstract.html
# https://github.com/Liuhong99/EAML/tree/master
import EAML.meta_test as T
from EAML.model import Learner, weight_init
from EAML.utils import DAN, get_mini_batches
from EAML.lip_reg import compute_margins, get_grad_hl_norms


# +
def get_parser():    
    parser = argparse.ArgumentParser(description='Evolution Adaptive Meta-Learning')
    parser.add_argument("--dataset", type=str, default="moon", choices=DATASETS)
    parser.add_argument("--n_dim", type=int, default=None, help="the number of dimension of dataset")
    parser.add_argument("--file_name", type=str, required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--no_inter", type=eval, default=False)
    parser.add_argument("--source_only", type=eval, default=False)
    parser.add_argument("--label_ratio", type=str, default="100-0-0", help="ratio of labeled samples of each domain")
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument("--hidden_dim", type=float, default=64)
    
    # args from original implementation
    parser.add_argument("--batch_size", default=100, type=int)
    parser.add_argument('--lr-in',  default=3e-2, type=float, help='initial learning rate in the inner loop')
    parser.add_argument('--lr-out', default=3e-3, type=float, help='initial learning rate in the outer loop')
    parser.add_argument('--balance_in', default=0.5, type=float, help='the trade-off hyper-parameter for transfer loss in the inner loop')
    parser.add_argument('--balance_out', default=1.0, type=float, help='the trade-off hyper-parameter for transfer loss in the outer loop')
    parser.add_argument('--inner_iter', default=3, type=int, help='iteration of each inner loop')
    parser.add_argument('--lip_balance', default = 0.2, type=float, help='balance of regularization')
    parser.add_argument('--lip_jth', default = 0.01, type=float, help='thresh of regularization')
    
    return parser


def meta_training(args, x_all, y_all):
    domain_length = len(x_all)
    
    feature_extractor = Learner(config, None, None)
    classifier = Learner(config1, None, None)
    feature_extractor, classifier = util.torch_to(feature_extractor, classifier)
    
    optimizer = torch.optim.SGD(feature_extractor.parameters(), lr=args.lr_out, weight_decay=0.0005 ,momentum=0.9)
    criterion = torch.nn.CrossEntropyLoss()
    
    dataset = util.preprocess_input(x_all[0], y_all[0])
    train_loader = DataLoader(dataset, batch_size=args.batch_size * (domain_length + 1), shuffle=True, drop_last=True)
    assert train_loader.batch_size < x_all[0].shape[0]

    tt = get_mini_batches(x_all, y_all, s_size=args.batch_size, q_size=args.batch_size, len_epoch=args.epochs)

    epoch = 0
    step = 0
    lr_r = args.lr_out
    loss_history = []
    while epoch < args.epochs:
        if epoch > int(args.epochs / 2) & epoch <= int(args.epochs / 4 * 3):
            lr_r = 0.1 * args.lr_out
        if epoch > int(args.epochs / 4 * 3):
            lr_r = 0.01 * args.lr_out

        for g in optimizer.param_groups:
            g['lr'] = lr_r
            
        for (i, ((xspt, yspt, xqry, yqry),(im_s0,lb_s0))) in enumerate(zip(tt,train_loader)):
            ### original settings on Rot MNIST ###
            # batch_size=64, epochs=500, num_domain=10
            # im_s0 -> (704, 1, 28, 28); 704 = batch_size * (num_domain + 1)
            # lb_s0 -> (704)
            # len(tt) = 500
            # xspt -> (10, 64, 1, 28, 28)
            # yspt -> (10, 64)
            # xqry -> (10, 64, 1, 28, 28)
            # yqry -> (10, 64)
            
            if im_s0.shape[0] != args.batch_size * (domain_length + 1):
                continue
            losslst = []
            for xx in classifier.children():
                weight_init(xx)
            fast_weights1 = classifier.parameters()
            im_s1 = im_s0[:domain_length * args.batch_size]
            lb_s1 = lb_s0[:domain_length * args.batch_size]
            for domain_ind in range(domain_length):
                randind = np.random.permutation(domain_length * args.batch_size)[0:args.batch_size]
                im_s = util.torch_to(im_s1[randind])
                lb_s = util.torch_to(lb_s1[randind])
                im_source1 = util.torch_to(torch.from_numpy(xspt[domain_ind].astype('float32')))
                
                for i in range(args.inner_iter):
                    logits = classifier(feature_extractor(im_s), fast_weights1, bn_training=True)
                    logitt = classifier(feature_extractor(im_source1), fast_weights1, bn_training=True)
                    ce1 = F.cross_entropy(logits, lb_s) + args.balance_in * DAN(logits,logitt)
                    grad1 = torch.autograd.grad(ce1, fast_weights1)
                    fast_weights1 = list(map(lambda p: p[1] - args.lr_in * p[0], zip(grad1, fast_weights1)))
                
            im_s = im_s0[domain_length * args.batch_size:domain_length * args.batch_size + args.batch_size]
            im_s = util.torch_to(im_s)
            lb_s = lb_s0[domain_length * args.batch_size:domain_length * args.batch_size + args.batch_size]
            lb_s = util.torch_to(lb_s)
            fss = feature_extractor(im_s)
            logits = classifier(fss, fast_weights1, bn_training=True)

            reg_loss = 0
            
            if args.lip_balance != 0:
                margins = compute_margins(logits, lb_s)
                norm_sq_dict = get_grad_hl_norms({'ft':fss}, torch.mean(margins), classifier, create_graph=True, only_inputs=True)
                
                for val in norm_sq_dict.values():
                    j = val[1]
                    j_ind = j > args.lip_jth
                    if torch.sum(j_ind) > 0:
                        reg_loss += torch.mean(j[j_ind])

            for domain_ind in range(domain_length):
                im_source1q = torch.from_numpy(xqry[domain_ind].astype('float32'))
                im_source1q = util.torch_to(im_source1q)
                label_source1q = torch.from_numpy(yqry[domain_ind].astype('int64'))
                label_source1q = util.torch_to(label_source1q)
                logitt = classifier(feature_extractor(im_source1q), fast_weights1, bn_training=True)
                loss1 = DAN(logits,logitt)
                losslst.append(loss1)
        
            dloss = sum(losslst) / domain_length
            ce = F.cross_entropy(logits, lb_s)
            
            if args.lip_balance == 0:
                loss = args.balance_out * dloss + ce
            else:
                loss = args.balance_out * dloss + ce + args.lip_balance * reg_loss
            
            step += 1
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            loss_history.append(loss.item())
            
        epoch += 1
        print("\r"+'epoch:'+str(epoch), end="")
        
    return feature_extractor, classifier, loss_history


def meta_testing(args, x_all, y_all, feature_extractor_param):
    # build model    
    feature_extractor = T.Learner(config, None, None)
    classifier = T.Learner_cls(config1, None, None)
    feature_extractor.load_state_dict(feature_extractor_param)
    feature_extractor = util.torch_to(feature_extractor)
    classifier = util.torch_to(classifier)
    
    # prepare
    optimizer1 = torch.optim.SGD(classifier.parameters(), lr=0.01, weight_decay=0.0005, momentum=0.9)
    criterion = torch.nn.CrossEntropyLoss()
    tt = T.get_mini_batches(x_all, y_all, s_size=args.batch_size)
    dataset = util.preprocess_input(x_all[0], y_all[0])
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    
    # meta test
    for qq in range(len(x_all)):
        epoch=0
        step = 0
        while epoch < 2:
            xspt, yspt, xqry, yqry = tt[0]
            for i, (im_source,label_source) in enumerate(train_loader):

                im_target = torch.from_numpy(xspt[qq].astype('float32'))
                im_target = util.torch_to(im_target)
                im_source = util.torch_to(im_source)
                label_source = util.torch_to(label_source)

                feature_source = feature_extractor.forward(im_source)
                feature_target = feature_extractor.forward(im_target)

                fss, pred_source = classifier(feature_source)
                ftt, pred_target = classifier(feature_target)
                ce = criterion(pred_source, label_source)

                batch_size = im_source.shape[0]

                kernels = T.guassian_kernel(pred_source, pred_target)

                MMD = T.DAN(64,64,kernels)

                loss = ce + 0.2 * MMD
                step += 1
                optimizer1.zero_grad()
                loss.backward()
                optimizer1.step()
            epoch += 1
            
    return feature_extractor, classifier
    
    
def evaluation(args, feature_extractor, classifier, x_eval, y_eval):
    dataset = util.preprocess_input(x_eval, y_eval)
    x, y = dataset.tensors
    x, y = util.torch_to(x, y)

    feature_extractor.eval()
    classifier.eval()
    with torch.no_grad():
        _, pred = classifier(feature_extractor(x))
        pred = F.softmax(pred, dim=1) 
        pred = np.array(torch.Tensor.cpu(pred.argmax(dim=1)))
    accuracy = accuracy_score(y_eval, pred)
    return accuracy


def load_debug_dataset(flag:str, sampling=None):
    from torchvision.datasets import MNIST
    path = '/home/e12813/ISM/other_works/gradual_domain_adaptation/EAML/rot_mnist_28/'
    data_dir = '/home/e12813/ISM/data/'
    
    if flag == 'train':
        # data for meta-train
        x_all_train = []
        for i in range(0, 60, 3):
            x = np.load(path + 'mnist_train_img_' + str(i) + '.npy')
            x_all_train.append(x)

        dataset = MNIST(data_dir, train=True, download=False)
        y = dataset.targets.numpy()
        y_all_train = [y for i in range(len(x_all_train))]
        
        if sampling is not None:
            idx = np.random.choice(np.arange(y.size), size=sampling, replace=False)
            x_all_train = [x[idx].copy() for x in x_all_train]
            y_all_train = [y[idx].copy() for y in y_all_train]
        
        return x_all_train, y_all_train

    elif flag == 'test':
        # data for meta-test
        x_source = np.load(path + 'mnist_train_img_' + str(0) + '.npy')
        x_all_test = [x_source] 
        for i in range(120, 180, 6):
            x = np.load(path + 'mnist_train_img_' + str(i) + '.npy')
            x_all_test.append(x)

        dataset = MNIST(data_dir, train=True, download=False)
        y = dataset.targets.numpy()
        y_all_test = [y for i in range(len(x_all_test))]
        return x_all_test, y_all_test
    
    elif flag == 'eval':
        # data for eval
        x_all_eval = []
        for i in range(120, 180, 6):
            x = np.load(path + 'mnist_test_img_' + str(i) + '.npy')
            x_all_eval.append(x)

        dataset = MNIST(data_dir, train=False, download=False)
        y = dataset.targets.numpy()
        y_all_eval = [y for i in range(len(x_all_eval))]
        return x_all_eval, y_all_eval


# -

if __name__ == '__main__':
    # set args
    parser = get_parser()
    if 'get_ipython' in globals():
        # jupyter notebook env, for debug
        args = parser.parse_args(["--file_name", "debug"])
    else:
        args = parser.parse_args()

    # load data
    x_all, y_all, x_eval, y_eval = get_datasets(args, return_eval=True)
    x_source, y_source = x_all[0].copy(), y_all[0].copy()
    x_target, y_target = x_all.pop(), y_all.pop()
    
    # model settings
    config = [('linear', [args.hidden_dim, args.n_dim]),
              ('relu', [True]),
              ('linear', [args.hidden_dim, args.hidden_dim]),
              ('relu', [True])]

    config1 = [('linear', [args.hidden_dim, args.hidden_dim]),
               ('relu', [True]),
               ('linear', [args.n_class, args.hidden_dim]),]
        
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # meta train, Source and Intermedaite    
    feature_extractor, classifier, loss_history = meta_training(args, x_all, y_all)
    
    # meta test, Source and Target
    # Note: GDA does not assume the evolution of the target distribution.
    x_all, y_all = [x_source, x_target], [y_source, y_target]
    feature_extractor, classifier = meta_testing(args, x_all, y_all, feature_extractor.state_dict())
    
    # eval 
    accuracy = evaluation(args, feature_extractor, classifier, x_eval, y_eval)
    print()
    print(accuracy)

    # save args and accuracy score
    args.accuracy_score = accuracy
    file_name = save_args(args)