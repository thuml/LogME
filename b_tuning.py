import argparse
import os
from time import time

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm import tqdm
import torchvision.transforms as transforms
from utils.transforms import get_transforms
from utils.tools import AccuracyMeter, TenCropsTest


models_list = ['mobilenet_v2', 'mnasnet1_0', 'densenet121', 'densenet169', 'densenet201',
               'resnet34', 'resnet50', 'resnet101', 'resnet152', 'googlenet', 'inception_v3']


def get_configs():
    parser = argparse.ArgumentParser(
        description='Bayesian tuning')

    # train
    parser.add_argument('--gpu', default=0, type=int,
                        help='GPU id for training')
    parser.add_argument('--seed', type=int, default=2021)

    parser.add_argument('--batch_size', default=48, type=int)
    parser.add_argument('--total_iter', default=9050, type=int)
    parser.add_argument('--eval_iter', default=1000, type=int)
    parser.add_argument('--save_iter', default=9000, type=int)
    parser.add_argument('--print_iter', default=100, type=int)

    # dataset
    parser.add_argument('--dataset', default="aircraft",
                        type=str, help='Name of dataset')
    parser.add_argument('--data_path', default="./data/FGVCAircraft",
                        type=str, help='Path of dataset')
    parser.add_argument('--num_workers', default=2, type=int,
                        help='Num of workers used in dataloading')

    # model
    parser.add_argument('--model', default="resnet50", choices=models_list,
                        type=str, help='Name of NN')
    parser.add_argument('--teachers', nargs='+', help='Names of teahcer models')
    parser.add_argument('--class_num', default="196",
                        type=int, help='class number')

    # optimizer
    parser.add_argument('--lr', default=1e-3, type=float,
                        help='Learning rate for training')
    parser.add_argument('--gamma', default=0.1, type=float,
                        help='Gamma value for learning rate decay')
    parser.add_argument('--nesterov', default=True,
                        type=bool, help='nesterov momentum')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='Momentum value for optimizer')
    parser.add_argument('--weight_decay', default=5e-4,
                        type=float, help='Weight decay value for optimizer')

    # experiment
    parser.add_argument('--root', default='.', type=str,
                        help='Root of the experiment')
    parser.add_argument('--name', default='b-tuning', type=str,
                        help='Name of the experiment')
    parser.add_argument('--save_dir', default="model",
                        type=str, help='Path of saved models')
    parser.add_argument('--visual_dir', default="visual",
                        type=str, help='Path of tensorboard data for training')
    parser.add_argument('--temperature', default=0.1, type=float,
                        metavar='P', help='temperature of logme weight')
    parser.add_argument('--tradeoff', default=100,
                        type=float, help='b-tuning tradeoff')
    configs = parser.parse_args()

    return configs


def str2list(v):
    return v.split(',')


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


def get_writer(log_dir):
    return SummaryWriter(log_dir)


def get_data_loader(configs):
    # data augmentation
    data_transforms = get_transforms(resize_size=256, crop_size=224)

    # build dataset
    if configs.dataset == 'aircraft':
        train_dataset = datasets.ImageFolder(
            os.path.join(configs.data_path, 'train'),
            transform=data_transforms['train'])
        val_dataset = datasets.ImageFolder(
            os.path.join(configs.data_path, 'test'),
            transform=data_transforms['train'])
        test_datasets = {
            'test' + str(i):
                datasets.ImageFolder(
                    os.path.join(configs.data_path, 'test'),
                    transform=data_transforms["test" + str(i)]
            )
            for i in range(10)
        }
    else:
        # try your customized dataset
        raise NotImplementedError

    # build dataloader
    train_loader = DataLoader(train_dataset, batch_size=configs.batch_size, shuffle=True,
                              num_workers=configs.num_workers, pin_memory=True)

    val_loader = DataLoader(val_dataset, batch_size=configs.batch_size, shuffle=False,
                            num_workers=configs.num_workers, pin_memory=True)
    test_loaders = {
        'test' + str(i):
            DataLoader(
                test_datasets["test" + str(i)],
                batch_size=4, shuffle=False, num_workers=configs.num_workers
        )
        for i in range(10)
    }

    return train_loader, val_loader, test_loaders


def set_seeds(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def train(configs, train_loader, val_loader, test_loaders, net, teachers):
    train_len = len(train_loader) - 1
    train_iter = iter(train_loader)
    weight = torch.from_numpy(torch.load(f'logme_{configs.dataset}/weight_{configs.model}.pth')).float().cuda()

    logmes = torch.load(f'logme_{configs.dataset}/results.pth')
    pi = []
    for teacher in teachers:
        pi.append(logmes[teacher['name']])
    pi = torch.softmax(torch.tensor(pi) / configs.temperature, dim=0).float().cuda()

    # different learning rates for different layers
    params_list = [{"params": filter(lambda p: p.requires_grad, net.f_net.parameters())},
                   {"params": filter(lambda p: p.requires_grad, net.c_net.parameters()), "lr": configs.lr * 10}]

    # optimizer and scheduler
    optimizer = torch.optim.SGD(params_list, lr=configs.lr, weight_decay=configs.weight_decay,
                                momentum=configs.momentum, nesterov=configs.nesterov)
    milestones = [6000]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones, gamma=configs.gamma)

    # check visual path
    visual_path = os.path.join(configs.visual_dir, configs.name)
    if not os.path.exists(visual_path):
        os.makedirs(visual_path)
    writer = get_writer(visual_path)

    # check model save path
    save_path = os.path.join(configs.save_dir, configs.name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    criterion_cls = nn.CrossEntropyLoss()
    criterion_dis = nn.MSELoss()

    # start training
    for iter_num in range(configs.total_iter):
        net.train()
        
        if iter_num % train_len == 0:
            train_iter = iter(train_loader)

        # Data Stage
        data_start = time()

        train_inputs, train_labels = next(train_iter)
        train_inputs, train_labels = train_inputs.cuda(), train_labels.cuda()

        data_duration = time() - data_start

        # Calc Stage
        calc_start = time()

        train_features, train_outputs = net(train_inputs, with_feat=True)

        b_tuning_loss = 0.0
        
        target_features = torch.matmul(train_features, weight.t())
            
        source_features = torch.zeros_like(target_features)
        for i, teacher in enumerate(teachers):
            with torch.no_grad():
                input = train_inputs
                source_features += pi[i] * torch.matmul(teacher['model'](input), teacher['weight'].t())

        classifier_loss = criterion_cls(train_outputs, train_labels)
        b_tuning_loss = criterion_dis(target_features, source_features.detach())
        loss = classifier_loss + configs.tradeoff * b_tuning_loss
        
        writer.add_scalar('loss/b_tuning_loss', configs.tradeoff * b_tuning_loss, iter_num)
        writer.add_scalar('loss/classifier_loss', classifier_loss, iter_num)
        writer.add_scalar('loss/total_loss', loss, iter_num)

        net.zero_grad()
        optimizer.zero_grad()

        loss.backward()
        optimizer.step()
        scheduler.step()

        calc_duration = time() - calc_start

        if iter_num % configs.eval_iter == 0:
            acc_meter = AccuracyMeter(topk=(1,))
            with torch.no_grad():
                net.eval()
                for val_inputs, val_labels in tqdm(val_loader):
                    val_inputs, val_labels = val_inputs.cuda(), val_labels.cuda()
                    val_outputs = net(val_inputs)
                    acc_meter.update(val_outputs, val_labels)
                writer.add_scalar('acc/val_acc', acc_meter.avg[1], iter_num)
                print(
                    "Iter: {}/{} Val_Acc: {:2f}".format(
                        iter_num, configs.total_iter, acc_meter.avg[1])
                )
            acc_meter.reset()

        if iter_num % configs.save_iter == 0 and iter_num > 0:
            test_acc = TenCropsTest(test_loaders, net)
            writer.add_scalar('acc/test_acc', test_acc, iter_num)
            print(
                "Iter: {}/{} Test_Acc: {:2f}".format(
                    iter_num, configs.total_iter, test_acc)
            )
            checkpoint = {
                'state_dict': net.state_dict(),
                'iter': iter_num,
                'acc': test_acc,
            }
            torch.save(checkpoint,
                       os.path.join(save_path, '{}.pkl'.format(iter_num)))
            print("Model Saved.")

        if iter_num % configs.print_iter == 0:
            print(
                "Iter: {}/{} Loss: {:2f} Loss_CLS: {:2f} Loss_KD: {:2f}, d/c: {}/{}".format(iter_num, configs.total_iter,
                    loss, classifier_loss, configs.tradeoff * b_tuning_loss, data_duration, calc_duration))


def load_model(configs, pretrained=True, only_feature=False):
    model = models.__dict__[configs.model](pretrained=pretrained)
    if configs.model in ['mobilenet_v2', 'mnasnet1_0']:
        model.feature_dim = model.classifier[-1].in_features
        model.classifier[-1] = nn.Identity()
    elif configs.model in ['densenet121', 'densenet169', 'densenet201']:
        model.feature_dim = model.classifier.in_features
        model.classifier = nn.Identity()
    elif configs.model in [ 'resnet34', 'resnet50', 'resnet101', 
                            'resnet152', 'googlenet', 'inception_v3']:
        model.feature_dim = model.fc.in_features
        model.fc = nn.Identity()

    if only_feature:
        return model

    class Net(nn.Module):
        def __init__(self, model, feature_dim):
            super(Net, self).__init__()
            self.f_net = model
            self.feature_dim = feature_dim
            self.c_net = nn.Linear(feature_dim, configs.class_num)
            self.c_net.weight.data.normal_(0, 0.01)
            self.c_net.bias.data.fill_(0.0)

        def forward(self, x, with_feat=False):
            feature = self.f_net(x)
            out = self.c_net(feature)
            if with_feat:
                return feature, out
            else:
                return out

    return Net(model, model.feature_dim)


def main():
    configs = get_configs()
    print(configs)
    torch.cuda.set_device(configs.gpu)
    set_seeds(configs.seed)

    train_loader, val_loader, test_loaders = get_data_loader(configs)

    net = load_model(configs).cuda()

    student = configs.model
    teachers = []
    for teacher_name in configs.teachers:
        assert teacher_name in models_list
        configs.model = teacher_name
        model_t_feat = load_model(configs, only_feature=True).cuda().eval()      
        model_t = {'name':teacher_name,
                   'model': model_t_feat,
                   'weight': torch.from_numpy(torch.load(f'logme_{configs.dataset}/weight_{teacher_name}.pth')).float().cuda()
                  }
        teachers.append(model_t)

    configs.model = student
    train(configs, train_loader, val_loader, test_loaders, net, teachers)


if __name__ == '__main__':
    print("PyTorch {}".format(torch.__version__))
    print("TorchVision {}".format(torchvision.__version__))
    main()
