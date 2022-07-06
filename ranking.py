import argparse, os
import torch
import torchvision.models as models
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from torchvision import datasets
from LogME import LogME
import pprint

models_hub = ['mobilenet_v2', 'mnasnet1_0', 'densenet121', 'densenet169', 'densenet201',
               'resnet34', 'resnet50', 'resnet101', 'resnet152', 'googlenet', 'inception_v3']


def get_configs():
    parser = argparse.ArgumentParser(
        description='Ranking pre-trained models')
    parser.add_argument('--gpu', default=0, type=int,
                        help='GPU num for training')
    parser.add_argument('--batch_size', default=48, type=int)

    # dataset
    parser.add_argument('--dataset', default="aircraft",
                        type=str, help='Name of dataset')
    parser.add_argument('--data_path', default="/data/FGVCAircraft/train",
                        type=str, help='Path of dataset')
    parser.add_argument('--num_workers', default=2, type=int,
                        help='Num of workers used in dataloading')
    # model
    configs = parser.parse_args()

    return configs


def forward_pass(score_loader, model, fc_layer):
    """
    a forward pass on target dataset
    :params score_loader: the dataloader for scoring transferability
    :params model: the model for scoring transferability
    :params fc_layer: the fc layer of the model, for registering hooks
    returns
        features: extracted features of model
        outputs: outputs of model
        targets: ground-truth labels of dataset
    """
    features = []
    outputs = []
    targets = []
    
    def hook_fn_forward(module, input, output):
        features.append(input[0].detach().cpu())
        outputs.append(output.detach().cpu())
    
    forward_hook = fc_layer.register_forward_hook(hook_fn_forward)
    
    model.eval()
    with torch.no_grad():
        for _, (data, target) in enumerate(score_loader):
            targets.append(target)
            data = data.cuda()
            _ = model(data)
    
    forward_hook.remove()
    features = torch.cat([x for x in features])
    outputs = torch.cat([x for x in outputs])
    targets = torch.cat([x for x in targets])
    
    return features, outputs, targets


def main():
    configs = get_configs()
    torch.cuda.set_device(configs.gpu)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

    if not os.path.isdir(f'logme_{configs.dataset}'):
        os.mkdir(f'logme_{configs.dataset}')
    score_dict = {}
    for model in models_hub:
        configs.model = model
        if model == 'inception_v3': # inception_v3 is pretrained on 299x299 images
            transform=transforms.Compose([  
                transforms.Resize((299, 299)),
                transforms.ToTensor(),
                normalize
            ])
        else:
            transform=transforms.Compose([  # other models are pretrained on 224x224 images
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                normalize
            ])
        score_dataset = datasets.ImageFolder(configs.data_path, transform=transform)
        # or try your customized dataset
        score_loader = DataLoader(score_dataset, batch_size=configs.batch_size, shuffle=False,
            num_workers=configs.num_workers, pin_memory=True)
        score_dict[model] = score_model(configs, score_loader)
    results = sorted(score_dict.items(), key=lambda i: i[1], reverse=True)
    torch.save(score_dict, f'logme_{configs.dataset}/results.pth')
    print(f'Models ranking on {configs.dataset}: ')
    pprint.pprint(results)


def score_model(configs, score_loader):
    print(f'Calc Transferabilities of {configs.model} on {configs.dataset}')

    if configs.model == 'inception_v3':
            model = models.__dict__[configs.model](pretrained=True, aux_logits=False).cuda()
    else:
        model = models.__dict__[configs.model](pretrained=True).cuda()

    # different models has different linear projection names
    if configs.model in ['mobilenet_v2', 'mnasnet1_0']:
        fc_layer = model.classifier[-1]
    elif configs.model in ['densenet121', 'densenet169', 'densenet201']:
        fc_layer = model.classifier
    elif configs.model in ['resnet34', 'resnet50', 'resnet101', 'resnet152', 'googlenet', 'inception_v3']:
        fc_layer = model.fc
    else:
        # try your customized model
        raise NotImplementedError

    print('Conducting features extraction...')
    features, outputs, targets = forward_pass(score_loader, model, fc_layer)
    # predictions = F.softmax(outputs)

    print('Conducting transferability calculation...')
    logme = LogME(regression=False)
    score = logme.fit(features.numpy(), targets.numpy())

    # save calculated bayesian weight
    torch.save(logme.ms, f'logme_{configs.dataset}/weight_{configs.model}.pth')

    print(f'LogME of {configs.model}: {score}\n')
    return score
    

if __name__ == '__main__':
    main()
