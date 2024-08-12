import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from MyTorchUtils import Train,Test,set_same_seed
from ResNet import *
import argparse
import datetime
import torch
import torch.nn as nn
from torchvision import transforms,datasets
from torch.utils.data import Dataset,DataLoader,Subset
def Paramter_Generation():
    parameters = argparse.ArgumentParser(description='Training Paramters!')
    parameters.add_argument('--now',type=str,default=datetime.datetime.now().strftime('%F_%H-%M-%S'))
    parameters.add_argument('--Run',type=str,default=f'outdata/Run{parameters.parse_args().now}')
    if not os.path.exists(parameters.parse_args().Run):
        os.mkdir(parameters.parse_args().Run) 
    parameters.add_argument('--inputsize',type=tuple,default=(1,224,224))
    parameters.add_argument('--epochs',type=int,default=1)
    parameters.add_argument('--batch-size',type = int,default=8)
    parameters.add_argument('--num-workers',type=int,default=0)
    parameters.add_argument('--device',type=str,default='cuda' if torch.cuda.is_available() else 'cpu')
    parameters.add_argument('--lr',type=float,default=1e-5)
    parameters.add_argument('--betas',type=tuple,default=(0.9,0.999))
    parameters.add_argument('--weight-decay',type=float,default=0)
    parameters.add_argument('--writer-path',type=str,default=
                            parameters.parse_args().Run)
    parameters.add_argument('--modelframeworker-path',type = str,default=
                            os.path.join(parameters.parse_args().Run,f'model.txt'))
    parameters.add_argument('--bestmodel-path',type = str,default=
                            os.path.join(parameters.parse_args().Run,f'best.pth'))
    parameters.add_argument('--lastmodel-path',type=str,default=
                            os.path.join(parameters.parse_args().Run,f'last.pth'))
    parameters.add_argument('--result-path',type=str,default=
                            os.path.join(parameters.parse_args().Run,f'result.csv'))
    parameters.add_argument('--resultfig-path',type=str,default=
                            os.path.join(parameters.parse_args().Run,f'result.png'))
    parameters.add_argument('--testresult_path',type=str,default=
                            os.path.join(parameters.parse_args().Run,f'Test.csv'))
    parameters.add_argument('--shuffle',type =bool,default=True)
    parameters.add_argument('--loss-fn',default=nn.CrossEntropyLoss())
    parameters.add_argument('--seed',type=int,default=520)
    parameters.add_argument('--lradjust',type=list,default = [1e-6,1e-5,16,'down'])

    args = parameters.parse_args()
    with open(os.path.join(args.Run,'args.txt'),'w') as f:
        for arg, value in vars(args).items():
            f.write(f"{arg}: {value}\n")

    return args


def train(args):
    
    set_same_seed(seed=args.seed)
    model = ResNet34(num_classes=10,include_to_kan=True,toVit=True).to(args.device,non_blocking=True)
    optimizer  = torch.optim.Adam(model.parameters(),lr=args.lr,betas=args.betas,weight_decay=args.weight_decay)
    
    
    transforms_ =transforms.Compose([
            transforms.Resize([224,224]),
            transforms.RandomHorizontalFlip(p = 0.5),
            transforms.RandomVerticalFlip(p = 0.5),
            transforms.RandomRotation(20),
            transforms.ToTensor(),
            transforms.Normalize([0.5,], [0.5,])
        ])
    transforms__ = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5,], [0.5,])
        ])
    traindataall = datasets.MNIST('data', train=True,download=True, transform=transforms_)
    testdataall = datasets.MNIST('data', train=False,download=True, transform=transforms__)
    traindata = Subset(traindataall,list(range(0,600)))
    valdata = Subset(testdataall,list(range(0, 100)))
    testdata = Subset(testdataall,list(range(100,200)))
    trainloader = DataLoader(dataset=traindata, batch_size=args.batch_size,
                    num_workers = args.num_workers,pin_memory=True,shuffle=True)
    valloader = DataLoader(dataset=valdata, batch_size=args.batch_size,
                    num_workers = args.num_workers,pin_memory=True,shuffle=True)
    testloader = DataLoader(dataset=testdata, batch_size=args.batch_size,
                    num_workers = args.num_workers,pin_memory=True,shuffle=True)
    print('train data num:',len(traindata),'val data num:',len(valdata),'test data num:',len(testdata))
    Train(
        model=model,
        inputsize=(1,224,224),
        epochs=args.epochs,
        train_data_loader=trainloader,
        val_data_loader=valloader,
        optimizer=optimizer,
        loss_function=args.loss_fn,
        modelframeworker_path=args.modelframeworker_path,
        result_path = args.result_path,
        lastmodel_path = args.lastmodel_path,
        bestmodel_path = args.bestmodel_path,
        resultfig_path  = args.resultfig_path,
        writer_path=args.writer_path,
        lradjust = args.lradjust,
        device = args.device,
    )
    Test(
        model = model,
        modelpath=args.bestmodel_path,
        test_data_loader=testloader,
        outdata_path=args.testresult_path,
        device=args.device
    )
if __name__ == '__main__':
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = "0"
    args = Paramter_Generation()
    train(args=args)
