import timm 
import time 
import torch
import argparse 
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from functorch.experimental import replace_all_batch_norm_modules_

from build_model.util import get_imagenet
from util import * 

def validate(val_loader, named_network, criterion, print_freq, device):
    net_name, net = named_network
    
    logger = Logger()
    result = {}
    result['networks'] = [net_name]
    result['dataset'] = ['imagenet']

    with torch.no_grad():
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        # switch to evaluate mode
        net.eval()

        end = time.time()
        for i, (imgs, labels) in enumerate(val_loader):
            imgs, labels = imgs.to(device), labels.to(device)

            # compute logits
            logits = net(imgs)
            loss = criterion(logits, labels)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(logits.detach(), labels.detach(), topk=(1, 5))
            losses.update(loss.detach().item(), imgs.size(0))
            top1.update(prec1.item(), imgs.size(0))
            top5.update(prec5.item(), imgs.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       i, len(val_loader), batch_time=batch_time, loss=losses,
                       top1=top1, top5=top5))

        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))
               
        result['loss'] = [losses.avg]
        result['acc_top1'] = [top1.avg]
        result['acc_top5'] = [top5.avg]

        logger._save_to_log(result, 'networks', 'results/val_results_on_imagenet.csv')

        return top1.avg, top5.avg


def main():

    parser = argparse.ArgumentParser(description='Training vits and covits')

    parser.add_argument('--net_name', type=str, default='resnet50', 
                        help='the name of the neural network')
    parser.add_argument('--print_freq', type=int, default=10, 
                        help='the frequence to evaluate the network')
    parser.add_argument('--batch_size', type=int, default=256, 
                        help='the batch size for training and validating')

    args = parser.parse_args()

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # network configuration 
    net = timm.create_model(args.net_name, pretrained=True)
    net_cft = net.default_cfg
    net.to(DEVICE)
    net = nn.Sequential(InputResize(alpha=net_cft['mean'], beta=net_cft['std'], device=DEVICE), net)
    
    # loss fucntion 
    criterion = nn.CrossEntropyLoss().to(DEVICE)
    
    # dataset configuration
    img_test = get_imagenet(net_cft, train=False)
    test_loader = DataLoader(img_test,
                            batch_size=args.batch_size,
                            num_workers=2,
                            pin_memory=True,
                            shuffle=False) 

    validate(test_loader, (args.net_name, net), criterion, args.print_freq, DEVICE)
 
if __name__ == '__main__':
    main()



