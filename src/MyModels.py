'''
This script defines the CoaTNet classifier used in PostFocus
'''

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from coatnet import CoAtNet
class ImageClassificationBase(nn.Module):
    '''
    this class defines binary classfication model
    '''
    def training_step(self, batch):
        '''
        Training process:
        compute loss using results from model
        '''
        images, labels = batch
        images, labels = images.cuda(), labels.cuda()
        out = self(images)
        loss = F.cross_entropy(out, labels)
        acc = accuracy(out, labels)
        return loss, acc

    def validation_step(self, batch):
        '''
        Validation: same as training, but return matrices
        '''
        images, labels = batch
        images, labels = images.cuda(), labels.cuda()
        out = self(images)
        loss = F.cross_entropy(out, labels).cuda()
        acc = accuracy(out, labels)
        mat = matrices(out, labels)
        return {'val_loss': loss.detach(), 'val_acc': acc, 'mat': mat}

    def validation_epoch_end(self, outputs):
        '''
        Derive matrices using results from validation_step
        '''
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()
        batch_mat = [x['mat'] for x in outputs]
        epoch_mat = np.sum(np.array(batch_mat), axis=0)
        t_pos,f_pos,f_neg = epoch_mat[0], epoch_mat[2], epoch_mat[3]
        pre = t_pos/(t_pos+f_pos)
        rec = t_pos/(t_pos+f_neg)
        f1_s = 2*t_pos/(2*t_pos+f_pos+f_neg)
        return {'val_loss': epoch_loss.item(),\
        'val_acc': epoch_acc.item(),\
        'Precision':pre, 'Recall':rec, 'F1':f1_s}

    def epoch_end(self, epoch, result):
        '''
        print epoch summary
        '''
        lines = []
        lines.append(f'Epoch {epoch}')
        train_loss = result['train_loss']
        lines.append(f'train_loss: {train_loss}')
        train_acc = result['train_acc']
        lines.append(f'train_acc: {train_acc}')
        val_loss = result['val_loss']
        lines.append(f'train_loss: {val_loss}')
        val_acc = result['val_acc']
        lines.append(f'train_acc: {val_acc}')
        precision = result['Precision']
        lines.append(f'precision: {precision}')
        recall = result['Recall']
        lines.append(f'recall: {recall}')
        f1_score = result['F1']
        lines.append(f'F1: {f1_score}')
        print(', '.join(lines))

class CoatNet(ImageClassificationBase):
    '''
    CoatNet model for classification
    Use the backbone from Dai et. al
    With a classification head
    '''
    def __init__(self, args):
        super().__init__()
        model = CoAtNet((args.size, args.size),\
        3, [2, 2, 3, 5, 2], [64, 96, 192, 384, 768], num_classes=2)
        model.act = nn.Softmax()
        self.network = model

    def forward(self, input_x):
        '''
        Compute probability
        '''
        return self.network(input_x)

def accuracy(outputs, labels):
    '''
    helper function for model validation
    '''
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

def matrices(outputs, labels):
    '''
    helper function to get performance indicators
    '''
    _, preds = torch.max(outputs, dim=1)

    t_ans = (preds == labels).cpu().numpy()
    f_ans = (preds != labels).cpu().numpy()
    pos = (labels==0).cpu().numpy()
    neg = (labels==1).cpu().numpy()
    t_pos = sum(np.logical_and(t_ans,pos))
    t_neg = sum(np.logical_and(t_ans,neg))
    f_pos = sum(np.logical_and(f_ans,neg))
    f_neg = sum(np.logical_and(f_ans,pos))

    return np.array([t_pos,t_neg,f_pos,f_neg])
