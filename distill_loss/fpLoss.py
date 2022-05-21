import torch
import torch.nn as nn
import torch.nn.functional as F
eps = 1e-10

class fpLoss(nn.Module):
    def __init__(self, ):
        super(fpLoss, self).__init__()

    def cross_entropy(self, logits, onehot_labels, ls=False):
        if ls:
            onehot_labels = 0.9 * onehot_labels + 0.1 / logits.size(-1)
            onehot_labels = onehot_labels.double()
        return (-1.0 * torch.mean(torch.sum(onehot_labels * F.log_softmax(logits, -1), -1), 0))


    def neg_entropy(self, logits):
        probs = F.softmax(logits, -1)
        return torch.mean(torch.sum(probs * F.log_softmax(logits, -1), -1), 0)

    def forward(self, targets, outputs, bi_outputs,):
        # Loss_cls
        difference = F.softmax(outputs, -1) - F.softmax(bi_outputs, -1)    # 与FRSKD比较时，使用了detach()
        onehot_labels = F.one_hot(targets, outputs.size(-1))
        loss_cls = self.cross_entropy(outputs + difference, onehot_labels, True)
        # tiny-imagenet 上， alpha=1, beta=1, ls=True, best test acc: 0.5870
        # tiny-imagenet 上， alpha=1, beta=1, ls=False, best test acc: 0.5840
        # 所以ls不是主要原因

        # R_attention

        # multi_warm_lb = bi_outputs.detach() > 0.0
        '''因为推导发现，使用multi-warm label的交叉熵梯度为 hat{y(x)} - m(x),
        其中hat{y(x)}是clonalnet预测的概率分布，m(x)表示的是multi-warm label，其中非零值为1/len(m(x)!=0)
        对比正常交叉熵的损失值是 hat{y(x)} - y(x) 其中y(x)为one-hot label,
        所以正确位置的梯度为负值，不正确位置的梯度为正值，也就实现了正确位置预测变大，不正确位置预测变小,也就使得预测的概率更加接近于one-hot label
        但是发现，
        使用multi-warm label的交叉熵会使得 hat{y(x)} 大于 m(x) 的梯度为正，小于 m(x) 的梯度为负值，这意味着，
        预测的概率会趋向于m(x)的分布，所以，应该使得m(x)中的非零值尽量少一些，这样只关注几个很混淆的类就可以了，这可以使非零值更大一些
        如果m(x)的非零值太小，就损害了自信的预测了
        所以将multi-warm label做了调整
        '''
        # multi_warm_lb = bi_outputs > 0.0
        multi_warm_lb = F.softmax(bi_outputs/2, -1) > 1.0/bi_outputs.size(-1)
        multi_warm_lb = torch.clamp(multi_warm_lb.double() + onehot_labels, 0, 1)
        multi_warm_lb = multi_warm_lb/torch.sum(multi_warm_lb, -1, True)
        R_attention = self.cross_entropy(outputs, multi_warm_lb.detach(), False)# 与FRSKD比较时，使用了detach()

        # R_entropy
        R_negtropy = self.neg_entropy(outputs)

        fp_loss = loss_cls + R_attention + R_negtropy

        # test for CE + neg_entropy
        # loss_cls = self.cross_entropy(outputs, onehot_labels)
        # fp_loss = loss_cls + R_negtropy # 已经试验证明 CE + negtive_entropy的CUB200精度（59.10%）低于loss_cls + negtive_entropy的精度(60.72%)
        return fp_loss