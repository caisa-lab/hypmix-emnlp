"""
 Learning from Between-class Examples for Deep Sound Recognition.
 Yuji Tokozume, Yoshitaka Ushiku, and Tatsuya Harada

"""

import sys
import os
import chainer
from os import listdir
from os.path import isfile, join
import opts
import models
import dataset_unsup
from train_unsup import Trainer

# os.environ["CUDA_VISIBLE_DEVICES"] = str(3)

def main():
    opt = opts.parse()
    chainer.cuda.get_device_from_id(opt.gpu).use()
    train(opt, 1)


def train(opt, split):
    model = getattr(models, opt.netType)(opt.nClasses)
    model.to_gpu()
    optimizer = chainer.optimizers.NesterovAG(lr=opt.LR, momentum=opt.momentum)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(opt.weightDecay))
    label_train_data, unlabel_train_data, val_iter = dataset_unsup.setup(opt, split)
    trainer = Trainer(model, optimizer, label_train_data, unlabel_train_data, val_iter, opt)
    if ('model_split_partial_10_{}_{}.npz'.format(split, opt.nPartial) in [f for f in listdir(opt.save) if isfile(join(opt.save, f))]):
        print("Pre-trained available for model_split_partial_10_{}_{}.npz".format(split, opt.nPartial))
        chainer.serializers.load_npz(
                os.path.join(opt.save, 'model_split_partial_10_{}_{}.npz'.format(split, opt.nPartial)), model)
    if opt.testOnly:
        chainer.serializers.load_npz(
            os.path.join(opt.save, 'model_split_partial_10_{}_{}.csv'.format(split, opt.nPartial)), trainer.model)
        val_top1 = trainer.val()
        print('| Val: top1 {:.2f}'.format(val_top1))
    best = 100
    for epoch in range(1, opt.nEpochs + 1):
        train_loss, train_top1 = trainer.train(epoch)
        val_top1 = trainer.val()
        if(val_top1<best):
            best = val_top1
        sys.stderr.write('\r\033[K')
        sys.stdout.write(
            '| Epoch: {}/{} | Train: LR {}  Loss {:.3f}  top1 {:.2f} | Val: top1 {:.2f} | best {:.2f}\n'.format(
                epoch, opt.nEpochs, trainer.optimizer.lr, train_loss, train_top1, val_top1,best))
        sys.stdout.flush()
        if epoch%50==0:
            chainer.serializers.save_npz(
            os.path.join(opt.save, 'model_split_partial_speechmix_10_{}_{}.npz'.format(split, opt.nPartial)), model)


if __name__ == '__main__':
    main()
