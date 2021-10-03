import sys
import numpy as np
import chainer
from chainer import cuda
import chainer.functions as F
import time
import cupy as cp

import utils
import numpy as np
import torch

def linear_rampup(current, rampup_length=1200):
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current / rampup_length, 0.0, 1.0)
        return float(current)

class Trainer:
    def __init__(self, model, optimizer, label_train_data, unlabel_train_data, val_iter, opt):
        self.model = model
        self.optimizer = optimizer
        self.label_train_data = label_train_data
        self.unlabel_train_data = unlabel_train_data
        self.label_train_iter = chainer.iterators.MultiprocessIterator(label_train_data, 16, repeat=False)
        self.unlabel_train_iter = chainer.iterators.MultiprocessIterator(unlabel_train_data, 16, repeat=False)
        self.num_iterations = 5
        self.val_iter = val_iter
        self.opt = opt
        # self.n_batches = (len(train_iter.dataset) - 1) // opt.batchSize + 1
        self.start_time = time.time()

    def train(self, epoch):
        self.optimizer.lr = self.lr_schedule(epoch)
        train_loss = 0
        train_acc = 0
        
        for i in range(self.num_iterations):
            try:
                inputs_x, targets_x = chainer.dataset.concat_examples(self.label_train_iter.next())
            except:
                self.label_train_iter = chainer.iterators.MultiprocessIterator(self.label_train_data, 16, repeat=False)
                inputs_x, targets_x = chainer.dataset.concat_examples(self.label_train_iter.next())
            
            try:
                (inputs_u, inputs_ori) = chainer.dataset.concat_examples(self.unlabel_train_iter.next())
            except:
                self.unlabel_train_iter = chainer.iterators.MultiprocessIterator(self.unlabel_train_data, 48, repeat=False)
                (inputs_u, inputs_ori) = chainer.dataset.concat_examples(self.unlabel_train_iter.next())
                
            batch_size = inputs_x.shape[0]
            batch_size_2 = inputs_ori.shape[0]
            inputs_x = inputs_x.astype(np.float32)
            targets_x = targets_x.astype(np.float32)
            inputs_u = inputs_u.astype(np.float32)
            inputs_ori = inputs_ori.astype(np.float32)
            with chainer.using_config('train',False):
                inputs_u_t = chainer.Variable(cuda.to_gpu(inputs_u[:, None, None, :]))
                inputs_ori_t = chainer.Variable(cuda.to_gpu(inputs_ori[:, None, None, :]))
                with chainer.no_backprop_mode():
                    outputs_u = F.softmax(self.model(inputs_u_t))
                    outputs_u = F.reshape(outputs_u, (outputs_u.shape[0] // 1, 1, outputs_u.shape[1]))
                    outputs_u = F.mean(outputs_u, axis=1)
                    outputs_ori = F.softmax(self.model(inputs_ori_t))
                    outputs_ori = F.reshape(outputs_ori, (outputs_ori.shape[0] // 1, 1, outputs_ori.shape[1]))
                    outputs_ori = F.mean(outputs_ori, axis=1)
                    outputs_u = utils.numpy_expmap0(outputs_u, c=cp.array([1.0]))
                    outputs_ori = utils.numpy_expmap0(outputs_ori, c=cp.array([1.0]))
                    sum_outputs = utils.numpy_mobius_add(outputs_u, outputs_ori, c=cp.array([1.0]))
                    p = utils.numpy_logmap0(sum_outputs, c=cp.array([1.0]))
                    pt = p**(1/0.5)
                    targets_u = pt / pt.data.sum(axis=1, keepdims=True)
                    targets_u = targets_u.data.get()
            # print ('Guessed labels shape: ', targets_u.shape)
            # print ('Guessed labels type: ', targets_u.dtype)
            all_targets = np.concatenate((targets_x, targets_u, targets_u, targets_u)).astype(np.float32)
            # print ('Concatenated targets...')
            all_inputs = np.concatenate((inputs_x, inputs_u, inputs_ori, inputs_ori)).astype(np.float32)

            # all_inputs = cp.concatenate((cp.asarray(inputs_x), cp.asarray(inputs_u), cp.asarray(inputs_ori), cp.asarray(inputs_ori)))
            # all_targets = cp.concatenate((cp.asarray(targets_x), cp.asarray(targets_u), cp.asarray(targets_u), cp.asarray(targets_u)))
            # print ('All inputs shape: ', all_inputs.shape)
            # print ('All targets shape: ', all_targets.shape)
            # x_array, t_array = chainer.dataset.concat_examples(batch)
            # x_array = np.reshape(x_array,(self.opt.batchSize*2,-1)).astype('float32')
            # t_array = np.reshape(t_array,(self.opt.batchSize*2,-1)).astype('float32')

            x = chainer.Variable(cuda.to_gpu(all_inputs[:, None, None, :]))
            t = chainer.Variable(cuda.to_gpu(all_targets))
            
            # print ('Input variable shape: ', x.data.shape)
            # print ('Input target shape: ', t.data.shape)
            idx1 = torch.randperm(all_inputs.shape[0] - batch_size_2)
            idx2 = torch.arange(batch_size_2) + \
                all_inputs.shape[0] - batch_size_2
            idx = torch.cat([idx1, idx2], dim=0)
            perm_mat = idx.tolist()

            # perm_mat = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

            # print (perm_mat)
            self.model.cleargrads()
            y , t = self.model(x, t, self.opt.mixup_type, self.opt.eligible, self.opt.batchSize, perm_mat)
            # print (y)
            # print (t)
            # print ('Y: ', y.shape)
            # print ('T: ', t.shape)
            # print ('Y: ', y[:batch_size].shape)

            loss_1 = utils.kl_divergence(y[:batch_size], t[:batch_size])
            # print ('Loss l: ', loss_1)

            Lu = utils.kl_divergence(y[batch_size:-batch_size_2], t[batch_size:-batch_size_2])
            
            # Lu2 = torch.mean(torch.clamp(torch.sum(-F.softmax(outputs_u, dim=1)
                                                  #  * F.log_softmax(outputs_u, dim=1), dim=1) - args.margin, min=0))

            # print ('Loss: ', loss_1)
            # print ('Lu: ', Lu)                              
            loss = loss_1 + linear_rampup(epoch)*Lu
            # loss = loss_1
            print ('Loss: ', loss)
            acc = F.accuracy(y, F.argmax(t, axis=1))
            
            loss.backward()
            self.optimizer.update()
            train_loss += float(loss.data) * len(t.data)
            train_acc += float(acc.data) * len(t.data)

            elapsed_time = time.time() - self.start_time
            progress = (self.num_iterations * (epoch - 1) + i + 1) * 1.0 / (self.num_iterations * self.opt.nEpochs)
            if ((progress)!=0):
                eta = elapsed_time / progress - elapsed_time
            else:
                eta = 0
            line = '* Epoch: {}/{} ({}/{}) | Train: LR {} | Time: {} (ETA: {})'.format(
                epoch, self.opt.nEpochs, i + 1, self.num_iterations,
                self.optimizer.lr, utils.to_hms(elapsed_time), utils.to_hms(eta))
            sys.stderr.write('\r\033[K' + line)
            sys.stderr.flush()

        # self.train_iter.reset()
        train_loss /= len(self.label_train_data)*2
        train_top1 = 100 * (1 - train_acc / (len(self.label_train_data)*2))

        return train_loss, train_top1

    def val(self):
        with chainer.using_config('train',False):
            val_acc = 0
            for batch in self.val_iter:
                x_array, t_array = chainer.dataset.concat_examples(batch)
                if self.opt.nCrops > 1:
                    x_array = x_array.reshape((x_array.shape[0] * self.opt.nCrops, x_array.shape[2]))
                x = chainer.Variable(cuda.to_gpu(x_array[:, None, None, :]))
                t = chainer.Variable(cuda.to_gpu(t_array))
                with chainer.no_backprop_mode():
                    y = F.softmax(self.model(x))
                    y = F.reshape(y, (y.shape[0] // self.opt.nCrops, self.opt.nCrops, y.shape[1]))
                    y = F.mean(y, axis=1)
                    acc = F.accuracy(y, t)
                    val_acc += float(acc.data) * len(t.data)

            self.val_iter.reset()
        val_top1 = 100 * (1 - val_acc / len(self.val_iter.dataset))

        return val_top1

    def lr_schedule(self, epoch):
        divide_epoch = np.array([self.opt.nEpochs * i for i in self.opt.schedule])
        decay = sum(epoch > divide_epoch)
        if epoch <= self.opt.warmup:
            decay = 1

        return self.opt.LR * np.power(0.1, decay)
