"""
 Implementation of EnvNet-v2 (ours)
 opt.fs = 44100
 opt.inputLength = 66650

"""
import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
from models.convbnrelu import ConvBNReLU
import utils as U
import random
from chainer import cuda
import cupy
# class EnvNetv2(chainer.Chain):
#     def __init__(self, n_classes):
#         super(EnvNetv2, self).__init__(
#             conv1=ConvBNReLU(1, 32, (1, 64), stride=(1, 2)),
#             conv2=ConvBNReLU(32, 64, (1, 16), stride=(1, 2)),
#             conv3=ConvBNReLU(1, 32, (8, 8)),
#             conv4=ConvBNReLU(32, 32, (8, 8)),
#             conv5=ConvBNReLU(32, 64, (1, 4)),
#             conv6=ConvBNReLU(64, 64, (1, 4)),
#             conv7=ConvBNReLU(64, 128, (1, 2)),
#             conv8=ConvBNReLU(128, 128, (1, 2)),
#             conv9=ConvBNReLU(128, 256, (1, 2)),
#             conv10=ConvBNReLU(256, 256, (1, 2)),
#             fc11=L.Linear(256 * 10 * 8, 4096),
#             fc12=L.Linear(4096, 4096),
#             fc13=L.Linear(4096, n_classes)
#         )
#         self.train = True
#         self.num_classes = n_classes

#     def __call__(self, x, target = None, mixup_type = 'sound', eligible = None, bs = None):
#         if eligible is not None:
#             layer_mix = random.choice(eligible)
#         else:
#             layer_mix = None
#         if target is not None :
#             target_reweighted = target
#         if layer_mix == 0:
#             if mixup_type == 'sound':
#                 shape = x.shape
#                 x = F.reshape(x, (x.shape[0],-1))
#                 for i in range(bs):
#                     r = random.random()
#                     sound1 = x[i*2]
#                     sound2 = x[i*2+1]
#                     x[i*2].data = U.mix(x[i*2],x[i*2+1],r,44100)
#                     x[i*2+1].data = U.mix(x[i*2+1],x[i*2],r,44100)
#                     target[i*2].data = (target[i*2]*r + target[i*2+1]*(1-r))
#                     target[i*2+1].data = (target[i*2]*(1-r) + target[i*2+1]*(r))
#                 x = F.reshape(x, shape)
#         h = self.conv1(x, self.train)
#         h = self.conv2(h, self.train)
#         h = F.max_pooling_2d(h, (1, 64))
#         h = F.swapaxes(h, 1, 2)
#         if layer_mix == 1:
#             if mixup_type == 'sound':
#                 shape = h.shape
#                 h = F.reshape(h, (h.shape[0],-1))
#                 for i in range(bs):
#                     r1 = random.random()
#                     r2 = random.random()
#                     h[i*2].data = U.mix(h[i*2],h[i*2+1],r1,44100)
#                     h[i*2+1].data = U.mix(h[i*2+1],h[i*2],r2,44100)
#                     target[i*2].data = (target[i*2]*r1 + target[i*2+1]*(1-r1))
#                     target[i*2+1].data = (target[i*2]*r2 + target[i*2+1]*(1-r2))
#                 h = F.reshape(h, shape)
#             elif mixup_type == 'normal':
#                 r1 = cupy.random.rand(bs,1,1)
#                 r2 = cupy.random.rand(bs,1,1)
#                 indices = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62]

#                 indices2 = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59, 61, 63]
#                 # out_shape = [r1.shape[0]] + [1 for _ in range(len(b.shape) - 1)]
#                 h[indices].data = r1*h[indices] + (1-r1)*h[indices2]
#                 h[indices2].data = h[indices]*(r2) + h[indices]*(1-r2)
#                 target[indices].data = (target[indices]*r1 + target[indices2]*(1-r1))
#                 target[indices2].data = (target[indices]*(r2) + target[indices2]*(1-r2))
#         h = self.conv3(h, self.train)
#         h = self.conv4(h, self.train)
#         h = F.max_pooling_2d(h, (5, 3))
#         if layer_mix == 2:
#             if mixup_type == 'sound':
#                 shape = h.shape
#                 h = F.reshape(h, (h.shape[0],-1))
#                 for i in range(bs):
#                     r1 = random.random()
#                     r2 = random.random()
#                     h[i*2].data = U.mix(h[i*2],h[i*2+1],r1,44100)
#                     h[i*2+1].data = U.mix(h[i*2+1],h[i*2],r2,44100)
#                     target[i*2].data = (target[i*2]*r1 + target[i*2+1]*(1-r1))
#                     target[i*2+1].data = (target[i*2]*r2 + target[i*2+1]*(1-r2))
#                 h = F.reshape(h, shape)
#             elif mixup_type == 'normal':
#                 r1 = cupy.random.rand(bs,1,1)
#                 r2 = cupy.random.rand(bs,1,1)
#                 indices = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62]

#                 indices2 = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59, 61, 63]
#                 # out_shape = [r1.shape[0]] + [1 for _ in range(len(b.shape) - 1)]
#                 h[indices].data = r1*h[indices] + (1-r1)*h[indices2]
#                 h[indices2].data = h[indices]*(r2) + h[indices]*(1-r2)
#                 target[indices].data = (target[indices]*r1 + target[indices2]*(1-r1))
#                 target[indices2].data = (target[indices]*(r2) + target[indices2]*(1-r2))
#         h = self.conv5(h, self.train)
#         h = self.conv6(h, self.train)
#         h = F.max_pooling_2d(h, (1, 2))
#         if layer_mix == 3:
#             if mixup_type == 'sound':
#                 shape = h.shape
#                 h = F.reshape(h, (h.shape[0],-1))
#                 for i in range(bs):
#                     r1 = random.random()
#                     r2 = random.random()
#                     h[i*2].data = U.mix(h[i*2],h[i*2+1],r1,44100)
#                     h[i*2+1].data = U.mix(h[i*2+1],h[i*2],r2,44100)
#                     target[i*2].data = (target[i*2]*r1 + target[i*2+1]*(1-r1))
#                     target[i*2+1].data = (target[i*2]*r2 + target[i*2+1]*(1-r2))
#                 h = F.reshape(h, shape)
#             elif mixup_type == 'normal':
#                 r1 = cupy.random.rand(bs,1,1)
#                 r2 = cupy.random.rand(bs,1,1)
#                 indices = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62]

#                 indices2 = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59, 61, 63]
#                 # out_shape = [r1.shape[0]] + [1 for _ in range(len(b.shape) - 1)]
#                 h[indices].data = r1*h[indices] + (1-r1)*h[indices2]
#                 h[indices2].data = h[indices]*(r2) + h[indices]*(1-r2)
#                 target[indices].data = (target[indices]*r1 + target[indices2]*(1-r1))
#                 target[indices2].data = (target[indices]*(r2) + target[indices2]*(1-r2))
#         h = self.conv7(h, self.train)
#         h = self.conv8(h, self.train)
#         h = F.max_pooling_2d(h, (1, 2))
#         if layer_mix == 4:
#             if mixup_type == 'sound':
#                 shape = h.shape
#                 h = F.reshape(h, (h.shape[0],-1))
#                 for i in range(bs):
#                     r1 = random.random()
#                     r2 = random.random()
#                     h[i*2].data = U.mix(h[i*2],h[i*2+1],r1,44100)
#                     h[i*2+1].data = U.mix(h[i*2+1],h[i*2],r2,44100)
#                     target[i*2].data = (target[i*2]*r1 + target[i*2+1]*(1-r1))
#                     target[i*2+1].data = (target[i*2]*r2 + target[i*2+1]*(1-r2))
#                 h = F.reshape(h, shape)
#             elif mixup_type == 'normal':
#                 r1 = cupy.random.rand(bs,1,1)
#                 r2 = cupy.random.rand(bs,1,1)
#                 indices = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62]

#                 indices2 = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59, 61, 63]
#                 # out_shape = [r1.shape[0]] + [1 for _ in range(len(b.shape) - 1)]
#                 h[indices].data = r1*h[indices] + (1-r1)*h[indices2]
#                 h[indices2].data = h[indices]*(r2) + h[indices]*(1-r2)
#                 target[indices].data = (target[indices]*r1 + target[indices2]*(1-r1))
#                 target[indices2].data = (target[indices]*(r2) + target[indices2]*(1-r2))
#         h = self.conv9(h, self.train)
#         h = self.conv10(h, self.train)
#         h = F.max_pooling_2d(h, (1, 2))
#         h = F.dropout(F.relu(self.fc11(h)))
#         h = F.dropout(F.relu(self.fc12(h)))

#         if target is not None:
#             return self.fc13(h), target
#         else:
#             return self.fc13(h)

class EnvNetv2(chainer.Chain):
    def __init__(self, n_classes):
        super(EnvNetv2, self).__init__(
            conv1=ConvBNReLU(1, 32, (1, 64), stride=(1, 2)),
            conv2=ConvBNReLU(32, 64, (1, 16), stride=(1, 2)),
            conv3=ConvBNReLU(1, 32, (8, 8)),
            conv4=ConvBNReLU(32, 32, (8, 8)),
            conv5=ConvBNReLU(32, 64, (1, 4)),
            conv6=ConvBNReLU(64, 64, (1, 4)),
            conv7=ConvBNReLU(64, 128, (1, 2)),
            conv8=ConvBNReLU(128, 128, (1, 2)),
            conv9=ConvBNReLU(128, 256, (1, 2)),
            conv10=ConvBNReLU(256, 256, (1, 2)),
            fc11=L.Linear(256 * 10 * 8, 4096),
            fc12=L.Linear(4096, 4096),
            fc13=L.Linear(4096, n_classes)
        )
        self.train = True
        self.num_classes = n_classes

    def __call__(self, x, target = None, mixup_type = 'sound', eligible = None, bs = None, perm=None):
        if eligible is not None:
            layer_mix = random.choice(eligible)
        else:
            layer_mix = None
        if target is not None :
            target_reweighted = target
        if layer_mix == 0:
            shape = x.shape
            x = F.reshape(x, (x.shape[0],-1))
            r = cupy.random.rand(bs*2).astype('float32')
            x, target = U.mixup_sound(x,target,r,44100,perm)
            x = F.reshape(x, shape)
        h = self.conv1(x, self.train)
        h = self.conv2(h, self.train)
        h = F.max_pooling_2d(h, (1, 64))
        h = F.swapaxes(h, 1, 2)
        if layer_mix == 1:
            if mixup_type == 'sound':
                shape = h.shape
                h = F.reshape(h, (h.shape[0],-1))
                r = cupy.random.rand(bs*2).astype('float32')
                h, target = U.mixup_sound(h,target,r,44100,perm)
                h = F.reshape(h, shape)
            elif mixup_type == 'normal':
                shape = h.shape
                h = F.reshape(h, (h.shape[0],-1))
                r1 = cupy.random.rand(bs*2,1).astype('float32')
                indices = [ 1,  0,  3,  2,  5,  4,  7,  6,  9,  8, 11, 10, 13, 12, 15, 14, 17,
                    16, 19, 18, 21, 20, 23, 22, 25, 24, 27, 26, 29, 28, 31, 30, 33, 32,
                    35, 34, 37, 36, 39, 38, 41, 40, 43, 42, 45, 44, 47, 46, 49, 48, 51,
                    50, 53, 52, 55, 54, 57, 56, 59, 58, 61, 60, 63, 62]
                # out_shape = [r1.shape[0]] + [1 for _ in range(len(b.shape) - 1)]
                h = (h*r1 + h[indices]*(1-r1))
                target = (target*cupy.reshape(r1,(bs*2,1)) + target[indices]*(1 - cupy.reshape(r1,(bs*2,1))))
                h = F.reshape(h, shape)
            elif mixup_type == 'normalized':
                r1 = cupy.random.rand(bs*2,1,1)
                indices = [ 1,  0,  3,  2,  5,  4,  7,  6,  9,  8, 11, 10, 13, 12, 15, 14, 17,
                    16, 19, 18, 21, 20, 23, 22, 25, 24, 27, 26, 29, 28, 31, 30, 33, 32,
                    35, 34, 37, 36, 39, 38, 41, 40, 43, 42, 45, 44, 47, 46, 49, 48, 51,
                    50, 53, 52, 55, 54, 57, 56, 59, 58, 61, 60, 63, 62]
                # out_shape = [r1.shape[0]] + [1 for _ in range(len(b.shape) - 1)]
                h.data = (r1*h.data + (1-r1)*h.data[indices])/( r1**2 + (1 - r1)**2 )
                target.data = (target*cupy.reshape(r1,(bs*2,1)) + target[indices]*(1 - cupy.reshape(r1,(bs*2,1))))

        h = self.conv3(h, self.train)
        h = self.conv4(h, self.train)
        h = F.max_pooling_2d(h, (5, 3))
        if layer_mix == 2:
            if mixup_type == 'sound':
                shape = h.shape
                h = F.reshape(h, (h.shape[0],-1))
                r = cupy.random.rand(bs*2).astype('float32')
                h, target = U.mixup_sound(h,target,r,44100,perm)
                h = F.reshape(h, shape)
            elif mixup_type == 'normal':
                shape = h.shape
                h = F.reshape(h, (h.shape[0],-1))
                r1 = cupy.random.rand(bs*2,1).astype('float32')
                indices = [ 1,  0,  3,  2,  5,  4,  7,  6,  9,  8, 11, 10, 13, 12, 15, 14, 17,
                    16, 19, 18, 21, 20, 23, 22, 25, 24, 27, 26, 29, 28, 31, 30, 33, 32,
                    35, 34, 37, 36, 39, 38, 41, 40, 43, 42, 45, 44, 47, 46, 49, 48, 51,
                    50, 53, 52, 55, 54, 57, 56, 59, 58, 61, 60, 63, 62]
                # out_shape = [r1.shape[0]] + [1 for _ in range(len(b.shape) - 1)]
                h = (h*r1 + h[indices]*(1-r1))
                target = (target*cupy.reshape(r1,(bs*2,1)) + target[indices]*(1 - cupy.reshape(r1,(bs*2,1))))
                h = F.reshape(h, shape)
            elif mixup_type == 'normalized':
                r1 = cupy.random.rand(bs*2,1,1)
                indices = [ 1,  0,  3,  2,  5,  4,  7,  6,  9,  8, 11, 10, 13, 12, 15, 14, 17,
                    16, 19, 18, 21, 20, 23, 22, 25, 24, 27, 26, 29, 28, 31, 30, 33, 32,
                    35, 34, 37, 36, 39, 38, 41, 40, 43, 42, 45, 44, 47, 46, 49, 48, 51,
                    50, 53, 52, 55, 54, 57, 56, 59, 58, 61, 60, 63, 62]
                # out_shape = [r1.shape[0]] + [1 for _ in range(len(b.shape) - 1)]
                h.data = (r1*h + (1-r1)*h[indices])/( r1**2 + (1 - r1)**2 )
                target.data = (target*cupy.reshape(r1,(bs*2,1)) + target[indices]*(1 - cupy.reshape(r1,(bs*2,1))))
        h = self.conv5(h, self.train)
        h = self.conv6(h, self.train)
        h = F.max_pooling_2d(h, (1, 2))
        if layer_mix == 3:
            if mixup_type == 'sound':
                shape = h.shape
                h = F.reshape(h, (h.shape[0],-1))
                r = cupy.random.rand(bs*2).astype('float32')
                h, target = U.mixup_sound(h,target,r,44100,perm)
                h = F.reshape(h, shape)
            elif mixup_type == 'normal':
                shape = h.shape
                h = F.reshape(h, (h.shape[0],-1))
                r1 = cupy.random.rand(bs*2,1).astype('float32')
                indices = [ 1,  0,  3,  2,  5,  4,  7,  6,  9,  8, 11, 10, 13, 12, 15, 14, 17,
                    16, 19, 18, 21, 20, 23, 22, 25, 24, 27, 26, 29, 28, 31, 30, 33, 32,
                    35, 34, 37, 36, 39, 38, 41, 40, 43, 42, 45, 44, 47, 46, 49, 48, 51,
                    50, 53, 52, 55, 54, 57, 56, 59, 58, 61, 60, 63, 62]
                # out_shape = [r1.shape[0]] + [1 for _ in range(len(b.shape) - 1)]
                h = (h*r1 + h[indices]*(1-r1))
                target = (target*cupy.reshape(r1,(bs*2,1)) + target[indices]*(1 - cupy.reshape(r1,(bs*2,1))))
                h = F.reshape(h, shape)
            elif mixup_type == 'normalized':
                r1 = cupy.random.rand(bs*2,1,1)
                indices = [ 1,  0,  3,  2,  5,  4,  7,  6,  9,  8, 11, 10, 13, 12, 15, 14, 17,
                    16, 19, 18, 21, 20, 23, 22, 25, 24, 27, 26, 29, 28, 31, 30, 33, 32,
                    35, 34, 37, 36, 39, 38, 41, 40, 43, 42, 45, 44, 47, 46, 49, 48, 51,
                    50, 53, 52, 55, 54, 57, 56, 59, 58, 61, 60, 63, 62]
                # out_shape = [r1.shape[0]] + [1 for _ in range(len(b.shape) - 1)]
                h.data = (r1*h + (1-r1)*h[indices])/( r1**2 + (1 - r1)**2 )
                target.data = (target*cupy.reshape(r1,(bs*2,1)) + target[indices]*(1 - cupy.reshape(r1,(bs*2,1))))
        h = self.conv7(h, self.train)
        h = self.conv8(h, self.train)
        h = F.max_pooling_2d(h, (1, 2))
        if layer_mix == 4:
            if mixup_type == 'sound':
                shape = h.shape
                h = F.reshape(h, (h.shape[0],-1))
                r = cupy.random.rand(bs*2).astype('float32')
                h, target = U.mixup_sound(h,target,r,44100,perm)
                h = F.reshape(h, shape)
            elif mixup_type == 'normal':
                shape = h.shape
                h = F.reshape(h, (h.shape[0],-1))
                r1 = cupy.random.rand(bs*2,1).astype('float32')
                indices = [ 1,  0,  3,  2,  5,  4,  7,  6,  9,  8, 11, 10, 13, 12, 15, 14, 17,
                    16, 19, 18, 21, 20, 23, 22, 25, 24, 27, 26, 29, 28, 31, 30, 33, 32,
                    35, 34, 37, 36, 39, 38, 41, 40, 43, 42, 45, 44, 47, 46, 49, 48, 51,
                    50, 53, 52, 55, 54, 57, 56, 59, 58, 61, 60, 63, 62]
                # out_shape = [r1.shape[0]] + [1 for _ in range(len(b.shape) - 1)]
                h = (h*r1 + h[indices]*(1-r1))
                target = (target*cupy.reshape(r1,(bs*2,1)) + target[indices]*(1 - cupy.reshape(r1,(bs*2,1))))
                h = F.reshape(h, shape)
            elif mixup_type == 'normalized':
                r1 = cupy.random.rand(bs*2,1,1)
                indices = [ 1,  0,  3,  2,  5,  4,  7,  6,  9,  8, 11, 10, 13, 12, 15, 14, 17,
                    16, 19, 18, 21, 20, 23, 22, 25, 24, 27, 26, 29, 28, 31, 30, 33, 32,
                    35, 34, 37, 36, 39, 38, 41, 40, 43, 42, 45, 44, 47, 46, 49, 48, 51,
                    50, 53, 52, 55, 54, 57, 56, 59, 58, 61, 60, 63, 62]
                # out_shape = [r1.shape[0]] + [1 for _ in range(len(b.shape) - 1)]
                h.data = (r1*h + (1-r1)*h[indices])/( r1**2 + (1 - r1)**2 )
                target.data = (target*cupy.reshape(r1,(bs*2,1)) + target[indices]*(1 - cupy.reshape(r1,(bs*2,1))))
        h = self.conv9(h, self.train)
        h = self.conv10(h, self.train)
        h = F.max_pooling_2d(h, (1, 2))
        h = F.dropout(F.relu(self.fc11(h)))
        h = F.dropout(F.relu(self.fc12(h)))
        if target is not None:
            return self.fc13(h), target
        else:
            return self.fc13(h)
