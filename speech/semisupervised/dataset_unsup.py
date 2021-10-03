import os
import numpy as np
import random
import chainer

import utils as U
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

class Label_SoundDataset(chainer.dataset.DatasetMixin):
    def __init__(self, sounds, labels, train=True):
        self.base = chainer.datasets.TupleDataset(sounds, labels)
        self.train = train
        self.preprocess_funcs = self.preprocess_setup()

    def __len__(self):
        return len(self.base)//2

    def preprocess_setup(self):
        funcs = []
        # if self.opt.strongAugment:
        funcs += [U.random_scale(1.25)]

        funcs += [U.padding(66650 // 2),
                  U.random_crop(66650),
                  U.normalize(32768.0),
                  ]

        return funcs

    def preprocess(self, sound):
        for f in self.preprocess_funcs:
            sound = f(sound)

        return sound

    def get_example(self, i):
        sound, label = self.base[i]
        sound = self.preprocess(sound).astype(np.float32)
        label = np.array(label, dtype=np.int32)

        sound = U.random_gain(6)(sound).astype(np.float32)
        eye = np.eye(10)
        label = eye[label]
        return sound, label

class Unlabel_SoundDataset(chainer.dataset.DatasetMixin):
    def __init__(self, sounds, train=True):
        self.base = sounds
        self.train = train
        self.preprocess_funcs = self.preprocess_setup()

    def __len__(self):
        return len(self.base)//2

    def preprocess_setup(self):
        funcs = []
        # if self.opt.strongAugment:
        funcs += [U.random_scale(1.25)]

        funcs += [U.padding(66650 // 2),
                  U.random_crop(66650),
                  U.normalize(32768.0),
                  ]

        return funcs

    def preprocess(self, sound):
        for f in self.preprocess_funcs:
            sound = f(sound)

        return sound

    def get_example(self, i):
        sound = self.base[i]
        sound = self.preprocess(sound).astype(np.float32)

        sound = U.random_gain(6)(sound).astype(np.float32)
        sound_fft = np.fft.ifft(np.fft.fft(sound)).real.astype(np.float32)
        return sound_fft, sound


class SoundDataset_Val(chainer.dataset.DatasetMixin):
    def __init__(self, sounds, labels, opt, train=True):
        self.base = chainer.datasets.TupleDataset(sounds, labels)
        self.opt = opt
        self.train = train
        self.mix = (opt.BC and train)
        self.preprocess_funcs = self.preprocess_setup()

    def __len__(self):
        return len(self.base)

    def preprocess_setup(self):
        if self.train:
            funcs = []
            if self.opt.strongAugment:
                funcs += [U.random_scale(1.25)]

            funcs += [U.padding(self.opt.inputLength // 2),
                      U.random_crop(self.opt.inputLength),
                      U.normalize(32768.0),
                      ]

        else:
            funcs = [U.padding(self.opt.inputLength // 2),
                     U.normalize(32768.0),
                     U.multi_crop(self.opt.inputLength, self.opt.nCrops),
                     ]

        return funcs

    def preprocess(self, sound):
        for f in self.preprocess_funcs:
            sound = f(sound)

        return sound

    def get_example(self, i):
        if self.mix:  # Training phase of BC learning
            # Select two training examples
            while True:
                sound1, label1 = self.base[random.randint(0, len(self.base) - 1)]
                sound2, label2 = self.base[random.randint(0, len(self.base) - 1)]
                if label1 != label2:
                    break
            sound1 = self.preprocess(sound1)
            sound2 = self.preprocess(sound2)

            eye = np.eye(self.opt.nClasses)
            label1 = eye[label1]
            label2 = eye[label2]
            sound = np.concatenate((sound1,sound2))
            label = np.concatenate((label1,label2))

        else:  # Training phase of standard learning or testing phase
            sound, label = self.base[i]
            sound = self.preprocess(sound).astype(np.float32)
            label = np.array(label, dtype=np.int32)

        if self.train and self.opt.strongAugment:
            sound1 = U.random_gain(6)(sound1).astype(np.float32)
            sound2 = U.random_gain(6)(sound1).astype(np.float32)
            sound = np.concatenate((sound1,sound2))
            label = np.concatenate((label1,label2))
        return sound, label


def setup(opt, split):
    dataset = np.load(os.path.join(opt.data, 'wav_urdu_train.npz'))
    val_dataset = np.load(os.path.join(opt.data, 'wav_urdu_val.npz'))
    # Split to train and val
    train_sounds = []
    train_labels = []
    unlabel_train_sounds = []
    unlabel_train_labels = []
    val_sounds = []
    val_labels = []
    split = 1
    label_cts = {0:0, 1:0, 2:0, 3:0}

    for i in range(1, 2):
        sounds = dataset['fold{}'.format(i)].item()['sounds']
        labels = dataset['fold{}'.format(i)].item()['labels']
        if i != split:
            val_sounds.extend(sounds)
            val_labels.extend(labels)
        else:
            label_t_sounds = []
            label_t_labels = []
            unlabel_t_sounds = []
            unlabel_t_labels = []
            sounds, labels = shuffle(np.array(sounds), np.array(labels))
            for l, s in zip(labels, sounds):
              if label_cts[l] >= opt.nPartial:
                if label_cts[l] == opt.nUnsuper:
                  continue
                unlabel_t_sounds.append(s)
                unlabel_t_labels.append(l)
                label_cts[l] += 1
              else:
                label_t_sounds.append(s)
                label_t_labels.append(l)
                label_cts[l] += 1
            train_sounds.extend(label_t_sounds)
            train_labels.extend(label_t_labels)
            unlabel_train_sounds.extend(unlabel_t_sounds)
            unlabel_train_labels.extend(unlabel_t_labels)
            # print(label_cts)

    print(len(train_sounds))
    print(len(train_labels))
    print(len(unlabel_t_sounds))
    print(len(unlabel_train_sounds))
    x_val = val_dataset["fold1"].item()['sounds']
    y_val = val_dataset["fold1"].item()['labels']
    val_data = SoundDataset_Val(x_val, y_val, opt, train=False)
    val_iter = chainer.iterators.SerialIterator(val_data, 1, repeat=False, shuffle=False)
    label_train_data = Label_SoundDataset(train_sounds, train_labels, train=True)
    unlabel_train_data = Unlabel_SoundDataset(unlabel_train_sounds, train=True)
    return label_train_data, unlabel_train_data, val_iter
