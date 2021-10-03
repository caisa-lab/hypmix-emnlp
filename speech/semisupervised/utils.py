import numpy as np
import random
import chainer.functions as F
import chainer
from chainer import cuda
import cupy
import cupy as cp

# Default data augmentation

def numpy_mobius_add(x, y, c, dim=-1):
    x2 = cp.sum(cp.power(x.data, 2), axis=dim, keepdims=True)
    y2 = cp.sum(cp.power(y.data, 2), axis=dim, keepdims=True)
    xy = cp.sum((x.data * y.data), axis=dim, keepdims=True)
    num = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
    denom = 1 + 2 * c * xy + c ** 2 * x2 * y2
    return num / cp.clip(denom, a_min=1e-15)



def numpy_tanh(x, clamp=15):
    return cp.tanh(cp.clip(x, a_min=-clamp, a_max=clamp))

def numpy_expmap0(u, c):
    sqrt_c = c ** 0.5
    u_norm = cp.clip(cp.linalg.norm(u.data, axis=-1, ord=2, keepdims=True), a_min=1e-15)
    gamma_1 = numpy_tanh(sqrt_c * u_norm) * u / (sqrt_c * u_norm)
    return gamma_1


def numpy_artanh(x):
    return cp.arctanh(cp.clip(x, a_min=-1 + 1e-15, a_max=1 - 1e-15))

def numpy_logmap0(p, c):
    # print ('In logmap')
    sqrt_c = c ** 0.5
    p_norm = cp.clip(cp.linalg.norm(p.data, axis=-1, ord=2, keepdims=True), a_min=1e-15)
    scale = 1. / sqrt_c * numpy_artanh(sqrt_c * p_norm) / p_norm
    return scale * p


def numpy_mobius_scalar_mul(r, x, c, dim = -1):
    x_norm = cp.clip(cp.linalg.norm(x.data, axis=dim, ord=2, keepdims=True), a_min=1e-15)
    res_c = numpy_tanh(r * numpy_artanh(x_norm)) * (x / x_norm)
    return res_c

def padding(pad):
    def f(sound):
        return np.pad(sound, pad, 'constant')

    return f


def random_crop(size):
    def f(sound):
        org_size = len(sound)
        start = random.randint(0, org_size - size)
        return sound[start: start + size]

    return f


def normalize(factor):
    def f(sound):
        return sound / factor

    return f


# For strong data augmentation
def random_scale(max_scale, interpolate='Linear'):
    def f(sound):
        scale = np.power(max_scale, random.uniform(-1, 1))
        output_size = int(len(sound) * scale)
        ref = np.arange(output_size) / scale
        if interpolate == 'Linear':
            ref1 = ref.astype(np.int32)
            ref2 = np.minimum(ref1 + 1, len(sound) - 1)
            r = ref - ref1
            scaled_sound = sound[ref1] * (1 - r) + sound[ref2] * r
        elif interpolate == 'Nearest':
            scaled_sound = sound[ref.astype(np.int32)]
        else:
            raise Exception('Invalid interpolation mode {}'.format(interpolate))

        return scaled_sound

    return f


def random_gain(db):
    def f(sound):
        return sound * np.power(10, random.uniform(-db, db) / 20.0)

    return f


# For testing phase
def multi_crop(input_length, n_crops):
    def f(sound):
        stride = (len(sound) - input_length) // (n_crops - 1)
        sounds = [sound[stride * i: stride * i + input_length] for i in range(n_crops)]
        return np.array(sounds)

    return f


# For BC learning
# def a_weight(fs, n_fft, min_db=-80.0):
#     freq = np.linspace(0, fs // 2, n_fft // 2 + 1)
#     freq_sq = np.power(freq, 2)
#     freq_sq[0] = 1.0
#     weight = 2.0 + 20.0 * (2 * np.log10(12194) + 2 * np.log10(freq_sq)
#                            - np.log10(freq_sq + 12194 ** 2)
#                            - np.log10(freq_sq + 20.6 ** 2)
#                            - 0.5 * np.log10(freq_sq + 107.7 ** 2)
#                            - 0.5 * np.log10(freq_sq + 737.9 ** 2))
#     weight = np.maximum(weight, min_db)

#     return weight

def a_weight(fs, n_fft, min_db=-80.0):
    freq = cupy.linspace(0, fs // 2, n_fft // 2 + 1)
    freq_sq = cupy.power(freq, 2)
    freq_sq[0] = 1.0
    weight = 2.0 + 20.0 * (2 * cupy.log10(12194) + 2 * cupy.log10(freq_sq)
                           - cupy.log10(freq_sq + 12194 ** 2)
                           - cupy.log10(freq_sq + 20.6 ** 2)
                           - 0.5 * cupy.log10(freq_sq + 107.7 ** 2)
                           - 0.5 * cupy.log10(freq_sq + 737.9 ** 2))
    weight = cupy.maximum(weight, min_db)

    return weight


# def compute_gain(sound, fs, min_db=-80.0, mode='A_weighting'):
#     if fs == 16000:
#         n_fft = 2048
#     elif fs == 44100:
#         n_fft = 4096
#     else:
#         raise Exception('Invalid fs {}'.format(fs))
#     stride = n_fft // 2

#     gain = []
#     for i in range(0, len(sound) - n_fft + 1, stride):
#         if mode == 'RMSE':
#             g = np.mean(sound[i: i + n_fft] ** 2)
#         elif mode == 'A_weighting':
#             spec = np.fft.rfft(np.hanning(n_fft + 1)[:-1] * sound[i: i + n_fft])
#             power_spec = np.abs(spec) ** 2
#             a_weighted_spec = power_spec * np.power(10, a_weight(fs, n_fft) / 10)
#             g = np.sum(a_weighted_spec)
#         else:
#             raise Exception('Invalid mode {}'.format(mode))
#         gain.append(g)

#     gain = np.array(gain)
#     gain = np.maximum(gain, np.power(10, min_db / 10))
#     gain_db = 10 * np.log10(gain)

#     return gain_db

def compute_gain(sound, fs, min_db=-80.0, mode='A_weighting'):
    if fs == 16000:
        n_fft = 2048
    elif fs == 44100:
        n_fft = 4096
    else:
        raise Exception('Invalid fs {}'.format(fs))
    stride = n_fft // 2

    gain = None
    for i in range(0, len(sound[0]) - n_fft + 1, stride):
        if mode == 'RMSE':
            g = cupy.mean(sound[i: i + n_fft] ** 2, axis=1)
        elif mode == 'A_weighting':
            spec = cupy.fft.rfft(cupy.hanning(n_fft + 1)[:-1] * sound[:,i: i + n_fft])
            power_spec = cupy.abs(spec) ** 2
            a_weighted_spec = power_spec * cupy.power(10, a_weight(fs, n_fft) / 10)
            g = cupy.sum(a_weighted_spec, axis=1)
        else:
            raise Exception('Invalid mode {}'.format(mode))
        if i==0:
            gain = g.reshape([-1,1])
        else:
            gain = cupy.concatenate((gain, g.reshape([-1,1])),axis=1)

    gain = cupy.maximum(gain, cupy.power(10, min_db / 10))
    gain_db = 10 * cupy.log10(gain)

    return gain_db


# def mix(sound1, sound2, r, fs):
#     gain1 = np.max(compute_gain(cuda.to_cpu(sound1.data), fs))  # Decibel
#     gain2 = np.max(compute_gain(cuda.to_cpu(sound2.data), fs))
#     t = 1.0 / (1 + np.power(10, (gain1 - gain2) / 20.) * (1 - r) / r)
#     sound = ((sound1 * t + sound2 * (1 - t)) / np.sqrt(t ** 2 + (1 - t) ** 2))

#     return sound

def mix(sound1, sound2, r, fs):
    gain1 = cupy.max(compute_gain(sound1.data, fs),axis=1)  # Decibel
    gain2 = cupy.max(compute_gain(sound2.data, fs),axis=1)
    t = 1.0 / (1 + cupy.power(10, (gain1 - gain2) / 20.) * (1 - r) / r)
    sound1_hyp = numpy_expmap0(sound1, c=cp.array([1.0]))
    sound2_hyp = numpy_expmap0(sound2, c=cp.array([1.0]))
    # print ('Sound1 hyp: ', sound1_hyp.shape, ' type: ', type(sound1_hyp))
    # print ('t: ', t)
    # sound1_t =  pmath_geo.mobius_pointwise_mul(sound1_hyp, torch.FloatTensor(t), c=torch.tensor([1.0]))
    # sound2_t =  pmath_geo.mobius_pointwise_mul(sound2_hyp, torch.FloatTensor((1-t)), c=torch.tensor([1.0]))
    # sound1_t =  numpy_mobius_scalar_mul(cp.array([t]), sound1_hyp, c=cp.array([1.0]))

    sound1_t =  numpy_mobius_scalar_mul(t[:, None], sound1_hyp, c=cp.array([1.0]))
    sound2_t =  numpy_mobius_scalar_mul(1-t[:, None], sound2_hyp, c=cp.array([1.0]))
    # sound2_t =  numpy_mobius_scalar_mul(cp.array(([1-t])), sound2_hyp, c=cp.array([1.0]))
    sound1_plus_sound2 = numpy_mobius_add(sound1_t, sound2_t, c=cp.array([1.0]))
    updated_sound = numpy_logmap0(sound1_plus_sound2, c=cp.array([1.0]))
    # updated_sound = updated_sound.cpu().detach().numpy()
    # sound = ((sound1 * t + sound2 * (1 - t)) / np.sqrt(t ** 2 + (1 - t) ** 2))
    sound = (updated_sound / cp.sqrt(t[:,None] ** 2 + (1 - t[:,None]) ** 2))


    # sound = ((sound1 * t[:,None] + sound2 * (1 - t[:,None])) / cupy.sqrt(t[:,None] ** 2 + (1 - t[:,None]) ** 2))

    return sound

def mixup_sound(out, target_reweighted,r,fs, perm):
    # indices = [ 1,  0,  3,  2,  5,  4,  7,  6,  9,  8, 11, 10, 13, 12, 15, 14, 17,
    #    16, 19, 18, 21, 20, 23, 22, 25, 24, 27, 26, 29, 28, 31, 30, 33, 32,
    #    35, 34, 37, 36, 39, 38, 41, 40, 43, 42, 45, 44, 47, 46, 49, 48, 51,
    #    50, 53, 52, 55, 54, 57, 56, 59, 58, 61, 60, 63, 62]
    indices = perm
    sound = mix(out,out[indices],r,fs)
    target_shuffled_onehot = target_reweighted[indices]
    target_reweighted.data = target_reweighted.data * r[:,None] + target_shuffled_onehot.data * (1 - r[:,None])
    return sound, target_reweighted


def kl_divergence(y, t):
    entropy = - F.sum(t[t.data.nonzero()] * F.log(t[t.data.nonzero()]))
    crossEntropy = - F.sum(t * F.log_softmax(y))

    return (crossEntropy - entropy) / y.shape[0]


# Convert time representation
def to_hms(time):
    h = int(time // 3600)
    m = int((time - h * 3600) // 60)
    s = int(time - h * 3600 - m * 60)
    if h > 0:
        line = '{}h{:02d}m'.format(h, m)
    else:
        line = '{}m{:02d}s'.format(m, s)

    return line
