import math
import numpy as np
from scipy import stats


def divisorGenerator(n):
    large_divisors = []
    for i in range(1, int(math.sqrt(n) + 1)):
        if n % i == 0:
            yield i
            if i * i != n:
                large_divisors.append(n / i)
    for divisor in reversed(large_divisors):
        yield divisor


def transprob(aval, nregions):  # (t,r)
    mat = np.zeros((nregions, nregions))
    norm = np.sum(aval, axis=0)
    for t in range(len(aval) - 1):
        ini = np.where(aval[t] == 1)
        mat[ini] += aval[t + 1]
    mat[norm != 0] = mat[norm != 0] / norm[norm != 0][:, None]
    return mat


def Transprob(ZBIN, nregions):  # (t,r)
    mat = np.zeros((nregions, nregions))
    A = np.sum(ZBIN, axis=1)
    a = np.arange(len(ZBIN))
    idx = np.where(A != 0)[0]
    aout = np.split(a[idx], np.where(np.diff(idx) != 1)[0] + 1)
    ifi = 0
    for iaut in range(len(aout)):
        if len(aout[iaut]) > 2:
            mat += transprob(ZBIN[aout[iaut]], nregions)
            ifi += 1
    mat = mat / ifi
    return mat


def consecutiveRanges(a):
    n = len(a)
    length = 1;
    list = []
    if (n == 0):
        return list
    for i in range(1, n + 1):
        if (i == n or a[i] - a[i - 1] != 1):
            if (length > 0):
                if (a[i - length] != 0):
                    temp = (a[i - length] - 1, a[i - 1])
                    list.append(temp)
            length = 1
        else:
            length += 1
    return list


def go_avalanches(data, thre=3., direc=0, binsize=1):
    if direc == 1:
        Zb = np.where(stats.zscore(data) > thre, 1, 0)
    elif direc == -1:
        Zb = np.where(stats.zscore(data) < -thre, 1, 0)
    elif direc == 0:
        Zb = np.where(np.abs(stats.zscore(data)) > thre, 1, 0)
    else:
        print('wrong direc')

    nregions = len(data[0])

    Zbin = np.reshape(Zb, (-1, binsize, nregions))
    Zbin = np.where(np.sum(Zbin, axis=1) > 0, 1, 0)

    dfb_ampl = np.sum(Zbin, axis=1).astype(float)
    dfb_a = dfb_ampl[dfb_ampl != 0]
    bratio = np.exp(np.mean(np.log(dfb_a[1:] / dfb_a[:-1])))
    NoAval = np.where(dfb_ampl == 0)[0]

    inter = np.arange(1, len(Zbin) + 1);
    inter[NoAval] = 0
    Avals_ranges = consecutiveRanges(inter)
    Avals_ranges = Avals_ranges[1:-1]  # remove the first and last avalanche for avoiding boundary effects

    Naval = len(Avals_ranges)  # number of avalanches
    Avalanches = {'dur': [], 'siz': [], 'ranges': Avals_ranges, 'Zbin': Zbin, 'bratio': bratio}  # duration and size
    for i in range(Naval):
        xi = Avals_ranges[i][0];
        xf = Avals_ranges[i][1]
        Avalanches['dur'].append(xf - xi)
        Avalanches['siz'].append(len(np.where(np.sum(Zbin[xi:xf], axis=0) > 0)[0]))

    return Avalanches


def Extract(lst):
    return list(list(zip(*lst))[0])


def Extract_end(lst):
    return list(list(zip(*lst))[1])
