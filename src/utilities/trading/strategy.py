import numpy as np

def cross(ind1, ind2, is_above=True) -> np.ndarry:
    assert ind1.shape[0] == ind2.shape[0], "indicators are not the same size"
    current = ind1 > ind2
    previous = ind1.shift(1) < ind2.shift(1)
    cross = current & previous if is_above else ~current & ~previous
    assert ind1.shape[0] == cross.shape[0], "comparison results are not the same size as indicators"
    return cross


def cross_num(ind1, num, is_above=True) -> np.ndarry:
    current = ind1 > num
    previous = ind1.shift(1) < num
    cross = current & previous if is_above else ~current & ~previous
    exact = (ind1 == num)
    return (cross | exact)


def widen(ind1, ind2, period) -> np.ndarry:
    assert ind1.shape[0] == ind2.shape[0], "indicators are not the same size"
    res = (ind1.shift(1) - ind2.shift(1)) < (ind1 - ind2)
    for i in range(1, period):
        res = res & ((ind1.shift(i+1) - ind2.shift(i+1)) < (ind1.shift(i) - ind2.shift(i)))
    assert ind1.shape[0] == res.shape[0], "comparison results are not the same size as indicators"
    return res


def shrink(ind1, ind2, period) -> np.ndarry:
    assert ind1.shape[0] == ind2.shape[0], "indicators are not the same size"
    res = (ind1.shift(1) - ind2.shift(1)) > (ind1 - ind2)
    for i in range(1, period):
        res = res & ((ind1.shift(i+1) - ind2.shift(i+1)) > (ind1.shift(i) - ind2.shift(i)))
    assert ind1.shape[0] == res.shape[0], "comparison results are not the same size as indicators"
    return res


def count_positive(ind, period) -> np.ndarry:
    return ((ind.rolling(window=period).agg(lambda x: (x > 0).sum())) > period-1)


def count_negative(ind, period) -> np.ndarry:
    return ((ind.rolling(window=period).agg(lambda x: (x < 0).sum())) > period-1)



