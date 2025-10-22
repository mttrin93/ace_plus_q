import tensorflow as tf


def embedding_function(rho, mexp, ftype='FinnisSinclairShiftedScaled'):
    if ftype == 'FinnisSinclairShiftedScaled':
        return f_exp_shsc(rho, mexp)
    elif ftype == 'FinnisSinclair':
        return f_exp_old(rho, mexp)


def f_exp_old(rho, mexp):
    return tf.where(tf.less(tf.abs(rho), tf.constant(1e-10, dtype=rho.dtype)), mexp * rho, en_func_old(rho, mexp))


def en_func_old(rho, mexp):
    w = tf.constant(10., dtype=rho.dtype)
    y1 = w * rho ** 2
    g = tf.where(tf.less(tf.constant(30., dtype=rho.dtype), y1), 0. * rho, tf.exp(tf.negative(y1)))

    omg = 1. - g
    a = tf.abs(rho)
    y3 = tf.pow(omg * a + 1e-20, mexp)
    # y3 = tf.pow(omg * a, mexp)
    y2 = mexp * g * a
    f = tf.sign(rho) * (y3 + y2)

    return f


def f_exp_shsc(rho, mexp):
    eps = tf.constant(1e-10, dtype=rho.dtype)
    cond = tf.abs(tf.ones_like(rho, dtype=rho.dtype) * mexp - tf.constant(1., dtype=rho.dtype))
    mask = tf.where(tf.less(cond, eps), tf.ones_like(rho, dtype=tf.bool), tf.zeros_like(rho, dtype=tf.bool))

    arho = tf.abs(rho)
    # func = tf.where(mask, rho, tf.sign(rho) * (tf.sqrt(tf.abs(arho + 0.25 * tf.exp(-arho))) - 0.5 * tf.exp(-arho)))
    exprho = tf.exp(-arho)
    nx = 1. / mexp
    xoff = tf.pow(nx, (nx / (1.0 - nx))) * exprho
    yoff = tf.pow(nx, (1 / (1.0 - nx))) * exprho
    func = tf.where(mask, rho, tf.sign(rho) * (tf.pow(xoff + arho, mexp) - yoff))

    return func