import tensorflow as tf
import numpy as np

from scipy.special import sici


def chebvander(x, deg):
	"""TF implementation of pydoc:numpy.polynomial.chebyshev.chebvander."""
	# x = tf.convert_to_tensor(x)

	v = [tf.ones_like(x)]
	if deg > 0:
		x2 = 2 * x
		v += [x]

		for i in range(2, deg):
			v += [v[i - 1] * x2 - v[i - 2]]

	v = tf.stack(v)

	# v_shape = v.get_shape().as_list()
	# roll = tf.unstack(tf.range(len(v_shape)))
	# roll = tf.concat([roll[1:], [roll[0]]], axis=0)
	# v = tf.transpose(v, roll)

	v = tf.transpose(v, [1, 2, 0])

	return v

def bessel_j_base(x, order):
	j = [tf.math.special.bessel_j0(x)]

	if order > 0:
		j += [tf.math.special.bessel_j1(x)]
		for v in range(2, order):
			j += [((2 * (v - 1)) / x) * j[v - 1] - j[v - 2]]

	j = tf.stack(j)
	j = tf.transpose(j, [1, 2, 0])

	return j[:, 0, :]

def bessel_y_base(x, order):
	y = [tf.math.special.bessel_y0(x)]

	if order > 0:
		y += [tf.math.special.bessel_y1(x)]
		for v in range(2, order):
			y += [((2 * (v - 1)) / x) * y[v - 1] - y[v - 2]]

	y = tf.stack(y)
	y = tf.transpose(y, [1, 2, 0])

	return y[:, 0, :]

def sinc(x):
	return tf.where(x != 0, tf.math.sin(x)/x, tf.ones_like(x, dtype=x.dtype))

def fn(x, rc, n):
	PI = tf.constant(np.pi, dtype=x.dtype)
	return tf.pow(tf.constant(-1, dtype=x.dtype), n) * \
		   tf.math.sqrt(tf.constant(2., dtype=x.dtype)) * PI / tf.pow(rc, 3. / 2) \
		   * (n + 1) * (n + 2) / tf.math.sqrt((n + 1) ** 2 + (n + 2) ** 2) \
		   * (sinc(x * (n + 1) * PI / rc) + sinc(x * (n + 2) * PI / rc))

def simplified_bessel(x, rc, deg):
	sbf = [fn(x, rc, tf.constant(0, dtype=x.dtype))]
	d = [tf.constant(1, dtype=x.dtype)]
	if deg > 0:
		for i in range(1, deg):
			n = tf.constant(i, dtype=x.dtype)
			en = n ** 2 * (n + 2) ** 2 / (4 * (n + 1) ** 4 + 1)
			dn = 1 - en / d[i - 1]
			d += [dn]
			sbf += [1 / tf.math.sqrt(d[i]) * (fn(x, rc, n) + tf.math.sqrt(en / d[i - 1]) * sbf[i - 1])]

	sbf = tf.stack(sbf)
	sbf = tf.transpose(sbf, [1, 2, 0])

	return sbf[:, 0, :]

def legendre_base(x, order):
	l = [tf.ones_like(x)]
	if order > 0:
		l += [x]
		for i in range(2, order):
			l += [((2 * i - 1) * l[i - 1] * x - (i - 1) * l[i - 2])/i]

	l = tf.stack(l)
	l = tf.transpose(l, [1, 2, 0])

	return l[:, 0, :]

def cutoff_func_cos(x, rcut):
	condition1 = tf.less(x, rcut)
	cutoff = (1. + tf.cos(tf.constant(np.pi, dtype=x.dtype) * x/rcut)) * 0.5

	return tf.where(condition1, cutoff, tf.zeros_like(x, dtype=x.dtype))

def cutoff_func_poly(r, rin, delta):
	x = 1 - 2 * (1 + (r - rin) / (delta + 1e-8))
	f = 7.5 * (x / 4 - x ** 3 / 6 + x ** 5 / 20)
	condition1 = tf.less(rin, r)
	condition2 = tf.less(r, rin - delta)

	val1 = tf.zeros_like(r, dtype=r.dtype)
	val2 = tf.ones_like(r, dtype=r.dtype)

	return tf.where(condition1, val1, tf.where(condition2, val2, 0.5 * (1 + f)))


def cutoff_func_p_order_poly(r, p):
	return 1 - (p + 1) * (p + 2) / 2 * r ** p + p * (p + 2) * r ** (p + 1) - p * (p + 1) / 2 * r ** (p + 2)

def cutoff_func_lin_cos(x, rcut, lin_fraq):
	condition1 = tf.less(x, rcut)
	condition2 = tf.less(x, rcut * lin_fraq)
	rcc = (x - rcut * lin_fraq) / (rcut - rcut * lin_fraq)
	cutoff = (1. + tf.cos(tf.constant(np.pi, dtype=x.dtype) * rcc)) * 0.5

	return tf.where(condition1,
					tf.where(condition2, tf.ones_like(x, dtype=x.dtype), cutoff),
					tf.zeros_like(x, dtype=x.dtype))


def scale_distance(x, lmbda, rcut):
	x_scaled = 1. - 2. * ((tf.exp(-lmbda * (x / rcut - 1.)) - 1.) / (tf.exp(lmbda) - 1.))

	return x_scaled


def compute_mu_sigma_sin_bessel(nmax):
	ret = np.zeros((nmax, 2))
	rc = 1
	for i in range(nmax):
		n = i + 1
		npi = n * np.pi
		mu = np.sqrt(2 / rc) * sici(npi)[0]
		sigma2 = 2 * npi * sici(2 * npi)[0] / rc ** 2 + (np.cos(2 * npi) - 1) / rc / rc - \
				 2 ** (3 / 2) / np.sqrt(rc) * mu * sici(npi)[0] + mu ** 2 * rc
		sigma = np.sqrt(sigma2)
		ret[i, 0] = mu
		ret[i, 1] = sigma

	return ret

def norm_sin_bessel(r, cut, nfuncs, p):
	funcs = []
	PI = tf.constant(np.pi, dtype=r.dtype)
	rc = tf.constant(1, dtype=r.dtype)
	r = r / cut
	mu_s = tf.constant(compute_mu_sigma_sin_bessel(nfuncs), dtype=r.dtype)
	for i in range(nfuncs):
		mu = mu_s[i, 0]
		s = mu_s[i, 1]
		n = tf.constant(i + 1, dtype=r.dtype)
		f = (tf.math.sqrt(2 / rc) * tf.math.sin(n * r * PI / rc) / r - mu) / s
		funcs += [f * cutoff_func_p_order_poly(r / rc, p)]
	funcs = tf.stack(funcs)
	funcs = tf.transpose(funcs, [1, 2, 0])

	return funcs[:, 0, :]


def sin_bessel(r, cut, nfuncs, p):
	funcs = []
	PI = tf.constant(np.pi, dtype=r.dtype)
	rc = tf.constant(1, dtype=r.dtype)
	r = r/cut
	for i in range(nfuncs):
		n = tf.constant(i + 1, dtype=r.dtype)
		funcs += [tf.math.sqrt(2 / rc) * tf.math.sin(n * r * PI / rc) / r * cutoff_func_p_order_poly(r / rc, p)]
	funcs = tf.stack(funcs)
	funcs = tf.transpose(funcs, [1, 2, 0])

	return funcs[:, 0, :]


def gaussian(x, probe, width):
	g = tf.exp(-width * ((tf.reshape(x, [-1, 1]) - probe) ** 2))

	return g


def cheb_exp_cos(d_ij, nfuncs, cutoff, lmbda):
	d_ij_ch_domain = 1. - 2. * ((tf.exp(-lmbda * (d_ij / cutoff - 1.)) - 1.) / (tf.exp(lmbda) - 1.))
	chebpol = chebvander(d_ij_ch_domain, nfuncs)
	chebpol = chebpol[:, 0, :]
	cos_cut = cutoff_func_cos(d_ij, cutoff)
	gk = tf.where(tf.math.equal(chebpol, 1.),
				  chebpol * cos_cut, 0.5 * (1 - chebpol) * cos_cut,
				  name='where_chebexp')

	# y00 = 1  # / tf.sqrt(4 * tf.constant(np.pi, dtype=var_dtype))
	if nfuncs == 1:
		return gk[:, :-1]
	else:
		return gk  # * y00


def cheb_pow(d_ij, nfuncs, cutoff, lmbda):
	d_ij_ch_domain = 2.0 * (1.0 - tf.abs(1.0 - d_ij / cutoff) ** lmbda) - 1.0

	chebpol = chebvander(d_ij_ch_domain, nfuncs + 1)
	chebpol = chebpol[:, 0, :]
	chebpol = 0.5 - 0.5 * chebpol[:, 1:]
	res = tf.where(tf.less(d_ij, cutoff), chebpol, tf.zeros_like(chebpol, dtype=d_ij.dtype), name='where_chebpow')

	return res


def cheb_mag(d_ij, nfuncs, cutoff):
	d_ij_ch_domain = 1.0 - 2.0 * (d_ij / cutoff) ** 2.0

	chebpol = chebvander(d_ij_ch_domain, nfuncs + 1)
	chebpol = chebpol[:, 0, :]
	res = tf.where(tf.less(d_ij, cutoff), chebpol, tf.zeros_like(chebpol, dtype=d_ij.dtype), name='where_chebmag')

	return res[:, 1:]


def cheb_plain(d_ij, nfuncs):
	chebpol = chebvander(d_ij, nfuncs)[:, 0, :nfuncs]
	chebpol = tf.reshape(chebpol, [-1, nfuncs])

	return chebpol


def bessel_j(d_ij, nfuncs, cutoff, lmbda):
	x = d_ij ** lmbda
	j = bessel_j_base(x, nfuncs)
	# cut = radial_functions.cutoff_func_lin_cos(d_ij, cutoff, 0.5)
	cut = cutoff_func_cos(d_ij, cutoff)

	return j * cut


def bessel_y(d_ij, nfuncs, cutoff, lmbda):
	x = d_ij ** lmbda
	y = bessel_y_base(x, nfuncs)
	# cut = radial_functions.cutoff_func_lin_cos(d_ij, cutoff, 0.5)
	cut = cutoff_func_cos(d_ij, cutoff)

	return y * cut


def bessel_s(d_ij, nfuncs, cutoff):
	y = simplified_bessel(d_ij, cutoff, nfuncs)
	res = tf.where(tf.less(d_ij, cutoff), y, tf.zeros_like(y, dtype=d_ij.dtype), name='where_bessels')

	return res


def legendre(d_ij, nfuncs, cutoff, lmbda):
	d_ij_l_domain = 2.0 * (1.0 - tf.abs(1.0 - d_ij / cutoff) ** lmbda) - 1.0

	legpol = legendre_base(d_ij_l_domain, nfuncs + 1)
	legpol = (legpol[:, 1:] - 1) / 2
	res = tf.where(tf.less(d_ij, cutoff), legpol, tf.zeros_like(legpol, dtype=d_ij.dtype), name='where_legpol')

	return res

def rad_sin_bessel(d_ij, nfuncs, cutoff, p):
	#rbf = sin_bessel(d_ij, cutoff, nfuncs, p)
	rbf = norm_sin_bessel(d_ij, cutoff, nfuncs, p)
	# legpol = (legpol[:, 1:] - 1) / 2
	res = tf.where(tf.less(d_ij, cutoff), rbf, tf.zeros_like(rbf, dtype=d_ij.dtype), name='where_rbf')

	return res

def radial_function(d_ij, nfunc: int = 12, ftype: str = 'SBessel', cutoff: float = 5., lmbda: float = None):
	if ftype == 'ChebExpCos':
		return cheb_exp_cos(d_ij, nfunc, cutoff, lmbda)
	elif ftype == 'ChebPow':
		return cheb_pow(d_ij, nfunc, cutoff, lmbda)
	elif ftype == 'ChebMag':
		return cheb_mag(d_ij, nfunc, cutoff)
	elif ftype == 'TEST_LegendrePow':
		return legendre(d_ij, nfunc, cutoff, lmbda)
	elif ftype == 'TEST_BesselFirst':
		return bessel_j(d_ij, nfunc, cutoff, lmbda)
	elif ftype == 'TEST_BesselSecond':
		return bessel_y(d_ij, nfunc, cutoff, lmbda)
	elif ftype == 'TEST_SBessel':
		import warnings
		warnings.simplefilter('always', DeprecationWarning)
		warnings.warn('Name "TEST_SBessel" is deprecated, use "SBessel" instead.', DeprecationWarning)
		return bessel_s(d_ij, nfunc, cutoff)
	elif ftype == 'SBessel':
		return bessel_s(d_ij, nfunc, cutoff)
	elif ftype == 'RadSinBessel':
		return rad_sin_bessel(d_ij, nfunc, cutoff, lmbda)
	else:
		raise ValueError('Unknown radial function type {}'.format(ftype))
