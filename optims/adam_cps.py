import math
import numpy as np


class Adam_CPS(object):
	r"""Implements Adam algorithm for coherent pulse stacking

	It has been proposed in `Adam: A Method for Stochastic Optimization`_.

	Arguments:
		params (iterable): iterable of parameters to optimize or dicts defining
			parameter groups
		lr (float, optional): learning rate (default: 1e-3)
		betas (Tuple[float, float], optional): coefficients used for computing
			running averages of gradient and its square (default: (0.9, 0.999))
		eps (float, optional): term added to the denominator to improve
			numerical stability (default: 1e-8)
		amsgrad (boolean, optional): whether to use the AMSGrad variant of this
			algorithm from the paper `On the Convergence of Adam and Beyond`_
			(default: False)

	.. _Adam\: A Method for Stochastic Optimization:
		https://arxiv.org/abs/1412.6980
	.. _On the Convergence of Adam and Beyond:
		https://openreview.net/forum?id=ryQu7f-RZ
	"""

	def __init__(self, lr=1e-3, betas=(0.9, 0.999), eps=1e-15, amsgrad=False):
		if not 0.0 <= lr:
			raise ValueError("Invalid learning rate: {}".format(lr))
		if not 0.0 <= eps:
			raise ValueError("Invalid epsilon value: {}".format(eps))
		if not 0.0 <= betas[0] < 1.0:
			raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
		if not 0.0 <= betas[1] < 1.0:
			raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
		self.defaults = dict(lr=lr, betas=betas, eps=eps, amsgrad=amsgrad)
		self.state = {'step': 0}

	def step(self, grad):
		"""Performs a single optimization step.

		Arguments:
			closure (callable, optional): A closure that reevaluates the model
				and returns the loss.
		"""
		lr = self.defaults['lr']
		beta1, beta2 = self.defaults['betas']
		eps = self.defaults['eps']
		amsgrad = self.defaults['amsgrad']

		state = self.state
		if state['step'] == 0:
			# Exponential moving average of gradient values
			state['exp_avg'] = np.zeros_like(grad)
			# Exponential moving average of squared gradient values
			state['exp_avg_sq'] = np.zeros_like(grad)
			if amsgrad:
				# Maintains max of all exp. moving avg. of sq. grad. values
				state['max_exp_avg_sq'] = np.zeros_like(grad)

		exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
		if amsgrad:
			max_exp_avg_sq = state['max_exp_avg_sq']

		self.state['step'] += 1
		bias_correction1 = 1 - beta1 ** state['step']
		bias_correction2 = 1 - beta2 ** state['step']

		exp_avg = exp_avg*beta1 + (1-beta1)*grad
		exp_avg_sq = exp_avg_sq * beta2 + (1 - beta2) * np.multiply(grad,grad)
		self.state['exp_avg'] = exp_avg
		self.state['exp_avg_sq'] = exp_avg_sq
		if amsgrad:
			# Maintains the maximum of all 2nd moment running avg. till now
			max_exp_avg_sq = np.maximum(max_exp_avg_sq,exp_avg_sq)
			self.state['max_exp_avg_sq'] = max_exp_avg_sq
			denom = np.sqrt(max_exp_avg_sq) / math.sqrt(bias_correction2)+eps
		else:
			denom = np.sqrt(exp_avg_sq)
			denom = denom / math.sqrt(bias_correction2) + eps

		step_size = lr / bias_correction1
		delta = step_size*exp_avg/denom

		return delta
