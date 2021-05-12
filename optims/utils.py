from .adam_cps import Adam_CPS
from .sgd_cps import SGD_CPS

def get_optim(optim_param):
	if optim_param['optim'] == 'sgd_cps' or optim_param['optim'] == 'sgd':
		optim = SGD_CPS(lr=optim_param['lr'], momentum=optim_param['momentum'], nesterov=optim_param['nesterov'])
	elif optim_param['optim'] == 'adam_cps' or  optim_param['optim'] == 'adam':
		optim = Adam_CPS(lr=optim_param['lr'], betas=optim_param['betas'], amsgrad=optim_param['amsgrad'])
	else:
		raise NotImplementedError('only accept sgd_cps and adam_cps')
	return optim