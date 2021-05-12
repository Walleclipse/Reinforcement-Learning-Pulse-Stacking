

class SGD_CPS(object):
    r"""Implements stochastic gradient descent (optionally with momentum) for coherent pulse stacking

    Nesterov momentum is based on the formula from
    `On the importance of initialization and momentum in deep learning`__.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
        momentum (float, optional): momentum factor (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)


    __ http://www.cs.toronto.edu/%7Ehinton/absps/momentum.pdf

    .. note::
        The implementation of SGD with Momentum/Nesterov subtly differs from
        Sutskever et. al. and implementations in some other frameworks.

        Considering the specific case of Momentum, the update can be written as

        .. math::
                  v_{t+1} = \mu * v_{t} + g_{t+1} \\
                  p_{t+1} = p_{t} - lr * v_{t+1}

        where p, g, v and :math:`\mu` denote the parameters, gradient,
        velocity, and momentum respectively.

        This is in contrast to Sutskever et. al. and
        other frameworks which employ an update of the form

        .. math::
             v_{t+1} = \mu * v_{t} + lr * g_{t+1} \\
             p_{t+1} = p_{t} - v_{t+1}

        The Nesterov version is analogously modified.
    """

    def __init__(self,  lr, momentum=0,dampening=0,nesterov=False):
        if  lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")

        self.defaults= dict(lr=lr, momentum=momentum,dampening=dampening,nesterov=nesterov)
        self.state = {'step':0}



    def step(self, grad):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        lr = self.defaults['lr']
        momentum = self.defaults['momentum']
        dampening = self.defaults['dampening']
        nesterov = self.defaults['nesterov']

        d_p = grad
        if momentum != 0:
            if 'momentum_buffer' not in self.state:
                buf = d_p
            else:
                buf = self.state['momentum_buffer']
                buf = buf*momentum + (1-dampening)*d_p
            self.state['momentum_buffer'] = buf
            if nesterov:
                d_p = d_p + momentum*buf
            else:
                d_p = buf

        delta = lr * d_p
        self.state['step'] +=1

        return delta