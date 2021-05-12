import numpy as np

class NormalizedAct(object):
    """ Wrap action """

    def __init__(self, action_space):
        self.high = action_space.high
        self.low = action_space.low
        self.k = (self.high - self.low) / 2.
        self.b = (self.high + self.low) / 2.

    def normalize(self, action,clip=False):
        act = (action - self.b)/self.k
        if clip:
            act = np.clip(act, -1, 1)
        return act

    def reverse_normalize(self, action,clip=False):
        act = self.k * action + self.b
        if clip:
            act = np.clip(act, self.low, self.high)
        return act

try:
    from gym.envs.classic_control.rendering import Geom
    import pyglet
except:
    Geom=object

class Image(Geom):
    def __init__(self, fname, width, height,fileobj=None):
        import pyglet
        Geom.__init__(self)
        self.set_color(1.0, 1.0, 1.0)
        self.width = width
        self.height = height
        img = pyglet.image.load(fname,file=fileobj)
        self.img = img
        self.flip = False
    def render1(self):
        self.img.blit(-self.width/2, -self.height/2, width=self.width, height=self.height)

