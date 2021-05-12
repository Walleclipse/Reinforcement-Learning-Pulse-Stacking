
from scipy import constants
import numpy as np
from functools import partial
import pynlo
import math

polarization_mark={'p':'^ (up)','-p':'v (down)','s':'-> (in)','-s':'<- (out)'}
_C_mmps = constants.value('speed of light in vacuum') * 1e3 / 1e12 # light speed of  mm/ps

def copy_pulse(pulse):
	new_pulse = pynlo.light.PulseBase.Pulse()
	new_pulse.set_NPTS(pulse.NPTS)
	new_pulse.set_time_window_ps(pulse.time_window_ps)
	new_pulse.set_center_wavelength_nm(pulse.center_wavelength_nm)
	new_pulse._frep_MHz = pulse.frep_MHz
	new_pulse.set_AW(pulse.AW)
	return new_pulse

class PhotoDetector:
	def intensity(self,num):
		return abs(num) ** 2
	def dB(self,num):
		return 10 * np.log10(num)
	def f2_amp(self,num):
		return num * num.conjugate()
	def max_num(self,num):
		return np.max(np.abs(num))
	def integrate_num(self,num):
		return np.trapz(num)
	def avg_power(self,num):
		return self.integrate_num(self.intensity(num))
	def avg_f2_power(self,num):
		return self.integrate_num(self.intensity(self.f2_amp(num)))

def pulse_power(pulse):
	return 1e9 * pulse.dT_mks * PhotoDetector().avg_power(pulse.AT)

def pulse_f2_power(pulse):
	return 1e9 * pulse.dT_mks * PhotoDetector().avg_f2_power(pulse.AT)

def combine_trains(trains):
	pulse_s=None
	pulse_p=None
	s_aw = 0
	p_aw=0
	for pulse_info in trains:
		pu_dict = pulse_info['pulse']
		pu_s = pu_dict['s']
		pu_p = pu_dict['p']
		if pu_s is not None:
			s_aw += pu_s.AW
			if pulse_s is None:
				pulse_s = copy_pulse(pu_s)
		if pu_p is not None:
			p_aw += pu_p.AW
			if pulse_p is None:
				pulse_p = copy_pulse(pu_p)

	if pulse_s is not None:
		pulse_s.set_AW(s_aw)

	if pulse_p is not None:
		pulse_p.set_AW(p_aw)

	pulse_dict = {'p': pulse_p, 's': pulse_s}
	return pulse_dict

def detect(pulse_dict):
	pu_s = pulse_dict['s']
	pu_p = pulse_dict['p']
	I=0.
	pulse=None
	T_ps=0
	if pu_s is not None:
		I_s = PhotoDetector().intensity(pu_s.AT)
		I += I_s
		pulse= pu_s
		T_ps = pu_s.T_ps
	if pu_p is not None:
		I_p = PhotoDetector().intensity(pu_p.AT)
		I += I_p
		pulse = pu_p
		T_ps = pu_p.T_ps

	f2_I = PhotoDetector().f2_amp(I)
	f2_power = 1e9 * pulse.dT_mks * PhotoDetector().integrate_num(f2_I)
	ret_pulse={'I':I,'T_ps':T_ps}
	return ret_pulse,  f2_power


def init_PZM_l0(stage=3,frep_MHZ=1000):
	period = 1e6/frep_MHZ # ps
	l0_list=[0]
	for ii in range(1,stage+1):
		delay = 2**(ii-1)*period/2
		l = delay * _C_mmps
		l0_list.append(l)
	return l0_list

class PhaseModulator:
	def __init__(self,stage=3):
		self.stage=stage
		self.num_pulse = 2**stage

	def _phase_preset(self,out_p):
		if out_p == 'p':
			return 'p', 's'
		if out_p == '-p':
			return '-p', '-s'
		if out_p == 's':
			return '-p', 's'
		if out_p == '-s':
			return 'p', '-s'

	def _cal_polar(self,stage=None):
		if stage is None:
			stage = self.stage
		phases = ['-s']
		for ii in range(stage):
			temp = []
			for ph in phases:
				a, b = self._phase_preset(ph)
				temp += [a, b]
			phases = temp.copy()
		return phases

	def infer(self,pulse):
		frep = pulse._frep_MHz
		period = 1e6/frep # ps
		new_frep = frep/self.num_pulse
		phases = self._cal_polar()
		p_s_delay = 0.5 * period
		pulse_train=[]
		for ind in range(self.num_pulse):
			new_p = copy_pulse(pulse)
			new_phase = phases[ind]
			if ind%2==0:
				new_delay = (ind*period/2)
			else:
				new_delay = ((ind-1)*period/2 + p_s_delay)
			new_p.set_frep_MHz(new_frep)
			new_p.add_time_offset(new_delay)
			if '-' in new_phase:
				new_p.apply_phase_W(np.pi)
			pulse_dict={'p':None,'s':None}
			pulse_dict[new_phase[-1]] = new_p
			pulse_info = {'pulse':pulse_dict,'phase':new_phase,'name':'orig_'+str(ind)}
			pulse_train.append(pulse_info)
		return pulse_train

class StackStage:
	def __init__(self,PZM_fold=1, PZM_l0=0, optim_l0=0, noise_sigma=1.,np_rnd=None,name='s1'):
		self.fold = PZM_fold
		self.l0 = PZM_l0
		self.noise_sigma = noise_sigma
		self.optim_l0 = optim_l0

		if np_rnd is not None:
			self.np_random = np_rnd
		else:
			self.np_random = np.random.RandomState()
		self.noise = partial(self.np_random.normal,loc=0,scale=noise_sigma,size=1)

		self.name = name
		self.L = self.l0

		QWP_angle=45
		self.qwp_cos = math.cos(np.pi*QWP_angle/180)
		self.qwp_sin = math.sin(np.pi*QWP_angle/180)

	def _phase_postset(self,p1,p2):
		inp_ps=set([p1,p2])
		if inp_ps==set(['s','p']):
			return 'p'
		if inp_ps==set(['-s','-p']):
			return '-p'
		if inp_ps == set(['s', '-p']):
			return 's'
		if inp_ps == set(['-s', 'p']):
			return '-s'

	def cal_displacement(self):
		return self.fold*self.L

	def clip_pzm(self,low=-np.inf,high=np.inf):
		L = self.L
		L = np.clip(L, self.optim_l0+low, self.optim_l0+high)
		self.L=L


	def feedback(self,delta_l):
		self.L += delta_l

	def free_run(self,count=1):
		for ii in range(count):
			raw_noise = np.clip(self.noise(),-3*self.noise_sigma,3*self.noise_sigma)[0]
			self.L += raw_noise

	def infer(self,pulse_train):
		d = self.cal_displacement()
		offset_ps = - d / _C_mmps
		new_pulse_train = []
		n = len(pulse_train)
		for ind in range(0,n,2):
			pulse_info1,pulse_info2 = pulse_train[ind],pulse_train[ind+1]
			pu1_dict = pulse_info1['pulse']; ph1 = pulse_info1['phase'];  na1 =pulse_info1['name']
			pu2_dict = pulse_info2['pulse']; ph2 = pulse_info2['phase'];  na2 = pulse_info2['name']

			pu1_s = pu1_dict['s']
			pu2_s = pu2_dict['s']
			new_aw= 0
			pu_s=None
			if pu1_s is not None:
				pu1_s = copy_pulse(pu1_s)
				pu1_s.add_time_offset(offset_ps)
				new_aw += pu1_s.AW
				pu_s = pu1_s
			if pu2_s is not None:
				pu2_s = copy_pulse(pu2_s)
				pu2_s.add_time_offset(offset_ps)
				new_aw += pu2_s.AW
				if pu_s is None:
					pu_s = pu2_s
			pu_s.set_AW(new_aw)

			pu1_p = pu1_dict['p']
			pu2_p = pu2_dict['p']
			new_aw= 0
			pu_p=None
			if pu1_p is not None:
				new_aw += pu1_p.AW
				pu_p = pu1_p
			if pu2_p is not None:
				new_aw += pu2_p.AW
				if pu_p is None:
					pu_p = pu2_p
			pu_p.set_AW(new_aw)

			new_p_pulse = copy_pulse(pu_s)
			new_p_pulse.set_AW(self.qwp_cos*(pu_p.AW + pu_s.AW))

			new_s_pulse = copy_pulse(pu_p)
			new_s_pulse.set_AW(self.qwp_sin*(pu_p.AW - pu_s.AW))
			new_s_pulse.apply_phase_W(np.pi)

			new_phase = self._phase_postset(ph1,ph2)
			new_name = self.name + '_' + (na1.split('_')[-1]+ '&' +na2.split('_')[-1])
			pulse_dict={'p':new_p_pulse,'s':new_s_pulse}
			pulse_info = {'pulse':pulse_dict,'phase':new_phase,'name':new_name}
			new_pulse_train.append(pulse_info)

		return new_pulse_train

class MultiStack:
	def __init__(self,stacks_list,free_run_time=1):
		self.stacks_list = stacks_list
		self.stage = len(stacks_list)
		self.free_run_time = free_run_time

	def pzm(self):
		pzm_lens=[]
		for ii in range(self.stage):
			pzm_lens.append(self.stacks_list[ii].L)
		return pzm_lens

	def delta_pzm(self):
		pzm_lens=[]
		for ii in range(self.stage):
			d = self.stacks_list[ii].L - self.stacks_list[ii].optim_l0
			pzm_lens.append(d)
		return pzm_lens

	def clip_pzm(self,low=-np.inf,high=np.inf):
		for ii in range(self.stage):
			self.stacks_list[ii].clip_pzm(low,high)

	def feedback(self,delta_l_list):
		for ii in range(self.stage):
			self.stacks_list[ii].feedback(delta_l_list[ii])

	def free_run(self,count=None):
		if count is None:
			count = self.free_run_time
		for ii in range(self.stage):
			self.stacks_list[ii].free_run(count)

	def infer(self, trains):
		trains_hist=[trains]
		for ii,stagestack in enumerate(self.stacks_list):
			trains = stagestack.infer(trains)
			trains_hist.append(trains)

		return trains_hist




