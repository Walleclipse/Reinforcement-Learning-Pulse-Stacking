# Combining the power of reinforcement learning and stochastic parallel gradient decent for optimizing the delay lines of pulse stacking

## Experiments on controlling 7-stage delay line coherent pulse stacking (combining 128 pulse)

### Results on controlling coherent pulse stacking from the bad initial point 

In this scenario SPGDM (and SPGD) cannot find the maximum peak power, and controlled state is trapped into a saddle points (or bad local maximum). SAC-SPGDM and SAC could find the maximum and successfully control the system to obtain good combined pulses. The final power after 30 steps control for SAC-SPGDM and SAC is similar, but the control-convergence speed of SAC-SPGDM is slightly faster than SAC. 

<img src="SAC-SPGDM_BadInit.gif" width="400" height="300" alt="SAC-SPGDM from bad initial point"/><img src="SAC_BadInit.gif" width="400" height="300" alt="SAC from bad initial point"/>

<img src="SPGDM_BadInit.gif" width="400" height="300" alt="SPGDM from bad initial point"/><img src="SPGD_BadInit.gif" width="400" height="300" alt="SPGD from bad initial point"/>

### Results on controlling coherent pulse stacking from the good initial point 

In this scenario, SAC-SPGDM, SAC as well as SPGDM successfully control the system to obtain good combined pulses. This is consistent with the previous conclusion that, SPGD based controller is a good choice when the starting point is near a maximum. SAC-SPGDM still has the fastest convergence speed among them. It is worth noting that, as a training-free method, SPGDM achieved the maximum point with faster speed than SPGD.  

<img src="SAC-SPGDM_GoodInit.gif" width="400" height="300" alt="SAC-SPGDM from good initial point"/><img src="SAC_GoodInit.gif" width="400" height="300" alt="SAC from good initial point"/>

<img src="SPGDM_GoodInit.gif" width="400" height="300" alt="SPGDM from good initial point"/><img src="SPGD_GoodInit.gif" width="400" height="300" alt="SPGD from good initial point"/>
