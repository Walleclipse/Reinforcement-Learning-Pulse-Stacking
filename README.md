
# An Optical Control Environment for Benchmarking Reinforcement Learning Algorithms

This repository introduces the Optical Control Environment (OPS) designed for benchmarking reinforcement learning algorithms, as illustrated in the following papers:
 
Abulikemu Abuduweili, Jie Wang, Bowei Yang, Aimin Wang, and Zhigang Zhang, "[Reinforcement learning based robust control algorithms for coherent pulse stacking](https://opg.optica.org/oe/fulltext.cfm?uri=oe-29-16-26068&id=453824)," Opt. Express, 2021.    
Abulikemu Abuduweili, and Changliu Liu, "[An Optical Control Environment for Benchmarking Reinforcement Learning Algorithms](https://openreview.net/forum?id=61TKzU9B96)," TMLR, 2023.


## Introduction
Deep reinforcement learning has the potential to address various scientific problems. In this paper, we implement an optics simulation environment for reinforcement learning based controllers. The environment captures the essence of nonconvexity, nonlinearity, and time-dependent noise inherent in optical systems, offering a more realistic setting. 
Subsequently, we provide the benchmark results of several reinforcement learning algorithms on the proposed simulation environment. The experimental findings demonstrate the superiority of off-policy reinforcement learning approaches over traditional control algorithms in navigating the intricacies of complex optical control environments. 


## Tasks
In this paper, we present OPS (Optical Pulse Stacking), an open and scalable simulator designed for controlling typical optical systems. 
A visual illustration of the stacking procedure for combining two pulses is shown in the following gif.   
<img src="demo/Video1_SystemConfiguration_StackingTwoPulsesWithTimeDelay.gif" width="400" height="300" alt="System Configuration of Stacking TwoPulses With Time Delay Controller."/>

## Results
TD3 and SAC are capable of attaining maximum power in the OPS task by combining many pulses into one, as shown in the following gif.    
<img src="demo/Video2_Experiments_Controlling5StageOPS.gif" width="400" height="300" alt="Experiments of Controlling 5 Stage OPS (combining 128 pulses)."/>


## Citation
If you find the code helpful in your research or work, please cite the following papers.
```BibTex
@inproceedings{abuduweili2023an,
  title={An Optical Control Environment for  Benchmarking Reinforcement Learning Algorithms},
  author={Abuduweili, Abulikemu and Liu, Changliu},
  booktitle={Transactions on Machine Learning Research},
  year={2023},
url={https://openreview.net/forum?id=61TKzU9B96},
}    
@article{Abuduweili:21,
author = {Abulikemu Abuduweili and Jie Wang and Bowei Yang and Aimin Wang and Zhigang Zhang},
journal = {Opt. Express},
number = {16},
pages = {26068--26081},
title = {Reinforcement learning based robust control algorithms for coherent pulse stacking},
volume = {29},
month = {Aug},
year = {2021},
url = {https://opg.optica.org/oe/abstract.cfm?URI=oe-29-16-26068},
doi = {10.1364/OE.426906},
}
```



