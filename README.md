# FIR-based Future Trajectory Prediction in Nighttime Autonomous Driving


<img width="1247" alt="fig1 2" src="https://user-images.githubusercontent.com/124192573/216125928-53b3b0e2-8f77-4750-8265-aac44f2c3abe.png">

<img width="1434" alt="fig2_new" src="https://user-images.githubusercontent.com/124192573/216126468-2e8e7a1f-7569-4cab-ae61-e8f67021f38b.png">

The performance of the current collision avoidance systems in Autonomous Vehicles (AV) and Advanced Driver Assistance Systems (ADAS) can be drastically affected by low light and adverse weather conditions. Collisions with large animals such as deer in low light cause significant cost and damage every year. In this project, we propose the first AI-based method for future trajectory prediction of large animals and mitigating the risk of collision with them in low light. 

In order to minimize false collision warnings, in our multi-step framework, first, the large animal is accurately detected and a preliminary risk level is predicted for it and low-risk animals are discarded. In the next stage a multi-stream CONV-LSTM-based encoder-decoder framework is designed to predict the future trajectory of the potentially high-risk animals. The proposed model uses camera motion prediction as well as the local and global context of the scene to generate accurate predictions. Furthermore, this project introduces a new dataset of FIR videos for large animal detection and risk estimation in real nighttime driving scenarios. Our experiments show promising results of the proposed framework in adverse conditions. 
This repository contains the main blocks of our code. 

### Paper: 


[FIR-based Future Trajectory Prediction in Nighttime Autonomous Driving](https://arxiv.org/pdf/2304.05345)

IEEE Intelligent Vehicles 2023 (IEEE IV 2023)

citation: 
```
@article{rahimpour2023fir,
  title={FIR-based Future Trajectory Prediction in Nighttime Autonomous Driving},
  author={Rahimpour, Alireza and Fallahinia, Navid and Upadhyay, Devesh and Miller, Justin},
  journal={arXiv preprint arXiv:2304.05345},
  year={2023}
}
```
=======
## Code: 
Please refer to this repo for the code and more details: 
{NightVision code and details}[https://github.com/AlirezaRahimpour/NightVision_trajectory_prediction/tree/main]

