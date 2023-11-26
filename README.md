# YOLO applied to STRIKE thermal profile

I am applying YOLO to a research problem for my master's thesis. Here is the abstract:

# Abstract

This thesis addresses the challenge of estimating the parameters of beamlets from thermal images in the STRIKE diagnostic calorimeter, a crucial component for the neutral beam source SPIDER. The latter is a clone part of the accelerator MITICA, which will serve the ITER tokamak as an additional injection power system. SPIDER produces up to 1280 beamlets of H-/D-, which are collected by STRIKE during 10-second shots. STRIKE is observed via thermal cameras, and the thermal pattern given by every single beamlet has been experimentally proven to be approximated as a 2D Gaussian curve. Traditional methods for fitting beamlets are insufficient due to their time-consuming nature, and previously developed rapid methods are not feasible for the new operation conditions of SPIDER. 

This work uses two machine learning techniques, applying both unsupervised and supervised learning for fast and efficient beamlet parameter estimation. An unsupervised Gaussian Mixture Model (GMM) and a supervised deep learning model, YOLO (You Only Look Once), were trained on synthetic images to detect and localize Gaussian approximations of the beamlets. The YOLO model, in particular, demonstrated superior performance, accurately identifying beamlets with tight bounding boxes even in cases of significant overlap. Refinement techniques for YOLO as PX modifier and Ensemble were explored but didn't yield better results. 

Challenges remain in correctly estimating the amplitude of overlapping Gaussians. Therefore, the thesis emphasizes the need for future work in disentangling overlapped Gaussians and extending model training to experimental STRIKE data. Besides, the methods developed in this thesis offer a promising approach for characterizing SPIDER beam profiles, YOLO for detecting and characterizing the beamlets with fast predictions of less than one second usually, and GMM as a support method to label future experimental data.
