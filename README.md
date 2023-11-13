# YOLO applied to STRIKE thermal profile

This is not a new YOLO version, but the application of YOLO to a research problem for my master thesis. Here is the abstract:

# Abstract

This thesis addresses the challenge of estimating the parameters of beamlets from thermal images in the STRIKE diagnostic calorimeter, a crucial component for the SPIDER operation, which serves the ITER tokamak—a major international nuclear fusion project. Traditional methods for fitting beamlets are insufficient due to their time-consuming nature, and previous rapid methods developed are not feasible for the new operation conditions of SPIDER. 

This work uses two machine learning techniques, applying both unsupervised and supervised learning for fast and efficient beamlet parameter estimation. An unsupervised Gaussian Mixture Model (GMM) and a supervised deep learning model, YOLO (You Only Look Once), were trained on synthetic images to detect and localize Gaussian approximations of the beamlets. The YOLO model, in particular, demonstrated superior performance, accurately identifying beamlets with tight bounding boxes even in cases of significant overlap. Refinement techniques for YOLO as PX modifier and Ensemble were explored but didn't yield better results. 

Challenges remain in correctly estimating the amplitude of overlapping Gaussians. Therefore, the thesis emphasizes the need for future work in disentangling overlapped Gaussians and extending model training to experimental STRIKE data. Besides, the methods developed in this thesis offer a promising approach for characterizing SPIDER beam profiles, YOLO for detecting and characterizing the beamlets with fast predictions of less than one second usually, and GMM as a support method to label future experimental data.
