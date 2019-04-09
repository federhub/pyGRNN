# pyGRNN
Python implementation of General Regression Neural Network (GRNN, also known as Nadaraya-Watson Estimator). A Feature Selection module based on GRNN is also provided.

## Getting Started with GRNN

GRNN is an adaptation in terms of neural network of the Nadaraya-Watson estimator, with which the general regression of a scalar on a vector independent variable is computed as a locally weighted average with a kernel as a weighting function. The main advantage of this algorithm is that its calibration only requires the definition of a proper bandwidth for the kernel estimation. Hence, GRNN is faster than other feedforward artificial neural network algorithms.

The traditional GRNN architecture is based on the use of one unique value of the bandwidth for all the features. This Isotropic structure of the network (IGRNN) can be used as a wrapper for feature selection. This approach permits a complete description of the input space, identifying relevant, irrelevant and redundant features. Specifically, redundancy and irrelevancy are associated to the identification of relatedness, i.e. the non-linear predictability of an input variable using the other features of the input space. 
Anisotropic (or Adaptive) GRNN (AGRNN) are an evolution of GRNN in which different values are given to the bandwidth corresponding to each feature. A proper calibration of the bandwidths will scale the input features depending on their explanatory power. Specifically, a large smoothing parameter will give rise to a lower discriminative power of the associated feature, and vice versa. Hence, AGRNN can be considered as an embedded feature selection method in which the bandwidth values of the kernel express a measure of the relevancy of the features.

For more insights on GRNN, check the [poster I presented at the EGU 2019 conference](https://github.com/federhub/pyGRNN/blob/master/EGU2019_FS_using_simple_and_efficient_ML_models.pdf). 

## Authors

* **Federico Amato** - *Postdoctoral Research Fellow,* - [GeoKDD lab, University of Lausanne](https://wp.unil.ch/geokdd/)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.
