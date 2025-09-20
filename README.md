# Physics-Informed-NN
A physics-informed machine learning model is developed that can replace the numerical simulations of porous media.
By learning the communications among grid cells in the numerical domain, this model is capable of accurately predicting flow fields for new sets of simulation runs.
Because of the many possible random arrangements of particles and their orientation with respect to each other, generalization of permeability with high accuracy is not trivial - nor is it practical using conventional means. 
Furthermore, building a comprehensive database for different grain/pore arrangements is not possible because of the cost of running numerical simulations to generate the database that represent all possible arrangements. 
The objective is to predict grid-level flow fields in porous media as a priori to determining permeability of porous media.
The rationale is that once the detailed grid-level dynamics can be accurately predicted using data-driven approach, for any configuration/topology of the porous media, the detailed dynamics could be predicted without any need for new expensive numerical simulation runs.
In this work, a physics-informed deep learning model is developed by using the results from Lattice-Boltzmann simulations of randomly distributed circular grains to represent the porous media. A variety of porous structures are developed by changing the number, size, and location of circular grains. The deep U-Net and ResNet neural network architectures are combined to train a deep learning model which avoids the vanishing gradient issues. 
The continuity and momentum conservation equations are embedded into the loss function of deep learning architecture.
Robustness of the developed model is then tested for numerous variations of porous media which have not been used for developing the model.


# Files
- `1st-ten-epochs-only.py` outputs the **first 10 epoch** predictions for validation in the form of **two numpy arrays**. This script is indended for use on machines with limited computational power. Does not use CUDA

- `main-training-script-no-cuda.py` is the main training/validation script designed to **run only using CPU power**. Each epoch uses pytorch visualizations for predicted and actual. Does not use CUDA

- `npy-arr-only-training-script.py` outputs the model's predictions in the form of **two numpy arrays**. There is no cap to the number of epochs that the model will output. Does not use CUDA.

- `model1-multi.py` contains the deep learning functions used during training. Does not use CUDA at the moment.

- `test.py` script designed to perform testing and evaluation of model. Does not use CUDA at the moment.

# Data:
The data required to run the training and testing can be found here: <br />
Training:
Velocity in x-direction: <br />
https://u.pcloud.link/publink/show?code=XZvHaxVZoDSpyNXTHjpM5aLuPhDfKSut8iqy

Velocity in y-direction: <br />
https://u.pcloud.link/publink/show?code=XZVzaxVZoT4JYici0K0EI9dv3GRKbBYvNR4k

Circular packs: <br />
https://u.pcloud.link/publink/show?code=XZnzaxVZGMdjYktgEuHCCUy7zmP9uRiyatfk
