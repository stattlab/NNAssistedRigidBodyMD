# NNAssistedRigidBodyMD

**data sampling** : Contains hoomd scripts to simulate the systems of rigid bodies to sample pair configurations from. Also contains the scripts to process the raw data samples by considering relevant symmetries etc.     
**training** : Contains Pytorch scripts to train the two neural-nets that infer whether candidate pairs interact or not (selector, classification) and infer the interaction energy (regression).     
**nn_simulator** : Contains the C++ code that performs the Neural-Net assisted simulations. 
