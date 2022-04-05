# Setting Up the AWS Instance Environment

### The following information describes how to access the AWS instance used for the feature-encoding project, and how the environment was set up.

First log in to the instance using the SSH key:
```
ssh -i "neurohackademy-feature-encoding.pem" ubuntu@ec2-52-13-42-17.us-west-2.compute.amazonaws.com
```
To ensure the above code works, you will need to download the "neurohackademy-feature-encoding.pem" file and either run the code in the directory where that file is or specify the pathway to the file in the code.

Then the following code was used to set up the environment. For more details, please see [here](https://github.com/tridivb/slowfast_feature_extractor)
```
conda create --name attempt1
conda activate attempt1 
```
The following dependencies were then installed in the environment so that SlowFast can run

```
pip install 'git+https://github.com/facebookresearch/fvcore'
pip install simplejson av psutil opencv-python tensorboard moviepy cython
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
```
Then the PySlowFast was setup
```
git clone https://github.com/facebookresearch/slowfast
export PYTHONPATH=/path/to/slowfast:$PYTHONPATH
cd slowfast
python setup.py build develop
```

From here, the environment should be ready to start using PySlowFast!
