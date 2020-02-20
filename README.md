# CSC249/449 HW3: DCF tracker

# Prerequisite knowledge: Pytorch 

[Pytorch](https://pytorch.org/) is a software that help us easily train and test network with automatic gradient caculation mechanism. Since this assignment is based on pytorch, I highly recommend you to get familiar with pytorch by finishing [DEEP LEARNING WITH PYTORCH: A 60 MINUTE BLITZ](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html). And you should build up your google cloud pytorch environment and do your homework inside that google cloud. And the pytorch version should be greater than 0.4.0.

We provide a [google cloud tutorial](https://github.com/rochesterxugroup/google_cloud_tutorial).



# Part1. Warm up: train an classifier (15pt)

We will train a classifier on Cifar10 dataset. Please refer to [Pytorch Classification Tutorial](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html) to get an idea about how pytorch is used to train neural network. Furthermore, we will use the model trained on Cifar10 as the feature extractor for the DCF tracker later. 

You are required to implement the `train` function in `classifier.py`. The model should be run in GPU if you have access. Also, you are free to modify any other part of the code in `classifier.py` other than the `Net` class and the model saving code.

When you finished training, make sure the test accuracy over the whole test is higher than 60%.

Finally, the parameter of the model is saved as `classifier_param.pth`, which will be used in the DCF tracker later.

# Part2. Implement the DCF tracker (40pt)

The process of how DCF is used for visual tracking is given in  [DCFtracker tutorial](https://github.com/jshi31/csc249tracking/blob/master/DCFtracker.pdf). Now you are required to implement the DCF tracker based on our provided code. 

## Requirements

`pytoch version >= 0.4.0`

`opencv-python`

## Prepare Dataset

We will use OTB2013 dataset. To get the dataset,

```bash
cd csc249tracking/dataset
python gen_otb2013.py
cd OTB2015
./download.sh 
python unzip.py
cd ..
ln -s $ABSOLUTE_PATH_to_OTB2015 OTB2013
```

## Test

You will need to implement the `update` and `forward` method in `DCFNet` class in `network.py`. 

After you finished the code, just run

```bash
python DCFtracker.py --model classifier_param.pth
```

### Test Result

| Random init Param | Loading classifier param | accumulate w | accumulate x | result (AUC)  |
| :---------------: | :----------------------: | :----------: | :----------: | :-----------: |
|         ✓         |            ✕             |      ✓       |      ✓       | 0.4652±0.0209 |
|         ✕         |            ✓             |      ✕       |      ✕       |    0.2219     |
|         ✕         |            ✓             |      ✓       |      ✕       |    0.4424     |
|         ✕         |            ✓             |      ✕       |      ✓       |    0.5045     |
|         ✕         |            ✓             |      ✓       |      ✓       |    0.5259     |

This test result is based on TA's code, which can be a reference of your implementation.

- Random init param: The feature extractor for DCF tracker is random initialized. The old version of the code has bug in parameter loading, so that will be the case of random parameter.
- Loading classifier param: The feature extractor is initialized with the classifier parameter.
- Accumulate w: update w as ![\hat w=(1-lr)\times \hat w + lr\times w](http://latex.codecogs.com/gif.latex?%5Chat%20w%3D%281-lr%29%5Ctimes%20%5Chat%20w%20&plus;%20lr%5Ctimes%20w)
- Accumulate x: update x as ![\hat \phi(x)=(1-lr)\times \hat \phi(x) + lr\times \phi(x)](http://latex.codecogs.com/gif.latex?%5Chat%20%5Cphi%28x%29%3D%281-lr%29%5Ctimes%20%5Chat%20%5Cphi%28x%29%20&plus;%20lr%5Ctimes%20%5Cphi%28x%29)

### Visualization

You can visualize the tracking result by running 

```bash
python DCFtracker.py --model classifier_param.pth --visualization
```

### Compare with standard model

You can check the correctness of `network.py` by running

```shell
python DCFtracker.py
```

it will use the standard tracking model stored in `param.pth`, and the test result (AUC) should be greater than 0.6

# Write Up (5pt)

You need to summerize this assignment in a README.txt or README.pdf. 

You are required to write

- pytorch version.
- The training argument of the classifier, including batch size, learning rate, optimization method, epoch number.
- The classification testing accuracy, both overall and class-wise.
- The tracking testing result (AUC)

# Submission Process 

You need to upload the following files:

- classifier.py
- network.py

- classifier_param.pth
- README.txt or README.pdf
- HW3.pdf (including P1, P2, P4)

Put all the above files in a single folder named `YourNetID`, where YourNetID is replaced by your netid in lower case. Zip such folder with the name `YourNetID.zip` and submit. 

