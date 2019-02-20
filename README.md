# Prerequisite knowledge: Pytorch 

[Pytorch](https://pytorch.org/) is a software that help us easily train and test network with automatic gradient caculation mechanism. Since this assignment is based on pytorch, I highly recommend you to get familiar with pytorch by finishing [DEEP LEARNING WITH PYTORCH: A 60 MINUTE BLITZ](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html). And you should build up your google cloud pytorch environment and do your homework inside that google cloud. And the pytorch version should be greater than 0.4.0.

We provide a [google cloud tutorial](https://github.com/rochesterxugroup/google_cloud_tutorial).



# Part1. Warm up: train an classifier (30pt)

We will train a classifier on Cifar10 dataset. Please refer to [Pytorch Classification Tutorial](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html) to get an idea about how pytorch is used to train neural network. Furthermore, we will use the model trained on Cifar10 as the feature extractor for the DCF tracker later. 

You are required to implement the `train` function in `classifier.py`. The model should be run in GPU if you have access. Also, you are free to modify any other part of the code in `classifier.py` other than the `Net` class and the model saving code.

When you finished training, make sure the test accuracy over the whole test is higher than 60%.

Finally, the parameter of the model is saved as `classifier_param.pth`, which will be used in the DCF tracker later.

# Part2. Implement the DCF tracker (40pt)

The process of how DCF is used for visual tracking is given in the tutorial. Now you are required to implement the DCF tracker based on our provided code. 

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
ln -s absolute_path_to_OTB2015 OTB2013
```

## Test

You will need to implement the `update` and `forward` method in `DCFNet` class in `network.py`. 

After you finished the code, just run

```bash
python DCFtracker.py --model classifier_param.pth
```

And the test result (AUC) should be greater than 0.4.

### Visualization

You can visualize the tracking result by running 

```bash
python DCFtracker.py --model classifier_param.pth --visualization
```

and you can see the tracked object in `csc249tracking/visualization`

### Compare with standard model

You can check the correctness of `network.py` by running

```
python DCFtracker.py
```

it will use the standard tracking model stored in `param.pth`, and the test result (AUC) should be greater than 0.6

# Write Up (30pt)

You need to summerize this assignment in a README.txt or README.pdf. 

You are required to write

- pytorch version.
- The training argument of the classifier, including batch size, learning rate, optimization method, epoch number.
- The classification testing accuracy, both overall and class-wise.
- The tracking testing result (AUC)
- Description of how the tracker work according to your understanding of the code.

# Extra Credit (20pt)

We have two theoretical questions in the tutorial. That will be 10pt for each. Write your solution to these two questions in pdf and name it as `theoreticalHW.pdf`

# Hand In 

You need to upload the following files:

- classifier.py
- network.py

- classifier_param.pth
- README.txt or README.pdf
- theoreticalHW.pdf (optional)

Put all the above files in a single folder named `YourNetID`, where YourNetID is replaced by your netid in lower case. Zip such folder with the name `YourNetID.zip` and submit. 

