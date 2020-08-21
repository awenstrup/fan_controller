# Objective
This project is intended primarily for learning. It will allow me to build a complete data pipeline, get exposure to AWS, and integrate the power of machine learning with a real, tangible application.

# Background
My family has a very fluffy dog, named Tito. He is a long-coated Akita, which means he is not well suited for 90+ degree summers in New York, where we live. He likes to spend his summers lying in front of a fan, or better yet, on top of an air conditioning vent. We usually keep a fan on in the kitchen to keep him comfortable while he is spending time with us. 

# Description
This project consists of the following steps. I aim to automate each step as much as possible, within reason. 
* Tak photos of the area in front of a box fan
* Upload those photos to an S3 bucket
* Manually label then (do they contain Tito?)
* Download the labeled dataset to my laptop
* Train a Convolutional Neural Network to recognize Tito
* Load the network onto a Raspberry Pi
* Wire the Raspberry Pi to a relay that controlls Tito's fan
* Only turn the fan on when Tito is present

## Other notes
I initially intended to use AWS SageMaker Autopilot to train my model, but I abandoned that idea for a few reasons. First, my training set was small enough that training the model on my laptop didn't take very long (on the order of 1-2 minutes). Second, to the extend that my primary goal was learning, I figured I would learn more by writing my own model in Tensorflow. 

Because I am no longer tied to AWS for my machine learning purposes, I am considering migrating to Storj, a decentralized cloud storage platform. Before I do that, I need to resolve a few hesitations:
* I've been following Storj for a while, but need to read a few more reviews
* I still use SageMaker ground truth for labeling, so I would need another solution for that

I wanted to get this project over with quickly, so I could move onto building my own ML library (which I figured provided more opportunities for learning), but I quickly got sucked in. I'm having a hard time moving on, and am constantly adding more to it.

Finally, on the topic of running my model remotely on my Raspberry Pi, I am considering migrating to Docker. Copying the model and source code, installing requirements, and running a script to the Raspberry Pi feels a bit messy; moving all of that to a Docker image feels a little cleaner. 

# Tools used
* AWS Simple Storage Solution (S3)
* TensorFlow
* Raspberry Pi


