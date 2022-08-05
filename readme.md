# Overview
This is a keras neural network that tokenizes emails and classifies whether it thinks that they are spam or not.  So far the best accuracy I have gotten is 78%.

# Algorithm
The training data is read ny the program.
It is then converted into numbers, so the neural network can take it as input.  The way it does this is by assigning each word a number, and then converting each word in the email into one of those words.
The network is then trained with that data by using model.fit(), leaving out a few emails to use for benchmarking.

It is benchmarked with unused emails from the dataset.

We can test it on singular emails with model.predict().

For more information, I used the spamassassin dataset.

# Files

## train.py
Trains the network on the dataset (you don't have to run this since it's pretrained)
## bench.py 
Benchmarks the network (you also don't have to run this since I benched it, but you can if you want to know the accuracy)
## main.py
Classifies whether it thinks the email in "email.txt" is spam or not.
## everything.py
Is essentially train and bench combined, except it does everything worse so don't use it
