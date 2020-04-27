# Give Me Some Credit on Redis

This repository has all the code to build a machine learning model with scikit-learn that solves the Give Me Some Credit challenge on Kaggle. Once it builds the model, it will convert it to an ONNX model which can be hosted on RedisAI.

## Step 1: Get Some Data

I can't host the data here as it belongs to others. You will need to download it yourself, unzip it, and place the .csv file in the data folder. You can download it here from the Kaggle [challenge page](https://www.kaggle.com/c/GiveMeSomeCredit/data).

**NOTE**: We will only use the `cs-training.csv` file and will not use the `cs-test.csv` file. So download only what you need.

## Step 2: Setup Python Environment

You need a Python environment to make this all work. I used Python 3.8—the latest, greatest, and most updatest at the time of this writing—but Python 3.7 should work as well. As always, I used `venv` to manage my environment.

I'll assume you already have Python 3.8. Go ahead and setup the environment:

    $ python3.8 -m venv venv

Once `venv` is installed, you need to activate it:

    $ . venv/bin/activate

Now when you run `python` from the command line, it will always point to Python3.8 and any libraries you install will only be for this specific environment. Usually, this includes a dated version of `pip` so go ahead an update that as well:

    $ pip install --upgrade pip

If you want to deactivate this environment, you can do so from anywhere with the following command:

    $ deactivate

## Step 3: Install Dependencies

This project uses ONNX and ONNX uses protobuf. So, before you install the Python libraries, you'll need protobuf installed. I'm on a make and I use homebrew so for me this looks like this:

    $ brew install protobuf

Now, you can install all the dependencies. These are all listed in `requirements.txt` and can be installed with `pip` like this.

    $ pip install -r requirements.txt


## Step 4: Build the Model

Run the Python script to build the model:

   $ python build.py

When completed, you will have a file in the model folder. This file can be loaded into RedisAI. It will also output several examples from the training data that you can use to try out the model once you have deployed it to RedisAI.

## Step 5: Deploy the Model to RedisAI

This is actually pretty easy assuming you have RedisAI configured with Redis already. Just enter the following command:

    $ redis-cli -x AI.MODELSET models:gmsc:linearsvc ONNX CPU < model/give-me-some-credit_linear-svc.onnx

## Step 6: Play Around

This script takes the entire test dataset and generates Redis commands to set the tensors, run the model, and read the output tensors. These are in `samples.txt` and also include the original result in the test data, and the value predicted by the model. Look in there and play about.
