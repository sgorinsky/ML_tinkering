This repo is for some experiments I'm running to test out various python ML packages.
Starting with Google stock data, I ran a linear regression to see how my model predicts stock prices. Unfortunately, it's an incomplete model because predicting stock price from prior stock price may not be too correlative. The model as is should be taken with a grain of salt. It's merely there to test scikit-learn's Linear Regression.

Additionally, there are other experiments in this repo as well like a KNN algorithm predicting breast cancer in patients from the Wisconsin Breast Cancer dataset contributed to UCI's ML repo. I made a scratch KNN alg to tet against scikit-learn's. It was as effective although much slower, so no need to use my own.

I will test more ML packages from Tensorflow, Keras, and scikit-learn in coming months when I have time.
