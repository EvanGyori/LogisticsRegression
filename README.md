# Logistics Regression
The program takes in `bezdekIris.data` data and learns a model that outputs 3 seperate probabilities for each label given a feature vector.

More info about the dataset can be found at [Iris](https://archive.ics.uci.edu/dataset/53/iris)

## Specifics
The code is very messy but simply uses gradient acent of a log likelihood function covering all 3 probability functions.

## Results
The dataset is split into a training set and test set.

The model has an accuracy of 96% when used on the test set after 8000 epochs. After 8000 epochs, the probability outcomes become so small that float's lack of precision results in 0 causing divide by zero errors. 

## Usage
Make sure python3 is installed and run the following:
```
python LogisticsRegression.py
```
