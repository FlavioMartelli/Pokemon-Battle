# Pokemon-Battle

This repo aims at creating a machine learning model that is capable to predict whether player 1 will win or lose a match in a gen2 matchup.
Given in inputs:
P1:
- list of pokemon with types, stats and item
- leading p2's pokemon with types and stats
- a summary of what happened during the first 30 turns

To train and run this model, first a tuning has to be ran `python tune.py` to find what the best hyperparameters are, and then it is possible to run `python train.py` while features.py does not need to be run since it will be automatically called by train.py (it is also possible to adjust how many iterations are wanted in tuning by specifying it, more details at the beginning of the file). The whole program checks 5 different models (logistic regression, random forest, extra trees, histGBM, XGB) and at the end they are both averaged and stacked to find what yields the best result. In this case, the best result was a stacked version excluding the random forest, but this can be easily adjusted in the training file by hand.
