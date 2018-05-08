# Intro

This is the code behind my submission for https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection.

Private leaderboard: 0.9731251.

# Setup

## Virtualenv

    virtualenv -p python3 venv
    source venv/bin/activate
    pip install -r requirements.txt

## XGBoost

    git clone --recursive https://github.com/dmlc/xgboost.git
    cd xgboost/
    make
    pip install -e python-package/

# Pipeline

Converts the datasets from CSV to Pandas. Handles the click time timezone.

    python3 convert.py

Splits the training dataset into smaller datasets. Allows a tradeoff between training speed and accuracy.

    python3 split.py

Add new features.

    python3 features.py

Train, predict and evaluate the AUROC score.

    python3 xgboost_model.py
