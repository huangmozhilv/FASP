
# preprocess data before training/validating/testing
python survPred/data_prep/data_prep.py

# split all cases into folds for model development, validation, test
python survPred/data_prep/gen_data_splits_[XXX].py

python survPred/main.py