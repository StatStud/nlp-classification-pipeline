# nlp-classification-pipeline

![Alt Text](demo.jpg)

Train models, bayesian hyperparameter search, model tracking, and spreadsheet results.

This repo uses two scripts: main.py and train.py. 

## main.py

This script is for users who are looking to find the best performing models. To run the re-training, simply open the terminal and run:

python main.py --output_data_file var1 --model_name var2 --target_text var3

where `output_data_file` is the name of the CSV file that contains the spreadsheet of results, `model_name` is the name of the pre-trained language model, and `target_text` is the text you wish to train the model on. 

A successful re-training will result in a spreadsheet (that keeps track of all hyperparameters of the best models) and the best model predictions (saved as pickle files). 

## train.py

This script contains all the logic for re-training any given model. It is the training loop. 

**Note:** It is essential that the user supplies all three train, validation, and testing data within the data folder for the re-training to run successfully.

## Requirements

- Python 3.x
- Required packages listed in `requirements.txt`. Install them using `pip install -r requirements.txt`.
