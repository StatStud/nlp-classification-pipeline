from skopt import gp_minimize
from functools import partial

import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
import torch
from transformers import DistilBertForSequenceClassification, Trainer, TrainingArguments
from transformers import DistilBertTokenizerFast
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from sklearn.metrics import confusion_matrix
import time
from transformers import AutoConfig
import random
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize

from ast import literal_eval
import os
from scipy.special import softmax as softmax_funt
import datetime
from pathlib import Path
import argparse

train_data_file = os.path.join(os.path.abspath("./data"),"train.csv")
val_data_file = os.path.join(os.path.abspath("./data"),"val.csv")
test_data_file = os.path.join(os.path.abspath("./data"),"test.csv")

## asuming that the column containing the labels is called "labels'
## if not the case, variable label_column must be corrected
label_column = "labels"
data_str = "source_data"
BATCH_SIZE = 16
first_8k=False 
Augment=False
change_layers = False

class TextDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def open_search_data(train_data_file,val_data_file,
                     test_data_file,target_text):
    
    train_data = pd.read_csv(train_data_file)
    val_data = pd.read_csv(val_data_file)
    test_data = pd.read_csv(test_data_file)
    
    train_data[label_column] = train_data[label_column].apply(literal_eval)
    val_data.[label_column] = val_data[label_column].apply(literal_eval)
    test_data.[label_column] = test_data[label_column].apply(literal_eval)
    
    def get_usertags(lst):
        if 1 in lst:
            return 1
        else:
            return 0
    
    train_data[label_column] = train_data[label_column].apply(get_usertags)
    val_data[label_column] = val_data[label_column].apply(get_usertags)
    test_data[label_column = test_data[label_column].apply(get_usertags)
    
    train_data = train_data[[target_text,label_column]]
    val_data = val_data[[target_text,label_column]]
    test_data = test_data[[target_text,label_column]]
    
    return train_data,val_data,test_data

def stats(pred, true):
    with torch.no_grad():
        true, pred = torch.tensor(true), torch.tensor(pred)
        acc = torch.sum(true == pred).item()/len(true)
        prc, rec, fsc, _ = precision_recall_fscore_support(true.detach().cpu().numpy(),
                                                           pred.detach().cpu().numpy(), average='binary')
        pos = torch.sum(true == 1).item()
        neg = torch.sum(true == 0).item()
        return prc, rec, fsc, acc, pos, neg

def my_function(*args):
    #insert simple execute here, return NEGATIVE of fsc
    
    global train_data_file, val_data_file, test_data_file, tokenizer_name, model_name,data_str,BATCH_SIZE,target_text, first_8k, Augment,change_layers, output_data_name
    
    
    #save other model output for spreasheet
    global tn, fp, fn, tp, prc, rec, fsc, runtime, num_layers
            

    if change_layers:
        config = AutoConfig.from_pretrained(model_name)
        config.n_layers = change_layers 
        
        with torch.no_grad():
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            model = AutoModelForSequenceClassification.from_pretrained(model_name, config = config)
    else:
        with torch.no_grad():
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            model = AutoModelForSequenceClassification.from_pretrained(model_name)
          
    train_data, val_data, test_data = open_search_data(train_data_file,val_data_file,
                     test_data_file,target_text)

    train_encodings = tokenizer(train_data[target_text].tolist(), truncation=True, padding=True, max_length=512)
    val_encodings = tokenizer(val_data[target_text].tolist(), truncation=True, padding=True, max_length=512)
    test_encodings = tokenizer(test_data[target_text].tolist(), truncation=True, padding=True, max_length=512)

    train_dataset = TextDataset(train_encodings, train_data[label_column].tolist())
    val_dataset = TextDataset(val_encodings, val_data[label_column].tolist())
    test_dataset = TextDataset(test_encodings, test_data[label_column].tolist())


    t0 = time.time()
    training_args = TrainingArguments(
        output_dir='./base_training',          # output directory
        num_train_epochs= int(args[0][1]),              # total number of training epochs
        per_device_train_batch_size=BATCH_SIZE,  # batch size per device during training
        per_device_eval_batch_size=64,   # batch size for evaluation
        warmup_steps=500,                # number of warmup steps for learning rate scheduler
        weight_decay=0.01,               # strength of weight decay
        logging_dir='./base_training_logs',            # directory for storing logs
        logging_steps=10,
        evaluation_strategy = "epoch",
        learning_rate = float(args[0][0])
    )
    

    trainer = Trainer(
        model=model,                         # the instantiated 🤗 Transformers model to be trained
        args=training_args,                  # training arguments, defined above
        train_dataset=train_dataset,         # training dataset
        eval_dataset=val_dataset             # evaluation dataset
    )
    
    trainer.train()
    t1 = time.time()
    total = t1-t0
    
    runtime = round((total)/60,2)

    predictions = trainer.predict(test_dataset)
    preds = np.argmax(predictions.predictions, axis=-1)

    prc, rec, fsc, acc, pos, neg = stats(preds, test_data[label_column].tolist())
    tn, fp, fn, tp = confusion_matrix(test_data[label_column].tolist(), preds).ravel()
    
    try:
        num_layers = int(model.config.n_layers)
    except AttributeError:
        num_layers = int(model.config.num_hidden_layers) 
        
    
    return -fsc 
    
    
def print_result(params):
    
    #this is what we want to put on the spreadsheet
    #model_name, learning_rate, epochs, num_layers, data_str, 
    #text_str, tn, fp, fn, tp, prc, rec, fsc, round((total)/60,2), BATCH_SIZE
    # use global variables as needed
    
    lr_lst.append(params.x_iters[-1][0]) #learning rate
    epoch_lst.append(params.x_iters[-1][1]) #epochs
    f1_sc_lst.append(params.func_vals[-1])
    
    layer_lst.append(num_layers)
    tn_lst.append(tn)
    fp_lst.append(fp)
    fn_lst.append(fn)
    tp_lst.append(tp)
    precision_lst.append(prc)
    recall_lst.append(rec)
    time_lst.append(runtime)
    batch_size_lst.append(BATCH_SIZE)
    model_lst.append(model_name)
    data_lst.append(data_str)
    text_lst.append(text_str)
    print("learning rate:", params.x_iters[-1][0])
    print("epochs:", params.x_iters[-1][1])
    
    
def hyper_search(output_name,name,text):
    
    global train_data_file, val_data_file, test_data_file

#     output_data_name = "distilbert.csv"
#     model_name = "distilbert-base-uncased"
    global tokenizer_name,output_data_name,model_name,target_text
    tokenizer_name = name
    model_name = name
    output_data_name = output_name
    target_text = text
#     target_text = "Title"
    text_str = target_text


    ##########################################################

    model_lst = []
    lr_lst = []
    epoch_lst = []
    layer_lst = []
    data_lst = []
    text_lst = []

    tn_lst = []
    fp_lst = []
    fn_lst = []
    tp_lst = []
    precision_lst = []
    recall_lst = []
    f1_sc_lst = []

    batch_size_lst = []
    time_lst = []

    # what variables am I searching over?
    # IN ORDER: learning rate, epochs
    space = [(0.0000001, 0.001), 
             (3, 5)] 


    res = gp_minimize(my_function, space, n_calls=10, 
                      random_state=0, callback=print_result,
                     verbose = True)

    print("Best parameters: ", res.x, "with best score: ", -res.fun)
    

    final_df = pd.DataFrame({'model': model_lst,
                            'learning_rate': lr_lst,
                            'epochs': epoch_lst,
                            'layers': layer_lst,
                            'dataset': data_lst,
                            'text': text_lst,
                            'tn': tn_lst,
                            'fp': fp_lst,
                            'fn': fn_lst,
                            'tp': tp_lst,
                            'precision': precision_lst,
                            'recall': recall_lst,
                            'f1_score': f1_sc_lst,
                            'train_batch_size': batch_size_lst,
                            'train_time_min': time_lst})

    #sort data and reverse negative of f1_score
    final_df['f1_score'] = final_df['f1_score']*-1
    final_df = final_df.sort_values(['f1_score'],ascending=False)

    ## add timestamp
    ct = str(datetime.datetime.now())
    final_df['timestamp'] = [i for i in [ct] for _ in range(len(final_df))]

    # if the training file already exists; then append
    # otherwise, create a brand new csv file
    path = Path(f'./{output_file_name}')
    if path.is_file():
        final_df.to_csv(output_data_name, mode='a')
    else:
        final_df.to_csv(output_data_name)

        
def save_best_model(output_data_name, model_name, target_text):
    df = pd.read_csv(output_data_name,nrows=1)
    
    model_name = df[['model']].values.tolist()[0][0]
    learning_rate = df[['learning_rate']].values.tolist()[0][0]
    epochs = df[['epochs']].values.tolist()[0][0]
    layers = df[['layers']].values.tolist()[0][0]
    BATCH_SIZE = df[['train_batch_size']].values.tolist()[0][0]
    target_text = df[['text']].values.tolist()[0][0]
    tokenizer_name = model_name
    
    
    global train_data_file, val_data_file, test_data_file

    config = AutoConfig.from_pretrained(model_name) 
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if config.n_layers != layers:
        config.n_layers = layers
        model = AutoModelForSequenceClassification.from_pretrained(model_name, config = config)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        
        
    train_data, val_data, test_data = open_search_data(train_data_file,val_data_file,
                     test_data_file,target_text)
    

    train_encodings = tokenizer(train_data[target_text].tolist(), truncation=True, padding=True, max_length=512)
    val_encodings = tokenizer(val_data[target_text].tolist(), truncation=True, padding=True, max_length=512)
    test_encodings = tokenizer(test_data[target_text].tolist(), truncation=True, padding=True, max_length=512)

    train_dataset = TextDataset(train_encodings, train_data[label_column].tolist())
    val_dataset = TextDataset(val_encodings, val_data[label_column].tolist())
    test_dataset = TextDataset(test_encodings, test_data[label_column].tolist())


    training_args = TrainingArguments(
        output_dir='./base_training',          # output directory
        num_train_epochs= epochs,              # total number of training epochs
        per_device_train_batch_size=BATCH_SIZE,  # batch size per device during training
        per_device_eval_batch_size=64,   # batch size for evaluation
        warmup_steps=500,                # number of warmup steps for learning rate scheduler
        weight_decay=0.01,               # strength of weight decay
        logging_dir='./base_training_logs',            # directory for storing logs
        logging_steps=10,
        evaluation_strategy = "epoch",
        learning_rate = learning_rate
    )
    

    trainer = Trainer(
        model=model,                         # the instantiated 🤗 Transformers model to be trained
        args=training_args,                  # training arguments, defined above
        train_dataset=train_dataset,         # training dataset
        eval_dataset=val_dataset             # evaluation dataset
    )
    
    trainer.train()
    trainer.save_model(f"models/{output_data_name[:-4]}")
    
    
    
def run_inference(output_data_name,model_name, target_text):
    global train_data_file, val_data_file, test_data_file
    
    output_dir = f"stored_arrays/{output_data_name[:-4]}"
    saved_model_path = f"models/{output_data_name[:-4]}"
    
    elements_to_run = ['prob_scores_norm_pos', 'pred_labels']
    threshold = 0.75
    
    try:
        os.makedirs(output_dir)
    except FileExistsError:
        pass
    
    ######### manage data ###########
    train_data, val_data, test_data = open_search_data(train_data_file,val_data_file,
                     test_data_file,target_text)
    
    
    tokenizer = model_name
    train_encodings = tokenizer(train_data[target_text].tolist(), truncation=True, padding=True, max_length=512)
    val_encodings = tokenizer(val_data[target_text].tolist(), truncation=True, padding=True, max_length=512)
    test_encodings = tokenizer(test_data[target_text].tolist(), truncation=True, padding=True, max_length=512)

    train_dataset = TextDataset(train_encodings, train_data[label_column].tolist())
    val_dataset = TextDataset(val_encodings, val_data[label_column].tolist())
    test_dataset = TextDataset(test_encodings, test_data[label_column].tolist())
    
    ########################################################################
    
    
    model = AutoModelForSequenceClassification.from_pretrained(saved_model_path)
    trainer = Trainer(model)
    
    ########################################################################
    
    train_probs = trainer.predict(train_dataset).predictions
    val_probs = trainer.predict(val_dataset).predictions
    test_probs = trainer.predict(test_dataset).predictions
    
    ########################################################################
    
    train_probs = softmax_funt(train_probs, axis=1)[:,1]
    val_probs = softmax_funt(val_probs, axis=1)[:,1]
    test_probs = softmax_funt(test_probs, axis=1)[:,1]
    
    ########################################################################
    
    with open(f'{output_dir}/train.npy', 'wb') as f:
        np.save(f,train_probs)
        
    with open(f'{output_dir}/val.npy', 'wb') as f:
        np.save(f,val_probs)
        
    with open(f'{output_dir}/test.npy', 'wb') as f:
        np.save(f,test_probs)
        
    ################## get labels ###########################################
    
    train_probs = np.asarray([1 if x > threshold else 0 for x in train_probs])
    val_probs = np.asarray([1 if x > threshold else 0 for x in val_probs])
    test_probs = np.asarray([1 if x > threshold else 0 for x in test_probs])
        
    with open(f'{output_dir}/train_labels.npy', 'wb') as f:
        np.save(f,train_probs)
        
    with open(f'{output_dir}/val_labels.npy', 'wb') as f:
        np.save(f,val_probs)
        
    with open(f'{output_dir}/test_labels.npy', 'wb') as f:
        np.save(f,test_probs)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Say hello')
    parser.add_argument('--output_data_name', help="name of csv file to save results")
    parser.add_argument('--model_name', help="Huggingface Model Name")
    parser.add_argument('--target_text', help="Text field used for training")
    args = parser.parse_args()

    print("COMMENCING HYPERPARAMETER SERACH")
    hyper_search(args.output_data_name,args.model_name,args.target_text)
    print("HYPERPARAMETER SEARCH COMPLETE. SAVING BEST MODEL")
    save_best_model(args.output_data_name)
    print("RUNNIG MODEL INFERENCE")
    run_inference(args.output_data_name,args.model_name,args.target_text)
