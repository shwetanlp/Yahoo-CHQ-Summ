import json

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import RobertaTokenizer, ProphetNetTokenizer, T5Tokenizer, BartTokenizer, PegasusTokenizer
from transformers import ProphetNetForConditionalGeneration, T5ForConditionalGeneration, BartForConditionalGeneration, PegasusForConditionalGeneration
from torch.distributions import Categorical
import math
import glob
import shutil
from statistics import mean
import os
import wandb
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm
import shutil
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from datasets import load_metric

from data import CustomDataset, read_langs
from eval import get_rouge


from transformers.optimization import (

    get_constant_schedule,
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup,
)

arg_to_scheduler = {
    "linear": get_linear_schedule_with_warmup,
    "cosine": get_cosine_schedule_with_warmup,
    "cosine_w_restarts": get_cosine_with_hard_restarts_schedule_with_warmup,
    "polynomial": get_polynomial_decay_schedule_with_warmup,
    "constant": get_constant_schedule,
    "constant_w_warmup": get_constant_schedule_with_warmup,
}

from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'

# tokenzier for encoding the text

config = wandb.config  # Initialize config

def validate(model, tokenizer, device, loader, max_generated_length=15, beam_size=4):
    model.eval()
    predictions = []
    with torch.no_grad():
        for _, data in tqdm(enumerate(loader, 0)):
            y = data['target_ids'].to(device, dtype= torch.long)
            ids = data['source_ids'].to(device, dtype= torch.long)
            mask = data['source_mask'].to(device, dtype= torch.long)

            generated_ids = model.generate(
                input_ids=ids,
                attention_mask=mask,
                max_length=max_generated_length,
                num_beams=beam_size,
                repetition_penalty=2.5,
                length_penalty=1.0,
                early_stopping=True
                )

            preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]
            predictions.extend(preds)

    return predictions


def main(args):

    config.VALID_BATCH_SIZE = 16
    config.SEED = 42  # random seed (default: 42)
    config.MAX_LEN = args.MAX_LEN
    config.SUMMARY_LEN = args.SUMMARY_LEN
    config.DATASET_PATH = args.DATASET_PATH
    config.OUTPUT_PATH = args.OUTPUT_PATH
    if args.MODE=='val':
        config.SOURCE_TEST = config.DATASET_PATH + '/val.source'
        config.TARGET_TEST = config.DATASET_PATH + '/val.target'
    elif args.MODE=='test':
        config.SOURCE_TEST = config.DATASET_PATH+'/test.source'
        config.TARGET_TEST = config.DATASET_PATH+'/test.target'

    if not os.path.exists(config.TARGET_TEST):
        print("Warning: Using source as target...")
        config.TARGET_TEST=config.SOURCE_TEST   #### just to run the script for all the dataset setup

    file_test = (config.SOURCE_TEST, config.TARGET_TEST)

    output_path_dir = config.OUTPUT_PATH
    if not os.path.exists(output_path_dir):
        os.makedirs(output_path_dir, exist_ok=True)

    best_model_path=None
    best_rl = 0.0
    not_required_checkpoints=[]
    print(f"Model name: {args.MODEL_NAME}")


    if args.MODEL_NAME =='prophetnet':
        tokenizer = ProphetNetTokenizer.from_pretrained("microsoft/prophetnet-large-uncased")
        model_skeleton = ProphetNetForConditionalGeneration
    elif args.MODEL_NAME == 't5':
        tokenizer = T5Tokenizer.from_pretrained("t5-large")
        model_skeleton = T5ForConditionalGeneration
    elif args.MODEL_NAME == 'bart':
        tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")
        model_skeleton = BartForConditionalGeneration

    elif args.MODEL_NAME == 'pegasus':
        tokenizer = PegasusTokenizer.from_pretrained("google/pegasus-large")
        model_skeleton = PegasusForConditionalGeneration
    else:
        print(f"Model name: {args.MODEL_NAME}")
        print("Model name invalid!!")
        exit(-1)

    bertscore_metric = load_metric("bertscore")

    best_beam_info = {}
    for name in glob.glob(args.MODEL_PATH+'/checkpoint-*/'):
        if os.path.isdir(name):
            modelpath= os.path.join(args.MODEL_PATH, name)
            print(modelpath)
            model = model_skeleton.from_pretrained(modelpath)
            model = model.to(device)

            test_dataset, max_src_test, max_tgt_test = read_langs(file_test)

            test_set = CustomDataset(test_dataset, tokenizer, config.MAX_LEN, config.SUMMARY_LEN)

            test_params = {
                'batch_size': config.VALID_BATCH_SIZE,
                'shuffle': False,
                'num_workers': 0
            }

            test_loader = DataLoader(test_set, **test_params)
            actuals = []
            for data_item in test_dataset:
                actuals.append(data_item['y'])
            if args.MODE == 'val':

                print("Evaluating on val set...")
                best_beam_size = None
                for beam_size in range(1, 11, 2):
                    print(f"Beam Size: {beam_size}")
                    predictions = validate(model, tokenizer, device, test_loader,
                                                    max_generated_length=config.SUMMARY_LEN, beam_size=beam_size)

                    current_rl = get_rouge(predictions, actuals, is_print=True)
                    if current_rl>best_rl:
                        best_rl=current_rl
                        best_beam_size = beam_size
                        best_model_path=modelpath
                    else:
                        not_required_checkpoints.append(modelpath)

                    decoded_preds_temp = [pred.strip() for pred in predictions]
                    decoded_labels_temp = [label.strip() for label in actuals]


                    bertscore_result = bertscore_metric.compute(predictions=decoded_preds_temp,
                                                                references=decoded_labels_temp,
                                                                model_type="bert-base-uncased")
                    bertscore_f1 = mean([round(v, 4) for v in bertscore_result["f1"]])
                    print(f"bertscore_f1: {bertscore_f1}")

                best_beam_info[modelpath]=best_beam_size
                with open(os.path.join(config.OUTPUT_PATH, 'best_beam_size.json'), 'w') as wfile:
                    json.dump(best_beam_info, wfile)

            elif args.MODE=='test':
                print("Evaluating on test set...")
                with open(os.path.join(config.OUTPUT_PATH, 'best_beam_size.json'), 'r') as rfile:
                    best_beam_info_loaded = json.load(rfile)
                best_beam_size = best_beam_info_loaded[modelpath]
                print(f"Best Beam Size: {best_beam_size}")
                predictions = validate(model, tokenizer, device, test_loader,
                                       max_generated_length=config.SUMMARY_LEN, beam_size=best_beam_size)
                test_rl = get_rouge(predictions, actuals, is_print=True)
                decoded_preds_temp = [pred.strip() for pred in predictions]
                decoded_labels_temp = [label.strip() for label in actuals]


                bertscore_result = bertscore_metric.compute(predictions=decoded_preds_temp,
                                                            references=decoded_labels_temp,
                                                            model_type="bert-base-uncased")
                bertscore_f1 = mean([round(v, 4) for v in bertscore_result["f1"]])
                print(f"bertscore_f1: {bertscore_f1}")
                final_df = pd.DataFrame({'Generated Text': predictions, 'Actual Text': actuals})
                final_df.to_csv(config.OUTPUT_PATH+'/predictions-test.csv')
                print('Output Files generated for review')

    if args.MODE=='val':
        print(f"Best model path: {best_model_path}")
        print("Deleting the other checkpoints...")
        for name in glob.glob(args.MODEL_PATH + '/checkpoint-*/'):
            if os.path.isdir(name):
                modelpath = os.path.join(args.MODEL_PATH, name)
                if modelpath != best_model_path and best_model_path is not None:
                    try:
                        shutil.rmtree(modelpath, ignore_errors=True)
                        print(f"Deleted: {modelpath}")
                    except OSError as e:
                        print("Error: %s : %s" % (modelpath, e.strerror))

if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument('--MODEL_PATH', type=str, default=None)
    parser.add_argument('--DATASET_PATH', type=str, default=None)
    parser.add_argument('--OUTPUT_PATH', type=str, default=None)
    parser.add_argument('--MAX_LEN', type=int, default=None)
    parser.add_argument('--SUMMARY_LEN', type=int, default=None)
    parser.add_argument('--MODE', type=str, default=None)
    parser.add_argument('--MODEL_NAME', type=str, default=None)

    args = parser.parse_args()
    print(args)
    main(args)
