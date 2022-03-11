import os
import argparse
import pickle
from datetime import datetime

from transformers import \
    (AutoConfig,
     AutoTokenizer,
     AutoModelForTokenClassification,
     TrainingArguments,
     Trainer)
import torch
from torch.utils.data import TensorDataset
import numpy as np

from data.data_loader_time import TimeDataLoader
from utils import seed_everything, time_performance

parser = argparse.ArgumentParser()

# raw text data paths
parser.add_argument('--test_data_path', type=str)
parser.add_argument('--dev_data_path',type=str)

# Checkpoints
parser.add_argument('--output_path', type=str)

# Model and tokenizer names
parser.add_argument('--model_name', type=str, default='clulab/roberta-timex-semeval')
parser.add_argument('--tokenizer_name', type=str, default='clulab/roberta-timex-semeval')
parser.add_argument('--config_name', type=str, default='clulab/roberta-timex-semeval')

# MISC
parser.add_argument('--seed', type=int, default=42)

# Tunable
parser.add_argument('--per_gpu_train_batch_size', type=int, default=2)
parser.add_argument('--max_length', type=int, default=271)
parser.add_argument('--per_gpu_eval_batch_size', type=int, default=2)
parser.add_argument('--grad_accum_steps', type=int, default=1)
parser.add_argument('--epochs', type=int, default=3)
parser.add_argument('--warmup_steps', type=float, default=500)
parser.add_argument('--weight_decay', type=float, default=0.01)
parser.add_argument('--learning_rate', type=float, default=5e-5)


# Dataset class
class TimeDataset(torch.utils.data.Dataset):
    def __init__(self,
                 input_ids,
                 attention_mask,
                 offset_mapping=None,
                 doc_indices=None,
                 label_ids=None):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.offset_mapping = offset_mapping
        self.doc_indices = doc_indices
        self.label_ids = label_ids

    def __getitem__(self, item):
        outputs = dict(input_ids=self.input_ids[item],
                       attention_mask=self.attention_mask[item])
        if self.label_ids is not None:
            outputs['labels'] = self.label_ids[item]

        return outputs

    def __len__(self):
        return len(self.input_ids)


def evaluate(trained_trainer: Trainer,
             dataset: TimeDataset,
             data_provider: TimeDataLoader,
             pred_data_path: str,
             gold_data_path: str):
    pred_logits = trained_trainer.predict(test_dataset=dataset).predictions

    # Write out the predictions
    data_provider.write_anafora(output_dir=pred_data_path,
                                input_ids=dataset.input_ids,
                                offset_mapping=dataset.offset_mapping,
                                doc_indexes=dataset.doc_indices,
                                predictions=pred_logits)
    # Evaluate the two directories
    f1, precision, recall = time_performance(gold_standard_dir=gold_data_path,
                                             predictions_dir=pred_data_path)

    print(f"F1 = {f1:.3f}")
    print(f"Precision = {precision:.3f}")
    print(f"Recall = {recall:.3f}")
    print(f"Number of examples evaluated = {len(dataset)}")

    # Get the predictions to save
    avg_predictions = np.mean(pred_logits, axis=2)
    flattened_avg_predictions = avg_predictions.flatten()

    return f1, precision, recall, flattened_avg_predictions


if __name__ == "__main__":
    # Disable multiprocessing.
    os.environ['TOKENIZERS_PARALLELISM'] = "false"

    # Parse the command line arguments.
    args = parser.parse_args()

    # Make the output directory
    args.output_path = os.path.join(args.output_path, datetime.now().strftime('%Y-%m-%d-%H-%M'))
    os.makedirs(args.output_path, exist_ok=True)

    # Seed everything.
    seed_everything(args.seed)

    # Count the available number of GPUs and save the number to args.
    print("-" * 80)
    args.num_gpus = torch.cuda.device_count()
    print("Number of GPUs = {}".format(args.num_gpus))

    # Load the model, config and tokenizer
    print("-" * 80)
    print(f"Loading the model: {args.model_name}")
    config = AutoConfig.from_pretrained(args.config_name)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name,  # use fast version of the tokenizer.
                                              use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(args.model_name,
                                                            config=config)

    # Load the dev and test data.
    print("-" * 80)
    print("Loading the features from text file.")
    dev_set_provider = TimeDataLoader(corpus_dir=args.dev_data_path,
                                      tokenizer=tokenizer,
                                      config=config,
                                      max_length=args.max_length)
    test_set_provider = TimeDataLoader(corpus_dir=args.test_data_path,
                                       tokenizer=tokenizer,
                                       config=config,
                                       max_length=args.max_length)
    # Get the bert features
    dev_input_ids, dev_attention_mask, dev_offset_mapping, dev_doc_indices, dev_label_ids = \
        dev_set_provider.convert_documents_to_features(return_labels=True)
    test_input_ids, test_attention_mask, test_offset_mapping, test_doc_indices, test_label_ids = \
        test_set_provider.convert_documents_to_features(return_labels=True)
    print(f'Number of sentences in dev = {len(dev_input_ids)}')
    print(f'Number of sentences in test = {len(test_input_ids)}')

    # Make the dataset objects
    # Build the dev dataset
    dev_dataset = TimeDataset(input_ids=dev_input_ids,
                              attention_mask=dev_attention_mask,
                              label_ids=dev_label_ids,
                              offset_mapping=dev_offset_mapping,
                              doc_indices=dev_doc_indices)

    test_dataset = TimeDataset(input_ids=test_input_ids,
                               attention_mask=test_attention_mask,
                               label_ids=test_label_ids,
                               offset_mapping=test_offset_mapping,
                               doc_indices=test_doc_indices)

    # Prepare the training argument and trainer
    training_args = TrainingArguments(output_dir=args.output_path,
                                      per_device_train_batch_size=args.per_gpu_train_batch_size,
                                      per_device_eval_batch_size=args.per_gpu_eval_batch_size,
                                      gradient_accumulation_steps=args.grad_accum_steps,
                                      learning_rate=args.learning_rate,
                                      num_train_epochs=args.epochs,
                                      warmup_steps=args.warmup_steps,
                                      save_total_limit=1,
                                      logging_steps=100,
                                      seed=args.seed,
                                      disable_tqdm=True,
                                      save_steps=10000,
                                      adam_epsilon=1e-08,
                                      max_grad_norm=1.0,
                                      weight_decay=args.weight_decay)
    trainer = Trainer(model=model, args=training_args, train_dataset=dev_dataset)

    # Evaluate the source-domain model on test set
    print('-' * 80)
    print('Evaluate the source-domain model on test set： ')
    source_f1, source_p, source_r, source_preds_to_save = evaluate(trained_trainer=trainer,
                                                                   dataset=test_dataset,
                                                                   pred_data_path=os.path.join(args.output_path, 'source-model-preds'),
                                                                   gold_data_path=args.test_data_path,
                                                                   data_provider=test_set_provider)
    # Save the flattened preds
    with open(os.path.join(args.output_path, 'flattened_source_preds.pkl'), 'wb') as f:
        pickle.dump(source_preds_to_save, f)

    # Train a oracle model on dev set
    print('-' * 80)
    print('Start training the model on dev set')
    print('=' * 80)
    print({'num_examples': len(trainer.train_dataset),
           'batch': trainer.args.per_device_train_batch_size,
           'epochs': trainer.args.num_train_epochs,
           'grad_accum': trainer.args.gradient_accumulation_steps,
           'warmup': trainer.args.warmup_steps}.__str__())
    seed_everything(args.seed)
    trainer.train()
    trainer.model.save_pretrained(args.output_path)
    tokenizer.save_pretrained(args.output_path)
    print('=' * 80)

    print('Evaluate the current model on test set')
    pred_data_path = os.path.join(args.output_path, 'oracle_preds')
    _, _, _, oracle_preds_to_save = evaluate(trained_trainer=trainer,
                                             dataset=test_dataset,
                                             pred_data_path=pred_data_path,
                                             gold_data_path=args.test_data_path,
                                             data_provider=test_set_provider)
    # Save the flattened preds
    with open(os.path.join(args.output_path, 'flattened_oracle_preds.pkl'), 'wb') as f:
        pickle.dump(oracle_preds_to_save, f)

    # Print out all the settings for future reference
    print('-' * 80)
    print('Settings: ')
    for key, value in args.__dict__.items():
        print(f"{key}: {value}")
