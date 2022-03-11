import os
import argparse
import pickle
import string
from collections import Counter

from transformers import \
    (AutoConfig,
     AutoTokenizer,
     AutoModelForSequenceClassification,
     TrainingArguments,
     Trainer)
import torch
import numpy as np
from torch.utils.data import TensorDataset
from sklearn.metrics import accuracy_score
from sklearn.model_selection._search import ParameterGrid

from utils import seed_everything

parser = argparse.ArgumentParser()

# Data path
parser.add_argument('--corpus_path', type=str)
parser.add_argument('--source_domain', type=str)

# Checkpoints
parser.add_argument('--output_path', type=str)

# Model and tokenizer names
parser.add_argument('--model_name', type=str, default='roberta-base')
parser.add_argument('--tokenizer_name', type=str, default='roberta-base')
parser.add_argument('--config_name', type=str, default='roberta-base')

# MISC
parser.add_argument('--seed', type=int, default=42)

# Tunable and model config
parser.add_argument('--num_labels', type=int, default=2)  # Binary classification task
parser.add_argument('--max_length', type=int, default=256)
parser.add_argument('--per_gpu_train_batch_size', type=int, default=8)
parser.add_argument('--per_gpu_eval_batch_size', type=int, default=8)
parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--warmup_steps', type=float, default=0)
parser.add_argument('--weight_decay', type=float, default=0.1)
parser.add_argument('--learning_rate', type=float, default=5e-5)
parser.add_argument('--max_grad_norm', type=float, default=1.0)

os.environ['WANDB_DISABLED'] = 'true'


# Dataset Class
class AmazonDataset(torch.utils.data.Dataset):
    def __init__(self,
                 examples,
                 pretrained_tokenizer,
                 max_length,
                 labels=None):
        self.examples = examples
        self.labels = labels
        self.pretrained_tokenizer = pretrained_tokenizer
        self.max_length = max_length

    def __getitem__(self, item):
        tokenized_features = self.pretrained_tokenizer(self.examples[item],
                                                       max_length=self.max_length,
                                                       padding='max_length',
                                                       truncation=True,
                                                       return_tensors='pt')
        input_ids = tokenized_features.input_ids.squeeze()
        attention_mask = tokenized_features.attention_mask.squeeze()

        outputs = dict(input_ids=input_ids,
                       attention_mask=attention_mask)

        if self.labels is not None:
            outputs['labels'] = torch.tensor(self.labels[item])

        return outputs

    def __len__(self):
        return len(self.examples)


def evaluate(trained_trainer, dataset):
    """Calculate the accuracy and return accuracy and predicted logits"""
    print('Evaluation: ')
    print(f'Number of examples to eval = {len(dataset)}')
    outputs = trained_trainer.predict(test_dataset=dataset)

    # Get true label
    true_labels = outputs.label_ids
    logits = outputs.predictions
    pred_labels = np.argmax(logits, -1)

    # Calculate the accuracy
    acc = accuracy_score(y_true=true_labels, y_pred=pred_labels)
    print(f"Accuracy = {acc:.3f}")

    return acc, logits


def clean_text(examples):
    examples = [s.translate(str.maketrans('', '', string.punctuation)) for s in examples]
    examples = [s.replace('\n', ' ').replace('\t', ' ') for s in examples]
    examples = [" ".join(s.split()) for s in examples]
    examples = [s.lower() for s in examples]

    return examples


def load_perl_data(corpus, domain):
    # Get the full path for train, dev and test
    full_train_path = os.path.join(corpus, domain, "fold-1", 'train')
    full_dev_path = os.path.join(corpus, domain, "fold-1", 'dev')
    full_test_path = os.path.join(corpus, domain, 'test')

    # Load the data
    with open(full_train_path, 'rb') as f:
        train_examples, train_labels = pickle.load(f)

    with open(full_dev_path, 'rb') as f:
        dev_examples, dev_labels, = pickle.load(f)

    with open(full_test_path, 'rb') as f:
        test_examples, test_labels = pickle.load(f)

    return {'train': (clean_text(train_examples), train_labels),
            'dev': (clean_text(dev_examples), dev_labels),
            'test': (clean_text(test_examples), test_labels)}


if __name__ == "__main__":
    # Parse the command line arguments.
    args = parser.parse_args()

    # Seed everything.
    seed_everything(args.seed)

    # Count the available number of GPUs and save the number to args.
    print("-" * 80)
    args.num_gpus = torch.cuda.device_count()
    print("Number of GPUs = {}".format(args.num_gpus))

    # Load the model
    print("-" * 80)
    print(f"Load model: {args.model_name}")
    config = AutoConfig.from_pretrained(args.config_name,
                                        num_labels=args.num_labels)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)

    # Get all target domain
    target_domains = ['books', 'dvd', 'electronics', 'kitchen']
    target_domains.remove(args.source_domain)

    # Load the PERL split data
    print('-' * 80)
    print("Load the PERL split amazon data: \n")
    source_data = load_perl_data(corpus=args.corpus_path,
                                 domain=args.source_domain)

    source_train_examples, source_train_labels = source_data['train']
    source_dev_examples, source_dev_labels = source_data['dev']

    print(f'Number of examples in source-domain training data: {len(source_train_examples)}')
    print(f'Label distribution: {dict(Counter(source_train_labels))}\n')

    print(f'Number of examples in source-domain dev data: {len(source_dev_examples)}')
    print(f'Label distribution: {dict(Counter(source_dev_labels))}\n')

    all_target_domain_data = {}
    for domain_name in target_domains:
        all_target_domain_data[domain_name] = load_perl_data(corpus=args.corpus_path,
                                                             domain=domain_name)

    # Prepare the datasets
    source_train_dataset = AmazonDataset(examples=source_train_examples,
                                         labels=source_train_labels,
                                         max_length=args.max_length,
                                         pretrained_tokenizer=tokenizer)
    source_dev_dataset = AmazonDataset(examples=source_dev_examples,
                                       labels=source_dev_labels,
                                       max_length=args.max_length,
                                       pretrained_tokenizer=tokenizer)

    # Prepare the training argument and trainer
    training_args = TrainingArguments(output_dir=args.output_path,
                                      per_device_train_batch_size=args.per_gpu_train_batch_size,
                                      per_device_eval_batch_size=args.per_gpu_eval_batch_size,
                                      gradient_accumulation_steps=args.gradient_accumulation_steps,
                                      learning_rate=args.learning_rate,
                                      num_train_epochs=args.epochs,
                                      warmup_steps=args.warmup_steps,
                                      save_total_limit=0,
                                      logging_steps=100,
                                      seed=args.seed,
                                      disable_tqdm=True,
                                      save_steps=10000,
                                      adam_epsilon=1e-08,
                                      max_grad_norm=args.max_grad_norm,
                                      eval_steps=500,
                                      weight_decay=args.weight_decay,
                                      evaluation_strategy="epoch",
                                      logging_dir=None)


    def compute_metric(eval_pred):
        """metric function for the trainer."""
        predictions, labels = eval_pred
        predictions = predictions.argmax(axis=-1)
        accuracy = accuracy_score(y_true=labels, y_pred=predictions)

        return {'accuracy': accuracy}


    def model_init():
        """A function that instantiates the model to be used."""
        model = AutoModelForSequenceClassification.from_pretrained(args.model_name,
                                                                   config=config)
        return model


    trainer = Trainer(args=training_args,
                      train_dataset=source_train_dataset,
                      eval_dataset=source_dev_dataset,
                      model_init=model_init,
                      compute_metrics=compute_metric)

    # Start doing tuning
    print('-' * 80)
    print("Start tuning the model")
    print('=' * 80)
    print({'num_examples': len(source_train_dataset),
           'per_device_batch': trainer.args.per_device_train_batch_size,
           'epochs': trainer.args.num_train_epochs,
           'grad_accum': trainer.args.gradient_accumulation_steps,
           'warmup': trainer.args.warmup_steps}.__str__())
    # Search space
    space = {"learning_rate": [1e-5, 2e-5, 3e-5],
             "gradient_accumulation_steps": [2, 4]}
    results = []

    # Loop through each setting
    num_trials = len(ParameterGrid(space))
    for idx, hyper_param in enumerate(ParameterGrid(space)):
        print("\n\n")
        print(f"Trail {idx + 1}/{num_trials}: ")
        print(f"Current setting: {hyper_param}")
        print("=" * 30)
        for n, v in hyper_param.items():
            setattr(trainer.args, n, v)
        trainer.train()
        print('*' * 3)
        # Evaluate the model on dev
        acc, _ = evaluate(trained_trainer=trainer, dataset=source_dev_dataset)
        results.append({"accuracy": acc,
                        "hyperparameters": hyper_param})

    # Sort the results
    sorted_results = sorted(results,
                            key=lambda k: k['accuracy'],
                            reverse=True)

    # Get the best and retrain the model
    best = sorted_results[0]
    print("\n\n")
    print("*" * 3)
    print(f"Best run: {best}")

    print('\n\n')
    for n, v in best["hyperparameters"].items():
        setattr(trainer.args, n, v)
    print('-' * 80)
    print('Start training the best model')
    print("=" * 80)
    print({'num_examples': len(source_train_dataset),
           'per_device_batch': trainer.args.per_device_train_batch_size,
           'epochs': trainer.args.num_train_epochs,
           'grad_accum': trainer.args.gradient_accumulation_steps,
           'warmup': trainer.args.warmup_steps}.__str__())
    trainer.train()

    # Save the model
    trainer.model.save_pretrained(args.output_path)
    tokenizer.save_pretrained(args.output_path)

    # Evaluate the model on target
    print('-' * 80)
    print("Evaluate on target domains")
    print('\n')

    for domain_name, domain_data in all_target_domain_data.items():
        print(f"domain: {domain_name}")
        print('-' * 3)
        target_test_examples, target_test_labels = domain_data['test']
        print(f'Number of examples in target-domain test data: {len(target_test_examples)}')
        print(f'Label distribution: {dict(Counter(target_test_labels))}\n')
        target_test_dataset = AmazonDataset(examples=target_test_examples,
                                            labels=target_test_labels,
                                            max_length=args.max_length,
                                            pretrained_tokenizer=tokenizer)
        evaluate(trained_trainer=trainer, dataset=target_test_dataset)
        print('\n\n')

    # Print out all args
    print("-" * 80)
    print("All commandline arguments: ")
    print(args.__dict__)
