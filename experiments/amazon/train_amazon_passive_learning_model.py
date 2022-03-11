import os
import argparse
import pickle
import random
from collections import Counter

from transformers import \
    (AutoConfig,
     AutoTokenizer,
     AutoModelForSequenceClassification,
     TrainingArguments,
     Trainer,
     )
import transformers
import torch
import numpy as np
from torch.utils.data import TensorDataset
from sklearn.metrics import accuracy_score

from data.data_loader_amazon import AmazonDataLoader
from utils import seed_everything

parser = argparse.ArgumentParser()

# Data path
parser.add_argument("--corpus_path", type=str)
parser.add_argument("--target_domain", type=str)

# Checkpoints
parser.add_argument('--output_path', type=str)

# Model and tokenizer names
parser.add_argument('--source_domain_model_name', type=str)

# MISC
parser.add_argument('--seed', type=int, default=42)

# Tunable
parser.add_argument('--max_length', type=int, default=256)
parser.add_argument('--per_gpu_train_batch_size', type=int, default=8)
parser.add_argument('--per_gpu_eval_batch_size', type=int, default=8)
parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--warmup_steps', type=float, default=0)
parser.add_argument('--weight_decay', type=float, default=0.1)
parser.add_argument('--max_grad_norm', type=float, default=1.0)
parser.add_argument('--learning_rate', type=float, default=5e-5)

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


if __name__ == "__main__":
    # Parse the command line arguments.
    args = parser.parse_args()

    # Print the current settings
    print('-' * 80)
    print(f"Train passive learning model on {args.target_domain}")

    # Seed everything.
    seed_everything(args.seed)

    # Count the available number of GPUs and save the number to args.
    print("-" * 80)
    args.num_gpus = torch.cuda.device_count()
    print("Number of GPUs = {}".format(args.num_gpus))

    # Load the source domain model
    print("-" * 80)
    print(f"Load source-domain model: {args.source_domain_model_name}")
    config = AutoConfig.from_pretrained(args.source_domain_model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.source_domain_model_name)
    model = AutoModelForSequenceClassification.from_pretrained(args.source_domain_model_name,
                                                               config=config)

    # Load target domain raw text data
    print('-' * 80)
    print(f"Load target domain ({args.target_domain}) data")
    amazon_data_provider = AmazonDataLoader(corpus_path=args.corpus_path)
    target_dev_examples, target_dev_labels, \
        target_test_examples, target_test_labels = amazon_data_provider.read_domain_data(domain=args.target_domain)
    print(f"Number of development data in target domain ({args.target_domain}) = {len(target_dev_labels)}")
    print(f"Target domain development set label distribution: {dict(Counter(target_dev_labels))}")
    print(f"Number of test data in target domain ({args.target_domain}) = {len(target_test_labels)}")
    print(f"Target domain test set label distribution: {dict(Counter(target_test_labels))}")

    # Prepare the datasets
    target_test_dataset = AmazonDataset(examples=target_test_examples,
                                        labels=target_test_labels,
                                        max_length=args.max_length,
                                        pretrained_tokenizer=tokenizer)

    # Passive learning dataset
    print("-" * 80)
    print('Build the passive learning dataset')

    # Random sample 96 examples from development set
    seed_everything(args.seed)
    passive_learning_examples, passive_learning_labels = zip(
        *random.sample(list(zip(target_dev_examples, target_dev_labels)), 96))
    print('\n')
    print(f"number of examples used in passive learning: {len(passive_learning_examples)}")
    print(f'passive learning dataset label distribution {dict(Counter(passive_learning_labels))}')

    # Create the dataset
    passive_learning_dataset = AmazonDataset(examples=passive_learning_examples,
                                             labels=passive_learning_labels,
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
                                      logging_dir=None,
                                      weight_decay=args.weight_decay)

    trainer = Trainer(model=model,
                      args=training_args,
                      train_dataset=passive_learning_dataset)
    transformers.logging.set_verbosity_warning()

    # Evaluate the model on test set
    print("Run source-domain model on test set: ")
    source_acc, source_logits = evaluate(trainer, dataset=target_test_dataset)

    # Save the source domain model's predicted logits
    with open(os.path.join(args.output_path, 'source_preds.pkl'), 'wb') as f:
        pickle.dump(source_logits, f)

    # Train the passive learning dataset
    print('-' * 80)
    print(f"Do passive learning on randomly selected 96 examples from {args.target_domain} domain")
    print({'num_examples': len(trainer.train_dataset),
           'learning_rate': training_args.learning_rate,
           'per_device_batch': trainer.args.per_device_train_batch_size,
           'epochs': trainer.args.num_train_epochs,
           'grad_accum': trainer.args.gradient_accumulation_steps,
           'warmup': trainer.args.warmup_steps}.__str__())
    trainer.train()

    # Save the trained model
    trainer.model.save_pretrained(args.output_path)
    tokenizer.save_pretrained(args.output_path)

    # Evaluate the model on dev again
    print('\n\n')
    print('Evaluate the fine-tuned (oracle) model on test set:')
    passive_acc, passive_logits = evaluate(trained_trainer=trainer, dataset=target_test_dataset)

    # Save the passive learned model's predicted logits
    with open(os.path.join(args.output_path, 'passive_model_preds.pkl'), 'wb') as f:
        pickle.dump(passive_logits, f)

    # Print all arguments
    print('-' * 80)
    print('All the arguments used: ')
    for key, value in args.__dict__.items():
        print(f"{key}: {value}")
    print("\n\n\n")
