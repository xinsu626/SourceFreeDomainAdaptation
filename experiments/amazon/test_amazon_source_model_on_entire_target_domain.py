import os
import argparse
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
parser.add_argument('--per_gpu_eval_batch_size', type=int, default=8)

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
    args = parser.parse_args()

    print('-' * 80)
    print(f"Test the source domain model on {args.target_domain}")

    seed_everything(args.seed)

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
    print(f"Load target domain ({args.target_domain}) data:")
    amazon_data_provider = AmazonDataLoader(corpus_path=args.corpus_path)
    target_dev_examples, target_dev_labels, \
        target_test_examples, target_test_labels = amazon_data_provider.read_domain_data(domain=args.target_domain)
    print(f"Number of development data in target domain ({args.target_domain}) = {len(target_dev_labels)}")
    print(f"Target domain development set label distribution: {dict(Counter(target_dev_labels))}\n")

    print(f"Number of test data in target domain ({args.target_domain}) = {len(target_test_labels)}")
    print(f"Target domain test set label distribution: {dict(Counter(target_test_labels))}\n")

    print('Combine them together to test the source-domain model on entire target domain: ')
    target_examples = target_dev_examples + target_test_examples
    target_labels = target_dev_labels + target_test_labels
    print(f"Total number of test examples = {len(target_examples)}")
    print(f"Entire label distribution = {dict(Counter(target_labels))}")

    # Prepare the datasets
    target_dataset = AmazonDataset(examples=target_examples,
                                   labels=target_labels,
                                   max_length=args.max_length,
                                   pretrained_tokenizer=tokenizer)

    target_test_dataset = AmazonDataset(examples=target_test_examples,
                                        labels=target_test_labels,
                                        max_length=args.max_length,
                                        pretrained_tokenizer=tokenizer)

    # Prepare the training argument and trainer
    training_args = TrainingArguments(output_dir=args.output_path,
                                      per_device_eval_batch_size=args.per_gpu_eval_batch_size,
                                      save_total_limit=0,
                                      logging_steps=100,
                                      seed=args.seed,
                                      disable_tqdm=True,
                                      save_steps=10000,
                                      logging_dir=None)

    # Build the trainer
    trainer = Trainer(model=model,
                      args=training_args)
    transformers.logging.set_verbosity_warning()

    # Evaluate the model on test set
    print("Run source-domain model on test set for sanity check: ")
    test_acc, test_logits = evaluate(trainer, dataset=target_test_dataset)
    print('\n')

    # Evaluate the model on entire target domain
    print('RUn source-domain model on entire target domain: ')
    target_acc, target_logits = evaluate(trainer, dataset=target_dataset)

    # Print all arguments
    print('-' * 80)
    print('All the arguments used: ')
    for key, value in args.__dict__.items():
        print(f"{key}: {value}")
    print("\n\n\n")
