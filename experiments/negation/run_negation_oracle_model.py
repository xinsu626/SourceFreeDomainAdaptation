import os
import argparse
import pickle
from datetime import datetime

from transformers import \
    (AutoConfig,
     AutoTokenizer,
     AutoModelForSequenceClassification,
     TrainingArguments,
     Trainer,
     )
import torch
from torch.utils.data import TensorDataset
import numpy as np

from data.data_loader_negation import DataLoaderNegation
from utils import seed_everything, negation_performance

parser = argparse.ArgumentParser()

# Data path
parser.add_argument('--test_data_pth', type=str)
parser.add_argument('--test_labels_path', type=str)
parser.add_argument('--dev_data_path', type=str)
parser.add_argument('--dev_labels_path', type=str)

# Checkpoints
parser.add_argument('--output_path', type=str)

# Model and tokenizer names
parser.add_argument('--model_name', type=str, default='tmills/roberta_sfda_sharpseed')
parser.add_argument('--tokenizer_name', type=str, default='tmills/roberta_sfda_sharpseed')
parser.add_argument('--config_name', type=str, default='tmills/roberta_sfda_sharpseed')

# MISC
parser.add_argument('--seed', type=int, default=42)

# Tunable
parser.add_argument('--max_length', type=int, default=128)
parser.add_argument('--per_gpu_train_batch_size', type=int, default=8)
parser.add_argument('--per_gpu_eval_batch_size', type=int, default=32)
parser.add_argument('--grad_accum_steps', type=int, default=1)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--warmup_steps', type=float, default=0)
parser.add_argument('--weight_decay', type=float, default=0.0)
parser.add_argument('--learning_rate', type=float, default=5e-5)

os.environ['WANDB_DISABLED'] = 'true'


# Dataset Class
class NegationDataset(torch.utils.data.Dataset):
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
    outputs = trained_trainer.predict(test_dataset=dataset)
    f1, precision, recall = negation_performance(labels_true=outputs.label_ids,
                                                 labels_pred=np.argmax(outputs.predictions, 1))

    print(f'F1 = {f1:.3f}')
    print(f'Precision = {precision:.3f}')
    print(f'Recall = {recall:.3f}')
    print(f'Number of examples evaluated = {len(dataset)}')

    pred_logits_flatten = outputs.predictions.flatten()

    return f1, precision, recall, pred_logits_flatten


if __name__ == "__main__":
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

    # Load the model
    print("-" * 80)
    print(f"Load model: {args.model_name}")
    config = AutoConfig.from_pretrained(args.config_name)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, config=config)

    # Load the raw text data
    print('-' * 80)
    print('Load the data')
    dev_provider = DataLoaderNegation(corpus_path=args.dev_data_path,
                                      label_path=args.dev_labels_path)
    test_provider = DataLoaderNegation(corpus_path=args.test_data_pth,
                                       label_path=args.test_labels_path)
    dev_examples, dev_label_ids = dev_provider.get_text_data()
    test_examples, test_label_ids = test_provider.get_text_data()
    print(f'Number of dev data = {len(dev_examples)}')
    print(f'Number of test data = {len(test_examples)}')

    # Prepare the dataset
    test_dataset = NegationDataset(examples=test_examples,
                                   labels=[0 if i == -1 else i for i in test_label_ids],
                                   max_length=args.max_length,
                                   pretrained_tokenizer=tokenizer)
    dev_dataset = NegationDataset(examples=dev_examples,
                                  labels=[0 if i == -1 else 1 for i in dev_label_ids],
                                  max_length=args.max_length,
                                  pretrained_tokenizer=tokenizer)

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
                                      max_grad_norm=1.0)
    trainer = Trainer(model=model, args=training_args, train_dataset=dev_dataset)

    # Evaluate the model on test set
    print('-' * 80)
    print("Run source-domain model on test set: ")
    source_f, source_p, source_r, preds_to_save = evaluate(trainer, dataset=test_dataset)

    # Save the flattened preds
    with open(os.path.join(args.output_path, 'flattened_source_preds.pkl'), 'wb') as f:
        pickle.dump(preds_to_save, f)

    # Train oracle model on the dev set
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

    # Eval on test set
    print('\nEvaluate on the test set')
    _, _, _, preds_to_save = evaluate(trainer, dataset=test_dataset)

    # Save the flattened preds
    with open(os.path.join(args.output_path, 'flattened_oracle_preds.pkl'), 'wb') as f:
        pickle.dump(preds_to_save, f)

    # Print all arguments
    print('-' * 80)
    print('All the arguments used: ')
    for key, value in args.__dict__.items():
        print(f"{key}: {value}")
