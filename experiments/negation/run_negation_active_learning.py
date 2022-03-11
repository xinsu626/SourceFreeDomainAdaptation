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
from scipy.special import softmax
from scipy.stats import entropy
import numpy as np

from data.data_loader_negation import DataLoaderNegation
from utils import seed_everything, negation_performance
from negation_augmenter import EntityReplacementAugmenter

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
parser.add_argument('--iteratively_train_model', type=str, default='False')
parser.add_argument('--iteratively_build_train_data', type=str, default='True')
parser.add_argument('--do_augmentation', type=str, default='False')
parser.add_argument('--num_new_examples', type=int, default=5)
parser.add_argument('--active_learning_iters', type=int, default=12)
parser.add_argument('--active_learning_size', type=int, default=8)
parser.add_argument('--max_length', type=int, default=128)
parser.add_argument('--per_gpu_train_batch_size', type=int, default=32)
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
    print('Evaluation: ')
    print(f'Number of examples to eval = {len(dataset)}')
    outputs = trained_trainer.predict(test_dataset=dataset)
    f1, precision, recall = negation_performance(labels_true=outputs.label_ids,
                                                 labels_pred=np.argmax(outputs.predictions, 1))

    print(f'F1 = {f1:.3f}')
    print(f'Precision = {precision:.3f}')
    print(f'Recall = {recall:.3f}')

    pred_logits_flatten = outputs.predictions.flatten()

    return f1, precision, recall, pred_logits_flatten


if __name__ == "__main__":
    # Parse the command line arguments.
    args = parser.parse_args()

    # Print the current settings
    print('-' * 80)
    info = {
        'AL_size': args.active_learning_size,
        'AL_iters': args.active_learning_iters,
        'train_epochs': args.epochs,
        'batch': args.per_gpu_train_batch_size,
        'imodel': args.iteratively_train_model,
        'idata': args.iteratively_build_train_data,
        'data_augmentation': args.do_augmentation
    }
    print(info)

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

    # Get the entity pool for dev sets
    augmenter = None
    if eval(args.do_augmentation):
        augmenter = EntityReplacementAugmenter(num_new_examples=args.num_new_examples)
        POOL = augmenter.get_all_entities(test_examples)
        augmenter.entity_pool = POOL

    # Prepare the dataset
    test_dataset = NegationDataset(examples=test_examples,
                                   labels=[0 if i == -1 else i for i in test_label_ids],
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
                                      logging_steps=1,
                                      seed=args.seed,
                                      disable_tqdm=True,
                                      save_steps=10000,
                                      adam_epsilon=1e-08,
                                      max_grad_norm=1.0)
    trainer = Trainer(model=model, args=training_args)

    # Evaluate the model on test set
    print("Run source-domain model on test set: ")
    source_f, source_p, source_r, preds_to_save = evaluate(trainer, dataset=test_dataset)

    # Save the flattened preds
    with open(os.path.join(args.output_path, 'flattened_source_preds.pkl'), 'wb') as f:
        pickle.dump(preds_to_save, f)

    # Active learning
    print("-" * 80)
    print('Start doing active learning on the dev set.')
    train_examples = []
    train_label_ids = []

    for iter_idx in range(args.active_learning_iters):
        print('-' * 40)
        print('Active Learning Iteration {}: '.format(iter_idx + 1))

        # Build the dev dataset, since some data is removed from it
        dev_dataset = NegationDataset(examples=dev_examples,
                                      labels=[0 if i == -1 else 1 for i in dev_label_ids],
                                      max_length=args.max_length,
                                      pretrained_tokenizer=tokenizer)

        # Run model on dev set
        pred_dev_logits = trainer.predict(test_dataset=dev_dataset).predictions
        pred_dev_probas = softmax(pred_dev_logits, axis=1)

        # Get entropy
        pred_dev_entropy = entropy(pred_dev_probas, axis=1)

        # Sort the sentence based on the entropies
        indices = np.argsort(pred_dev_entropy).tolist()[::-1][:args.active_learning_size]

        # Set the high entropy examples
        selected_examples = [dev_examples[i] for i in indices]
        selected_labels_ids = [dev_label_ids[i] for i in indices]

        # Iteratively build up the dataset
        if eval(args.iteratively_build_train_data):
            # Add then to train
            train_examples += selected_examples
            train_label_ids += selected_labels_ids

            # Remove selected from dev set
            dev_examples = [i for j, i in enumerate(dev_examples) if j not in indices]
            dev_label_ids = [i for j, i in enumerate(dev_label_ids) if j not in indices]
        else:
            train_examples = selected_examples
            train_label_ids = selected_labels_ids

        # Do data augmentation
        if eval(args.do_augmentation):
            print('Do data augmentation using entity replacement.')
            augmented_examples, augmented_label_ids = augmenter.augment(selected_examples,
                                                                        selected_labels_ids)
            if augmented_examples:
                train_examples += augmented_examples
                train_label_ids += augmented_label_ids

        # Build training set.
        train_dataset = NegationDataset(examples=train_examples,
                                        labels=[0 if i == -1 else i for i in train_label_ids],
                                        pretrained_tokenizer=tokenizer,
                                        max_length=args.max_length)

        # Update the output path with current iteration.
        current_iteration_output_path = os.path.join(args.output_path,
                                                     str(iter_idx))
        os.makedirs(current_iteration_output_path, exist_ok=True)
        training_args.output_dir = current_iteration_output_path

        # Update the trainer
        trainer = Trainer(model=trainer.model, args=training_args, train_dataset=train_dataset)

        # Reinit model
        if not eval(args.iteratively_train_model):
            model = AutoModelForSequenceClassification.from_pretrained(args.model_name, config=config)
            trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset)

        # Train the model
        print('Start training the model')
        print('=' * 80)
        print({'num_examples': len(trainer.train_dataset),
               'batch': trainer.args.per_device_train_batch_size,
               'epochs': trainer.args.num_train_epochs,
               'grad_accum': trainer.args.gradient_accumulation_steps,
               'warmup': trainer.args.warmup_steps}.__str__())
        seed_everything(args.seed)
        trainer.train()
        trainer.model.save_pretrained(current_iteration_output_path)
        tokenizer.save_pretrained(current_iteration_output_path)
        print('=' * 80)

        # Eval on test set
        print('\nEvaluate on the test set')
        _, _, _, preds_to_save = evaluate(trainer, dataset=test_dataset)

        # Save the flattened preds
        with open(os.path.join(current_iteration_output_path, 'flattened_preds.pkl'), 'wb') as f:
            pickle.dump(preds_to_save, f)

    # Print all arguments
    print('-' * 80)
    print('All the arguments used: ')
    for key, value in args.__dict__.items():
        print(f"{key}: {value}")
