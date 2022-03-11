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
from scipy.special import softmax
from scipy.stats import entropy
from sklearn.metrics import f1_score
import numpy as np

from data.data_loader_time import TimeDataLoader
from utils import seed_everything, time_performance
from time_augmenter import Augmenter
from time_active_learning_pool import FOOD_POOL, NEWS_POOL

parser = argparse.ArgumentParser()

# raw text data paths
parser.add_argument('--test_data_path', type=str)
parser.add_argument('--dev_data_path', type=str)
parser.add_argument('--domain', type=str)

# Checkpoints
parser.add_argument('--output_path', type=str)

# Model and tokenizer names
parser.add_argument('--model_name', type=str, default='clulab/roberta-timex-semeval')
parser.add_argument('--tokenizer_name', type=str, default='clulab/roberta-timex-semeval')
parser.add_argument('--config_name', type=str, default='clulab/roberta-timex-semeval')

# MISC
parser.add_argument('--seed', type=int, default=42)

# Tunable
parser.add_argument('--iteratively_train_model', type=str, default='False')
parser.add_argument('--iteratively_build_train_data', type=str, default='True')
parser.add_argument('--do_augmentation', type=str, default='True')
parser.add_argument('--num_new_examples_to_augment', type=int, default=5)
parser.add_argument('--active_learning_iters', type=int, default=12)
parser.add_argument('--active_learning_size', type=int, default=8)
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


def evaluate_predictions(trained_trainer: Trainer,
                         dataset: TimeDataset):
    pred_logits = trained_trainer.predict(test_dataset=dataset).predictions
    pred_label_ids = np.argmax(pred_logits, axis=2)
    true_label_ids = dataset.label_ids

    # Flatten the list
    pred_label_ids = [item for sublist in pred_label_ids for item in sublist]
    true_label_ids = [item for sublist in true_label_ids for item in sublist]

    # # Convert the -100 in true labels to 0
    # true_labels = [i if i != -100 else 0 for i in true_labels]

    pred_label_ids_to_use = []
    true_label_ids_to_use = []

    for p, t in zip(pred_label_ids, true_label_ids):
        if t != -100:
            pred_label_ids_to_use.append(p)
            true_label_ids_to_use.append(t)

    # Calculate the F1 score
    f1 = f1_score(y_true=true_label_ids_to_use, y_pred=pred_label_ids_to_use, average='micro')

    print(f"Prediction's micro F1 = {f1}")


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

    return f1, precision, recall, pred_logits


if __name__ == "__main__":
    # Disable multiprocessing.
    os.environ['TOKENIZERS_PARALLELISM'] = "false"

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

    # Load the model, config and tokenizer
    # Load the source-domain model.
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
    trainer = Trainer(model=model, args=training_args)

    # Evaluate the source-domain model on test set
    print('-' * 80)
    print('Evaluate the source-domain model on test set： ')
    source_f1, source_p, source_r, preds_to_save = evaluate(trained_trainer=trainer,
                                                            dataset=test_dataset,
                                                            pred_data_path=os.path.join(args.output_path, 'source-model-preds'),
                                                            gold_data_path=args.test_data_path,
                                                            data_provider=test_set_provider)
    # Save the flattened preds
    with open(os.path.join(args.output_path, 'flattened_source_preds.pkl'), 'wb') as f:
        pickle.dump(preds_to_save, f)

    # Active learning
    print("-" * 80)
    print('Start doing active learning on the dev set.')
    train_input_ids = []
    train_attention_mask = []
    train_label_ids = []

    for iter_idx in range(args.active_learning_iters):
        print('\n\n')
        print('-' * 40)
        print(f"Active Learning Iteration {iter_idx + 1}")

        # Build the dev dataset
        dev_dataset = TimeDataset(input_ids=dev_input_ids,
                                  attention_mask=dev_attention_mask,
                                  offset_mapping=dev_offset_mapping,
                                  doc_indices=dev_doc_indices)

        # Use the current model to predict the dev dataset
        pred_dev_logits = trainer.predict(test_dataset=dev_dataset).predictions
        pred_dev_probas = softmax(pred_dev_logits, axis=2)

        # Get the entropy
        pred_dev_entropy = entropy(pred_dev_probas, axis=2)

        # Get the special tokens masks
        special_tokens_mask = np.array(
            [tokenizer.get_special_tokens_mask(i.tolist(), already_has_special_tokens=True) for i in
             dev_input_ids])
        special_tokens_mask = 1 - special_tokens_mask

        # Count the actual length of each sentence
        sents_length = [i.count(1) for i in special_tokens_mask.tolist()]

        # Get the average entropy for each sentence
        pred_dev_entropy_avg = []
        for sent_length, entropy_list in zip(sents_length, pred_dev_entropy.tolist()):
            pred_dev_entropy_avg.append(sum(entropy_list[1:sent_length + 1]) / sent_length)

        # Sort the sentences with average entropy
        sorted_indices = np.argsort(pred_dev_entropy_avg)[::-1][:args.active_learning_size]

        # Select the examples and add them to
        selected_input_ids = [dev_input_ids[i] for i in sorted_indices]
        selected_attention_mask = [dev_attention_mask[i] for i in sorted_indices]
        selected_label_ids = [dev_label_ids[i] for i in sorted_indices]

        # Iteratively build up the dataset
        if eval(args.iteratively_build_train_data):
            # Add them to train set
            train_input_ids += selected_input_ids
            train_attention_mask += selected_attention_mask
            train_label_ids += selected_label_ids

            # Remove them from dev set
            dev_input_ids = [i for j, i in enumerate(dev_input_ids) if j not in sorted_indices]
            dev_attention_mask = [i for j, i in enumerate(dev_attention_mask) if j not in sorted_indices]
            dev_label_ids = [i for j, i in enumerate(dev_label_ids) if j not in sorted_indices]
        else:
            train_input_ids = selected_input_ids
            train_attention_mask = selected_attention_mask
            train_label_ids = selected_label_ids

        # Do data augmentation
        if eval(args.do_augmentation):
            print('Start doing data augmentation.')
            print(f"num_new_examples_to_augment = {args.num_new_examples_to_augment}")

            # Get the pool to use
            if args.domain == 'food':
                pool_to_use = FOOD_POOL
            elif args.domain == 'news':
                pool_to_use = NEWS_POOL
            else:
                raise ValueError('The domain input is not defined.')

            augmenter = Augmenter(tokenizer=tokenizer,
                                  config=config,
                                  entities_types_pool=pool_to_use,
                                  num_new_examples=args.num_new_examples_to_augment)
            new_examples_features = augmenter.augment(selected_input_ids,
                                                      selected_label_ids)
            if new_examples_features is not None:
                train_input_ids += new_examples_features['input_ids']
                train_attention_mask += new_examples_features['attention_mask']
                train_label_ids += new_examples_features['label_ids']

                # Augmentation: Align the augmented examples with current train examples.
                assembled_examples_features = augmenter.assemble_examples(train_input_ids,
                                                                          train_attention_mask,
                                                                          train_label_ids)
                train_input_ids = assembled_examples_features['input_ids']
                train_attention_mask = assembled_examples_features['attention_mask']
                train_label_ids = assembled_examples_features['label_ids']

        # Make the training set
        train_dataset = TimeDataset(input_ids=torch.stack(train_input_ids),
                                    attention_mask=torch.stack(train_attention_mask),
                                    label_ids=torch.stack(train_label_ids))

        # Update the output path
        current_iteration_output_path = os.path.join(args.output_path,
                                                     str(iter_idx))
        os.makedirs(current_iteration_output_path, exist_ok=True)
        training_args.output_dir = current_iteration_output_path

        # Update the trainer
        trainer = Trainer(model=trainer.model, args=training_args, train_dataset=train_dataset)

        # Reinit the model
        if not eval(args.iteratively_train_model):
            model = AutoModelForTokenClassification.from_pretrained(args.model_name,
                                                                    config=config)
            trainer = Trainer(model=model,
                              args=training_args,
                              train_dataset=train_dataset)

        # Start training
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

        print('Evaluate the current model on test set')
        pred_data_path = os.path.join(current_iteration_output_path,
                                      'preds')
        _, _, _, preds_to_save = evaluate(trained_trainer=trainer,
                                          dataset=test_dataset,
                                          pred_data_path=pred_data_path,
                                          gold_data_path=args.test_data_path,
                                          data_provider=test_set_provider)
        # Save the flattened preds
        with open(os.path.join(current_iteration_output_path, 'pred_logits.pkl'), 'wb') as f:
            pickle.dump(preds_to_save, f)

    # Print out all the settings for future reference
    print('-' * 80)
    print('Settings: ')
    for key, value in args.__dict__.items():
        print(f"{key}: {value}")
