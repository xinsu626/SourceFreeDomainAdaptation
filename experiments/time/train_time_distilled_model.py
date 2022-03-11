import os
import argparse
from copy import deepcopy

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
parser.add_argument('--dev_data_path', type=str)

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
parser.add_argument('--per_gpu_eval_batch_size', type=int, default=2)
parser.add_argument('--max_length', type=int, default=271)
parser.add_argument('--grad_accum_steps', type=int, default=1)
parser.add_argument('--epochs', type=int, default=3)
parser.add_argument('--warmup_steps', type=float, default=500)
parser.add_argument('--weight_decay', type=float, default=0.01)
parser.add_argument('--max_grad_norm', type=float, default=1.0)
parser.add_argument('--adam_epsilon', type=float, default=1e-08)
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

    # Get the predictions to save
    avg_predictions = np.mean(pred_logits, axis=2)
    flattened_avg_predictions = avg_predictions.flatten()

    return f1, precision, recall, flattened_avg_predictions


if __name__ == "__main__":
    # Disable multiprocessing.
    os.environ['TOKENIZERS_PARALLELISM'] = "false"

    # Parse the command line arguments.
    args = parser.parse_args()

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
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=True)
    source_domain_model = AutoModelForTokenClassification.from_pretrained(args.model_name, config=config)

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
    dev_input_ids, dev_attention_mask, dev_offset_mapping, dev_doc_indices, dev_labels = \
        dev_set_provider.convert_documents_to_features(return_labels=True)
    test_input_ids, test_attention_mask, test_offset_mapping, test_doc_indices, test_labels = \
        test_set_provider.convert_documents_to_features(return_labels=True)
    print(f'Number of sentences in dev = {len(dev_input_ids)}')
    print(f'Number of sentences in test = {len(test_input_ids)}')

    # Make the dataset objects
    dev_dataset = TimeDataset(input_ids=dev_input_ids,
                              attention_mask=dev_attention_mask,
                              label_ids=dev_labels,
                              offset_mapping=dev_offset_mapping,
                              doc_indices=dev_doc_indices)

    test_dataset = TimeDataset(input_ids=test_input_ids,
                               attention_mask=test_attention_mask,
                               label_ids=test_labels,
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
                                      adam_epsilon=args.adam_epsilon,
                                      max_grad_norm=args.max_grad_norm,
                                      weight_decay=args.weight_decay)

    # Build the trainer
    source_domain_model_trainer = Trainer(model=source_domain_model, args=training_args)

    # Evaluate the source-domain model on test set
    print('-' * 80)
    print('Evaluate the source-domain model on test set： ')
    source_f1, source_p, source_r, preds_to_save = evaluate(trained_trainer=source_domain_model_trainer,
                                                            dataset=test_dataset,
                                                            pred_data_path=os.path.join(args.output_path, 'source-model-preds'),
                                                            gold_data_path=args.test_data_path,
                                                            data_provider=test_set_provider)

    # Run source domain model on the dev set
    print('-' * 80)
    print("Prepare the distillation data")
    print("Run source-domain model on the dev set to get the pseudo labels")
    pred_dev_logits = source_domain_model_trainer.predict(test_dataset=dev_dataset).predictions
    pred_dev_labels = np.argmax(pred_dev_logits, axis=2)

    distillation_input_ids = deepcopy(dev_input_ids)
    distillation_attention_mask = deepcopy(dev_attention_mask)
    distillation_labels = deepcopy(pred_dev_labels)

    print(f"Number of sentences used for distillation = {len(distillation_input_ids)}")

    distillation_dataset = TimeDataset(input_ids=distillation_input_ids,
                                       attention_mask=distillation_attention_mask,
                                       label_ids=distillation_labels,
                                       offset_mapping=dev_offset_mapping,
                                       doc_indices=dev_doc_indices)

    # Student model
    print('Load a roberta-base model as a student model.')
    roberta_model = AutoModelForTokenClassification.from_pretrained('roberta-base', config=config)
    print('\n')

    roberta_trainer = Trainer(model=roberta_model,
                              args=training_args,
                              train_dataset=distillation_dataset)

    # Train the model
    print('\n')
    seed_everything(args.seed)
    print('Start training the model')
    print('=' * 80)
    print({'num_examples': len(distillation_dataset),
           'batch': roberta_trainer.args.per_device_train_batch_size,
           'epochs': roberta_trainer.args.num_train_epochs,
           'grad_accum': roberta_trainer.args.gradient_accumulation_steps,
           'warmup': roberta_trainer.args.warmup_steps}.__str__())
    roberta_trainer.train()
    roberta_trainer.model.save_pretrained(args.output_path)
    tokenizer.save_pretrained(args.output_path)
    print('=' * 80)

    # Evaluate on the test set
    evaluate(trained_trainer=roberta_trainer,
             dataset=test_dataset,
             pred_data_path=os.path.join(args.output_path, 'distilled_model_preds'),
             gold_data_path=args.test_data_path,
             data_provider=test_set_provider)

    # Print out all the settings for future reference
    print('-' * 80)
    print('Settings: ')
    for key, value in args.__dict__.items():
        print(f"{key}: {value}")
