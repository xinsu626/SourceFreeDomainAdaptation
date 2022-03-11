import random
import warnings
from typing import List, Dict

import torch
import numpy as np
from transformers import PreTrainedTokenizer, PretrainedConfig


class Augmenter:
    def __init__(self,
                 entities_types_pool: Dict = None,
                 tokenizer: PreTrainedTokenizer = None,
                 config: PretrainedConfig = None,
                 num_new_examples: int = None):
        self.entities_types_pool = entities_types_pool
        self.tokenizer = tokenizer
        self.config = config
        self.num_new_examples = num_new_examples

    def convert_examples_to_features(self, examples, labels):
        # Get the max length.
        special_tokens_count = 2
        max_len = max([len(x) for x in examples]) + special_tokens_count

        all_input_ids = []  # num_examples, max_len
        all_label_ids = []  # num_examples, max_len
        all_attention_mask = []

        for example, label in zip(examples, labels):
            label_ids = [self.config.label2id[x] for x in label]
            # Add </s> token
            example = example + [self.tokenizer.sep_token]
            label_ids = label_ids + [-100]

            # Add <s> token
            example = [self.tokenizer.cls_token] + example
            label_ids = [-100] + label_ids

            # Convert tokens to ids.
            example_ids = self.tokenizer.convert_tokens_to_ids(example)

            # Get attention mask
            attention_mask = [1] * len(example_ids)

            # Zero-pad up to the sequence length.
            padding_length = max_len - len(example_ids)

            # Padding here.
            example_ids = example_ids + ([self.tokenizer.pad_token_id] * padding_length)
            attention_mask = attention_mask + ([0] * padding_length)
            label_ids = label_ids + ([-100] * padding_length)

            assert len(example_ids) == max_len
            assert len(attention_mask) == max_len
            assert len(label_ids) == max_len

            all_input_ids.append(torch.tensor(example_ids))
            all_attention_mask.append(torch.tensor(attention_mask))
            all_label_ids.append(torch.tensor(label_ids))
        results = {'input_ids': all_input_ids,
                   'attention_mask': all_attention_mask,
                   'label_ids': all_label_ids}
        return results

    @staticmethod
    def pad_tensor(tensor, dim, pad_id, max_len):
        padding_length = max_len - tensor.size()[dim]
        padding_tensor = torch.tensor([pad_id] * padding_length)

        return torch.cat((tensor, padding_tensor), dim=dim)

    def assemble_examples(self,
                          all_input_ids,
                          all_attention_mask,
                          all_label_ids):
        # Get the max length
        max_len = max([x.size()[0] for x in all_input_ids])

        # Pad input_ids with pad_token_id, attention_mask with 0, label_ids with -100
        pad_input_id = self.tokenizer.pad_token_id
        pad_label_id = -100
        pad_attention_mask_id = 0

        padded_all_input_ids = []
        padded_all_attention_mask = []
        padded_all_label_ids = []

        for input_ids, attention_mask, label_ids in zip(all_input_ids, all_attention_mask, all_label_ids):
            if input_ids.size()[0] < max_len:
                input_ids = self.pad_tensor(input_ids, 0, pad_input_id, max_len)
                attention_mask = self.pad_tensor(attention_mask, 0, pad_attention_mask_id, max_len)
                label_ids = self.pad_tensor(label_ids, 0, pad_label_id, max_len)
            padded_all_input_ids.append(input_ids)
            padded_all_attention_mask.append(attention_mask)
            padded_all_label_ids.append(label_ids)
        return {'input_ids': padded_all_input_ids,
                'attention_mask': padded_all_attention_mask,
                'label_ids': padded_all_label_ids}

    @staticmethod
    def check_start_of_entity_span(current_tag):
        start = False

        if current_tag == 'B':
            start = True

        return start

    @staticmethod
    def check_end_of_entity_span(current_tag):
        end = False

        if current_tag == 'B':
            end = True

        if current_tag == 'O':
            end = True
        return end

    def get_entities_with_spans(self, labels):
        entities_spans = []
        start_offset = 0
        prev_tag = 'O'
        prev_type = 'O'

        for pos, label in enumerate(labels + ['O']):  # add O to handle the entity is at last position
            current_tag = label[0]
            if len(label) > 1:
                current_type = label[2:]  # B-[type], I-[type]
            else:
                current_type = label  # O

            if self.check_end_of_entity_span(current_tag):
                if prev_tag != 'O':
                    entities_spans.append((start_offset, pos - 1, prev_type))
            if self.check_start_of_entity_span(current_tag):
                start_offset = pos
            prev_tag = current_tag
            prev_type = current_type
        return entities_spans

    def get_augmented_examples_from_one_sent(self, tokens, labels):
        new_sent_tokens = []
        new_sents_labels = []
        entities_spans = self.get_entities_with_spans(labels)

        # If the sentence does not have any entities
        # Return None.
        if len(entities_spans) == 0:
            return

        for entity_span in entities_spans:
            start, end, e_type = entity_span

            # Get the entity
            entity = ''.join(tokens[start:end + 1])

            # Test if there is a pool for this entity
            # If there is no pool, then next entity.
            if e_type not in self.entities_types_pool.keys():
                continue

                # Get current pool
            current_entity_pool = self.entities_types_pool[e_type]

            # Try to remove current entity from the pool.
            current_entity_pool = list(filter(entity.__ne__, current_entity_pool))

            # After removing, check if current pool size is valid.
            # If the pool is not valid anymore, then next entity.
            if len(current_entity_pool) < 1:
                warnings.warn("The pool size of the entity must be greater than or equal to 1."
                              "Will return None")
                continue

                # Check if sample_size is greater than pool size.
            if self.num_new_examples > len(current_entity_pool):
                warnings.warn("The num_new_examples is greater than pool size. The actual number of new examples "
                              "will smaller than expected.")

            new_entities = random.sample(current_entity_pool,
                                         self.num_new_examples if len(
                                             current_entity_pool) >= self.num_new_examples else len(
                                             current_entity_pool))

            # tokenize the new entities
            tokenized_new_entities = [self.tokenizer.tokenize(i) for i in new_entities]

            # Add the labels for these new entities
            new_entities_labels = []
            for x in tokenized_new_entities:
                if len(x) == 1:
                    new_entities_labels.append(['B-' + e_type])
                else:
                    tmp = ['B-' + e_type]
                    tmp += (len(x) - 1) * ['I-' + e_type]
                    new_entities_labels.append(tmp)

            # insert new entities to tokens
            for x, y in zip(tokenized_new_entities, new_entities_labels):
                new_sent_tokens.append(tokens[:start] + x + tokens[end + 1:])
                new_sents_labels.append(labels[:start] + y + labels[end + 1:])
        return {'new_sents_tokens': new_sent_tokens, 'new_sents_labels': new_sents_labels}

    def get_new_sents_ids_from_one_sent_ids(self, sent_token_ids, sent_label_ids):
        """Get the new augmented sentences from ONE sentence tokens' ids and labels ids."""

        # Covert the tensors to lists.
        sent_token_ids = sent_token_ids.tolist()
        sent_label_ids = sent_label_ids.tolist()

        # Remove the special tokens here.
        # Get the special tokens mask.
        special_tokens_masks = self.tokenizer.get_special_tokens_mask(sent_token_ids, already_has_special_tokens=True)

        # Remove special tokens
        non_specials = np.count_nonzero(np.array(special_tokens_masks) == 0)
        sent_token_ids = sent_token_ids[1: non_specials + 1]
        sent_label_ids = sent_label_ids[1: non_specials + 1]

        # Convert the token ids to token names.
        sent_tokens = self.tokenizer.convert_ids_to_tokens(sent_token_ids)

        # Convert label ids to label names; If -100, then O label.
        sent_labels = [self.config.id2label[i] if i != -100 else 'O' for i in sent_label_ids]

        # Return a dictionary or None.
        result = self.get_augmented_examples_from_one_sent(sent_tokens, sent_labels)
        return result

    def augment(self,
                examples_tokens_ids: List[torch.Tensor],
                examples_labels_ids: List[torch.Tensor]):
        """Get new examples from multiple examples."""
        # Initialize two lists to store augmented examples' tokens and labels.
        new_examples_tokens_names = []
        new_examples_labels_names = []

        # Do augmentation based on each of the example.
        for example_token_ids, example_label_ids in zip(examples_tokens_ids, examples_labels_ids):
            # Get multiple new examples from current example.
            res = self.get_new_sents_ids_from_one_sent_ids(example_token_ids, example_label_ids)

            # If the augmentation result is not None, then add it to the list.
            if res is not None:
                new_examples_tokens_names += res['new_sents_tokens']
                new_examples_labels_names += res['new_sents_labels']

        # Test if there is any new examples
        if len(new_examples_tokens_names) != 0:
            # Convert examples to features
            new_examples_features = self.convert_examples_to_features(new_examples_tokens_names,
                                                                      new_examples_labels_names)
        else:
            new_examples_features = None
        return new_examples_features
