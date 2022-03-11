from typing import List
import time
import warnings
import random

from googletrans import Translator


class EntityReplacementAugmenter:
    def __init__(self,
                 num_new_examples: int,
                 entity_pool: List[str] = None):
        self.num_new_examples = num_new_examples
        self.entity_pool = entity_pool

    def augment_one_example(self, example, label_id):

        # New example
        new_examples = []

        # get current entity
        entity, start, end = self.get_entity(example)

        # remove it from pool
        current_entity_pool = list(filter(entity.__ne__, self.entity_pool))

        # get the entity start and end
        if len(current_entity_pool) < 1:
            warnings.warn("The pool size of the entity must be greater than or equal to 1."
                          "Will return None")
            return

        if self.num_new_examples > len(current_entity_pool):
            warnings.warn("The num_new_examples is greater than pool size. The actual number of new examples "
                          "will smaller than expected.")

        # Get new entity
        new_entities = random.sample(current_entity_pool,
                                     self.num_new_examples if len(
                                         current_entity_pool) >= self.num_new_examples else len(
                                         current_entity_pool))

        # Add the new examples
        for new_entity in new_entities:
            first_part = example[:start + 3]
            second_part = example[end:]

            new_examples.append(' '.join([first_part, new_entity, second_part]))

        return new_examples, [label_id] * len(new_examples)

    def augment(self,
                examples,
                label_ids):
        all_new_examples = []
        all_label_ids = []
        for e, l in zip(examples, label_ids):
            res = self.augment_one_example(e, l)
            if res:
                all_new_examples += res[0]
                all_label_ids += res[1]
        return all_new_examples, all_label_ids

    def get_all_entities(self, examples):
        entities = []
        for k in examples:
            entities.append(self.get_entity(k)[0])

        return list(set(entities))

    @staticmethod
    def get_entity(text):
        start = text.index('<e>')
        end = text.index('</e>')
        entity = text[start + 3:end]
        return entity, start, end


class BackTranslationAugmenter:
    def __init__(self,
                 languages: List[str]):
        self.translator = Translator()
        self.languages = languages

    def back_translate(self, text) -> List[str]:
        """Perform back translation for one sentence."""
        back_translated_sentences = []
        for language in self.languages:
            target_text = self.translator.translate(text=text, src='en', dest=language).text
            time.sleep(5)  # Handle the google translate timeout problem
            translated_back = self.translator.translate(text=target_text,
                                                        src=language,
                                                        dest='en').text

            back_translated_sentences.append(translated_back)
        return back_translated_sentences

    def augment(self,
                examples: List[str],
                label_ids: List[int]):
        """Get news examples from multiple examples"""

        # Initialize two lists to store augmented examples' tokens and labels.
        new_examples = []
        new_label_ids = []

        for example, label_id in zip(examples, label_ids):
            augmented_examples = self.back_translate(example)
            augmented_label_ids = [label_id] * len(augmented_examples)

            new_examples += augmented_examples
            new_label_ids += augmented_label_ids

        return new_examples, new_label_ids