import os

from transformers import PreTrainedTokenizer


class ActiveLearningTool:
    def __init__(self, tokenizer: PreTrainedTokenizer):
        self.tokenizer = tokenizer

    @staticmethod
    def get_user_inputs(prompt, valid_inputs):
        while True:
            result = input(prompt)
            if result not in valid_inputs:
                print('Invalid input!')
                continue
            else:
                break
        return result

    @staticmethod
    def convert_tensors_to_lists(tensor):

        if hasattr(tensor, 'tolist'):
            tensor = tensor.tolist()

        if hasattr(tensor[0], 'tolist'):
            tensor = [t.tolist() for t in tensor]

        return tensor

    def get_annotation(self,
                       all_input_ids,
                       all_pred_label_ids,
                       all_pred_entropies,
                       path2save_model,
                       iter_idx):

        # Initialize a list to store manually labeled labels ids.
        # The initial value should be predicted labels.
        all_label_ids = all_pred_label_ids  # num_examples

        # Initialize a list to store sentences' strings.
        all_sentence_strings = []

        # Get the valid labels list to prevent wrong label from human annotator.
        valid_label_ids = {0, 1}

        # Loop through each of the sentences
        for sentenceIndex, (input_ids, pred_label_id, pred_entropy) in enumerate(zip(
                self.convert_tensors_to_lists(all_input_ids),
                self.convert_tensors_to_lists(all_pred_label_ids),
                self.convert_tensors_to_lists(all_pred_entropies))
        ):
            # Convert the token ids to a sentence string.
            sentence_string = self.tokenizer.convert_tokens_to_string(
                self.tokenizer.convert_ids_to_tokens(input_ids, skip_special_tokens=True)
            )
            all_sentence_strings.append(sentence_string)

            # Print out the
            print("Index: {} | {} | Pred Label: {} | Entropy: {}".format(sentenceIndex,
                                                                         sentence_string,
                                                                         pred_label_id,
                                                                         pred_entropy))
            # Star doing annotation here.
            print("\nAnnotating Sentence {}".format(sentenceIndex + 1))
            print('=' * 80)

            # Ask if the user want to do annotation.
            do_annotation = self.get_user_inputs('Do annotation (Y or N)?',
                                                 valid_inputs={'Y', 'N'})

            # Do annotation.
            while {'Y': True, 'N': False}[do_annotation]:
                # Get the user input here.
                sentence_index_2_annotate = int(self.get_user_inputs('Sentence index to annotate: ',
                                                                     set([str(i) for i in range(len(all_input_ids))])))
                annotation = self.get_user_inputs("Annotation: ",
                                                  valid_label_ids)

                # Add user annotation
                all_label_ids[sentence_index_2_annotate] = annotation

                # Print current annotation
                print('current annotation: ')
                print(all_label_ids)

                # Ask if the user want to do annotation again.
                do_annotation = self.get_user_inputs('Do annotation (Y or N)?',
                                                     valid_inputs={'Y', 'N'})

                if not {'Y': True, 'N': False}[do_annotation]:
                    break

        # Finished the annotation.
        print('-' * 80)

        # Print out the labels.
        print("Finished annotation. Current annotation is : ")
        print(all_label_ids)

        # Save the sentence and the predictions
        # Get the path to save.
        path = os.path.join(path2save_model, 'sentence-annotated', str(iter_idx))

        # Make the directory.
        os.makedirs(path, exist_ok=True)

        # Save them.
        with open(os.path.join(path, 'sentences.txt'), 'w') as f:
            for s in all_sentence_strings:
                f.write(s)
                f.write('\n')
        with open(os.path.join(path, 'annotations.txt'), 'w') as f:
            for a in all_label_ids:
                f.write(str(a))
                f.write('\n')
        return all_label_ids
