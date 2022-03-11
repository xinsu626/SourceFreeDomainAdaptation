import os

import numpy as np
import anafora
from spacy.lang.en import English
from transformers import PreTrainedTokenizer, PretrainedConfig


class TimeDataLoader:
    def __init__(self,
                 corpus_dir,
                 config: PretrainedConfig,
                 tokenizer: PreTrainedTokenizer,
                 max_length: int = None) -> None:
        if not os.path.exists(corpus_dir):
            raise Exception("The {} directory does not exist.".format(
                corpus_dir
            ))
        self.corpus_dir = corpus_dir
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.config = config
        self.sentencizer = English()
        self.sentencizer.add_pipe(self.sentencizer.create_pipe("sentencizer"))

    @staticmethod
    def read_text_file(path):
        with open(path, "r") as f:
            text = f.read()
        return text

    def convert_sents_to_features(self,
                                  tokenized_sentences,
                                  sent_i,
                                  sent_offset,
                                  annotations=None):
        input_ids = tokenized_sentences['input_ids'][sent_i]
        attention_mask = tokenized_sentences["attention_mask"][sent_i]
        offset_mapping = tokenized_sentences["offset_mapping"][sent_i]

        labels = None
        if annotations is not None:
            labels = tokenized_sentences['labels'][sent_i]

        start_open = None
        for token_i, offset in enumerate(offset_mapping):
            start, end = offset.numpy()
            if start == end:
                continue
            start += sent_offset
            end += sent_offset
            offset_mapping[token_i][0] = start
            offset_mapping[token_i][1] = end

            # convert the annotations to labels
            if annotations is not None:
                if start_open is not None and annotations[start_open][0] <= start:
                    start_open = None
                if start_open is not None and start in annotations:
                    start_open = None
                elif start_open is not None:
                    annotation = annotations[start_open][1]
                    labels[token_i] = self.config.label2id["I-" + annotation]
                elif start in annotations:
                    annotation = annotations[start][1]
                    labels[token_i] = self.config.label2id["B-" + annotation]
                    start_open = start
                if start_open is not None and end == annotations[start_open][0]:
                    start_open = None

        return input_ids, attention_mask, offset_mapping, labels

    def convert_documents_to_features(self, return_labels=False, single_expression=False):
        docs_dir = anafora.walk(self.corpus_dir,
                                xml_name_regex=".*((?<![.].{3})|[.]txt)$")
        docs_dir = list(docs_dir)
        print("Number of documents = {}".format(len(docs_dir)))
        input_ids_all, attention_mask_all, offset_mapping_all, doc_indexes, labels_all =\
            [], [], [], [], []

        for files in docs_dir:
            # Current document start pos.
            doc_span_start = len(input_ids_all)
            doc_subdir, doc_name, doc_text_file_list = files
            if len(doc_text_file_list) != 1:
                raise Exception("Wrong number of text files in %s" % doc_subdir)
            # Text file full path.
            doc_text_file_dir = os.path.join(self.corpus_dir, doc_subdir, doc_text_file_list[0])
            with open(doc_text_file_dir) as f:
                text = f.read()
            # If the document only has one sentence, the no need to split.
            if single_expression:
                sents_in_doc = [text]
            else:
                # If it has more than one sentences, then split to sentence.
                doc_text = self.sentencizer(text)
                sents_in_doc = [sent.text_with_ws for sent in doc_text.sents]
            # Run tokenizer on sentences.
            tokenized_sents = self.tokenizer(sents_in_doc,
                                             return_tensors="pt",
                                             padding="max_length",
                                             truncation="longest_first",
                                             return_offsets_mapping=True,
                                             return_special_tokens_mask=True,
                                             max_length=self.max_length)

            annotations = None
            if return_labels:
                # Load the xml annotation file and get the annotations, and store the annotations into a dict.
                current_doc_dir = os.path.join(self.corpus_dir, doc_subdir)
                doc_xml_files_info = list(anafora.walk(current_doc_dir, xml_name_regex="[.]xml$"))
                doc_xml_subdir, doc_xml_name, doc_xml_files_list = doc_xml_files_info[0]
                if len(doc_xml_files_list) != 1:
                    raise Exception("There should be only one xml file.")
                doc_xml_file_dir = os.path.join(current_doc_dir, doc_xml_files_list[0])  # Get the xml file's full path.
                raw_annotations = anafora.AnaforaData.from_file(doc_xml_file_dir)  # Load the annotations.
                annotations = {}  # store the extracted annotations
                for a in raw_annotations.annotations:
                    label = a.type
                    for span in a.spans:
                        start, end = span
                        annotations[start] = (end, label)
                # Initialize labels matrix with masked special tokens; (num_sentences, max_len)
                tokenized_sents['labels'] = tokenized_sents['special_tokens_mask'] * self.config.label_pad_id

            sent_offset = 0
            for sent_i in range(len(sents_in_doc)):
                input_ids, attention_mask, offset_mapping, labels = \
                    self.convert_sents_to_features(tokenized_sents,
                                                   sent_i,
                                                   sent_offset,
                                                   annotations=annotations)
                sent_offset += len(sents_in_doc[sent_i])
                input_ids_all.append(input_ids)
                attention_mask_all.append(attention_mask)
                offset_mapping_all.append(offset_mapping)
                if labels is not None:
                    labels_all.append(labels)
            doc_indexes.append((doc_subdir, doc_span_start, len(input_ids_all)))

        outputs = (input_ids_all, attention_mask_all, offset_mapping_all, doc_indexes)

        if return_labels:
            outputs = (input_ids_all, attention_mask_all, offset_mapping_all, doc_indexes, labels_all)
        return outputs

    def write_anafora(self, output_dir, input_ids, offset_mapping, doc_indexes, predictions):
        def add_entity(data, doc_name, label, offset):
            entity_label = self.config.id2label[label] if label > 0 else None
            if entity_label is not None:
                anafora.AnaforaEntity()
                entity = anafora.AnaforaEntity()
                num_entities = len(data.xml.findall("annotations/entity"))
                entity.id = "%s@%s" % (num_entities, doc_name)
                entity.spans = ((offset[0], offset[1]),)
                entity.type = entity_label.replace("B-", "")
                data.annotations.append(entity)

        for doc_index in doc_indexes:
            doc_subdir, doc_start, doc_end = doc_index
            doc_name = os.path.basename(doc_subdir)
            doc_offset_mapping = offset_mapping[doc_start:doc_end]
            doc_input_ids = input_ids[doc_start:doc_end]
            doc_predictions = predictions[doc_start:doc_end]
            doc_predictions = np.argmax(doc_predictions, axis=2)
            data = anafora.AnaforaData()
            for sent_labels, sent_input_ids, sent_offset_mapping in zip(doc_predictions, doc_input_ids,
                                                                        doc_offset_mapping):
                # Remove padding and <s> </s>
                special_mask = self.tokenizer.get_special_tokens_mask(sent_input_ids, already_has_special_tokens=True)
                non_specials = np.count_nonzero(np.array(special_mask) == 0)
                sent_labels = sent_labels[1: non_specials + 1]
                sent_offsets = sent_offset_mapping[1: non_specials + 1]
                previous_label = 0
                previous_offset = [None, None]  # (start, end)
                for token_label, token_offset in zip(sent_labels, sent_offsets):
                    label_diff = token_label - previous_label
                    if token_label % 2 != 0:  # If odd number, it is B label
                        add_entity(data, doc_name, previous_label, previous_offset)
                        previous_label = token_label
                        previous_offset = token_offset
                    elif label_diff == 1:  # If even number and diff with previous is 1, it is I label
                        previous_offset[1] = token_offset[1]
                    elif previous_label > 0:  # If current is O label and previous not O we must write it.
                        add_entity(data, doc_name, previous_label, previous_offset)
                        previous_label = 0
                        previous_offset = [None, None]
                if previous_label > 0:  # If remaining previous not O we must write it.
                    add_entity(data, doc_name, previous_label, previous_offset)
            doc_path = os.path.join(output_dir, doc_subdir)
            os.makedirs(doc_path, exist_ok=True)
            doc_path = os.path.join(doc_path,
                                    "%s.TimeNorm.system.completed.xml" % doc_name)
            data.to_file(doc_path)
