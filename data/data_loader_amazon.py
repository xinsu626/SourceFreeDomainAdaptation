import os


class AmazonDataLoader(object):
    def __init__(self, corpus_path):
        self.corpus_path = corpus_path

    @staticmethod
    def read_examples(path):
        examples = []
        labels = []

        with open(path, 'r') as f:
            for line in f:
                split_line = line.split('\t')
                examples.append(split_line[1].replace('\n', ''))
                labels.append(int(split_line[0]))

        return examples, labels

    def read_domain_data(self, domain):
        domain_path_train = os.path.join(self.corpus_path, f"{domain}/train")
        domain_path_test = os.path.join(self.corpus_path, f"{domain}/test")
        train_examples, train_labels = self.read_examples(domain_path_train)
        test_examples, test_labels = self.read_examples(domain_path_test)

        return train_examples, train_labels, test_examples, test_labels
