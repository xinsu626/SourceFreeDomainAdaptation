import csv


class DataLoaderNegation(object):

    def __init__(self, corpus_path, label_path=None) -> None:
        self.corpus_path = corpus_path
        self.label_path = label_path

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        with open(input_file, "r", encoding="utf-8-sig") as f:
            return list(csv.reader(f, delimiter="\t", quotechar=quotechar))

    def get_text_data(self):
        examples = []

        lines_example = self._read_tsv(self.corpus_path)
        for i in lines_example:
            examples.append(' '.join(i))

        outputs = examples

        if self.label_path is not None:
            labels = []
            lines_label = self._read_tsv(self.label_path)
            for i in lines_label:
                labels.append(int(i[0]))
            outputs = (examples, labels)

        return outputs

