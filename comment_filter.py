import re


class Model:
    def __init__(self, pos_dict, neg_dict):
        self.pos_uniGram_dict = pos_dict[0]
        self.pos_biGram_dict = pos_dict[1]
        self.neg_uniGram_dict = neg_dict[0]
        self.neg_biGram_dict = neg_dict[1]

    def estimate(self, line):
        pos_prob = 0.5 * probability(self.pos_uniGram_dict, self.pos_biGram_dict, line)
        neg_prob = 0.5 * probability(self.neg_uniGram_dict, self.neg_biGram_dict, line)

        return pos_prob, neg_prob


def probability(uni_dictionary, bi_dictionary, line):
    l1 = 0.95
    l2 = 0.045
    l3 = 0.005
    words = line.split()
    try:
        prob = uni_dictionary[line[0]] / len(uni_dictionary)
    except KeyError:
        prob = l3 * 0.5

    for i in range(1, len(words)):
        try:
            biGram_prob = (bi_dictionary[(words[i - 1], words[i])] / uni_dictionary[words[i - 1]])
        except KeyError:
            biGram_prob = 0
        try:
            uniGram_prob = (uni_dictionary[words[i]] / len(uni_dictionary))
        except KeyError:
            uniGram_prob = 0
        interpolation_prob = l1 * biGram_prob + l2 * uniGram_prob + l3 * 0.5
        prob *= interpolation_prob

    return prob


def read_dataSet(address):
    try:
        with open(address, 'rt') as file:
            data = file.read(-1)
            return data
    except IOError:
        print('some thing went wrong in loading DataSet')


def preprocess():
    raw_pos_data = read_dataSet('DataSet/rt-polarity.pos')
    raw_neg_data = read_dataSet('DataSet/rt-polarity.neg')

    pos_lines = re.sub(r'[^A-Za-z0-9\n]+', ' ', raw_pos_data).split('\n')
    neg_lines = re.sub(r'[^A-Za-z0-9\n]+', ' ', raw_neg_data).split('\n')

    pos_cutoff = int(0.98 * len(pos_lines))
    neg_cutoff = int(0.98 * len(neg_lines))

    pos_train_set = pos_lines[0:pos_cutoff]
    pos_test_set = pos_lines[pos_cutoff:-1]

    neg_train_set = neg_lines[0:neg_cutoff]
    neg_test_set = neg_lines[neg_cutoff:-1]

    return (pos_train_set, pos_test_set), (neg_train_set, neg_test_set)


def train(pos_set, neg_set):
    def create_dict(lines):
        uniGram_dictionary = dict()
        biGram_dictionary = dict()

        for line in lines:
            words = line.split()
            for i in range(len(words)):
                if words[i] not in uniGram_dictionary:
                    uniGram_dictionary[words[i]] = 1
                else:
                    uniGram_dictionary[words[i]] += 1

                if i > 0:
                    if (words[i - 1], words[i]) not in biGram_dictionary:
                        biGram_dictionary[(words[i - 1], words[i])] = 1
                    else:
                        biGram_dictionary[(words[i - 1], words[i])] += 1

        return uniGram_dictionary, biGram_dictionary

    return Model(create_dict(pos_set), create_dict(neg_set))


def evaluate(model, pos_test_set, neg_test_set):
    TP = TN = FP = FN = 0
    for line in pos_test_set:
        pos_prob, neg_prob = model.estimate(line)
        if pos_prob > neg_prob:
            TP += 1
        else:
            FN += 1

    for line in neg_test_set:
        pos_prob, neg_prob = model.estimate(line)
        if pos_prob > neg_prob:
            FP += 1
        else:
            TN += 1

    recall = TP / (TP + FN)
    precision = TP / (TP + FP)
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    F1_score = (2 * precision * recall) / (precision + recall)
    return recall, precision, accuracy, F1_score


def run(model):
    while True:
        line = input()
        if line == '!q':
            return
        line = re.sub(r'[^A-Za-z0-9]+', ' ', line)
        pos_prob, neg_prob = model.estimate(line)
        if pos_prob > neg_prob:
            print("not filter this")
        else:
            print("filter this")


def main():
    (pos_train_set, pos_test_set), (neg_train_set, neg_test_set) = preprocess()
    my_model = train(pos_train_set, neg_train_set)
    # run(my_model)
    recall, precision, accuracy, F1_score = evaluate(my_model, pos_test_set, neg_test_set)
    print(recall, precision, accuracy, F1_score)


if __name__ == '__main__':
    main()
