import re
from random import shuffle

POS_BiGRAM_DICT = dict()
POS_UniGRAM_DICT = dict()
NEG_BiGRAM_DICT = dict()
NEG_UniGRAM_DICT = dict()


def read_dataSet(address):
    try:
        with open(address, 'rt') as file:
            lines = file.readlines()
            return [_[:-1] for _ in lines]
    except IOError:
        print('some thing went wrong in loading DataSet')


def create_dict(lines):
    uniGram_dictionary = dict()
    biGram_dictionary = dict()

    for line in lines:
        line = re.sub(r'[^A-Za-z0-9]+', ' ', line)
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


def train():

    global POS_BiGRAM_DICT, POS_UniGRAM_DICT, NEG_BiGRAM_DICT, NEG_UniGRAM_DICT

    pos_data = read_dataSet('DataSet/rt-polarity.pos')
    neg_data = read_dataSet('DataSet/rt-polarity.neg')

    pos_cutoff = int(0.95 * len(pos_data))
    neg_cutoff = int(0.95 * len(neg_data))

    pos_train_set = pos_data[0:pos_cutoff]
    neg_train_set = neg_data[0:neg_cutoff]

    POS_UniGRAM_DICT, POS_BiGRAM_DICT = create_dict(pos_train_set)
    NEG_UniGRAM_DICT, NEG_BiGRAM_DICT = create_dict(neg_train_set)

    test_set = pos_data[pos_cutoff:-1] + neg_data[neg_cutoff:-1]
    shuffle(test_set)

    return test_set


def test(test_set):

    global POS_BiGRAM_DICT, POS_UniGRAM_DICT, NEG_BiGRAM_DICT, NEG_UniGRAM_DICT

    for line in test_set:
        print(line)
        pos_prob = 0.5 * probability(POS_UniGRAM_DICT, POS_BiGRAM_DICT, line)
        neg_prob = 0.5 * probability(NEG_UniGRAM_DICT, NEG_BiGRAM_DICT, line)
        if pos_prob > neg_prob:
            print("not filter this")
        else:
            print("filter this")


def main():
    global POS_BiGRAM_DICT, POS_UniGRAM_DICT, NEG_BiGRAM_DICT, NEG_UniGRAM_DICT

    while True:
        line = input()
        if line == '!q':
            return
        pos_prob = 0.5 * probability(POS_UniGRAM_DICT, POS_BiGRAM_DICT, line)
        neg_prob = 0.5 * probability(NEG_UniGRAM_DICT, NEG_BiGRAM_DICT, line)
        if pos_prob > neg_prob:
            print("not filter this")
        else:
            print("filter this")


def probability(uni_dictionary, bi_dictionary, line):
    l1 = 0.85
    l2 = 0.10
    l3 = 0.05
    line = re.sub(r'[^A-Za-z0-9]+', ' ', line)
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


if __name__ == '__main__':
    test_ = train()
    main()
