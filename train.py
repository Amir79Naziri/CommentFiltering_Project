import re
from random import shuffle


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
        line = '<s> ' + line + ' </s>'
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

    pos_data = read_dataSet('DataSet/rt-polarity.neg')
    neg_data = read_dataSet('DataSet/rt-polarity.pos')

    pos_cutoff = int(0.95 * len(pos_data))
    neg_cutoff = int(0.95 * len(neg_data))

    pos_train_set = pos_data[0:pos_cutoff]
    neg_train_set = neg_data[0:neg_cutoff]

    pos_uniGram_dict, pos_biGram_dict = create_dict(pos_train_set)
    neg_uniGram_dict, neg_biGram_dict = create_dict(neg_train_set)

    test_set = pos_data[pos_cutoff:-1] + neg_data[neg_cutoff:-1]
    shuffle(test_set)



if __name__ == '__main__':
    # d = create_dict(read_dataSet('DataSet/rt-polarity.neg'))
    # print(len(d))

    t = {('a', 'b'): 0, ('c', 'd'): 0, ('a', 'b'): 0, ('e', 'f'): 0, ('a', 'b'): 0}

    print(('x', 'f') in t)
    print(t)
