import numpy as np
import re
import nltk, nltk.stem.porter


def process_email(email_contents):
    vocab_list = get_vocab_list()

    word_indices = np.array([], dtype=np.int64)

    # ===================== Preprocess Email =====================

    email_contents = email_contents.lower()

    email_contents = re.sub('<[^<>]+>', ' ', email_contents)

    # Any numbers get replaced with the string 'number'
    email_contents = re.sub('[0-9]+', 'number', email_contents)

    # Anything starting with http or https:// replaced with 'httpaddr'
    email_contents = re.sub('(http|https)://[^\s]*', 'httpaddr', email_contents)

    # Strings with "@" in the middle are considered emails --> 'emailaddr'
    email_contents = re.sub('[^\s]+@[^\s]+', 'emailaddr', email_contents)

    # The '$' sign gets replaced with 'dollar'
    email_contents = re.sub('[$]+', 'dollar', email_contents)

    # ===================== Tokenize Email =====================

    # Output the email
    print('==== Processed Email ====')

    stemmer = nltk.stem.porter.PorterStemmer()

    # print('email contents : {}'.format(email_contents))

    tokens = re.split('[@$/#.-:&*+=\[\]?!(){\},\'\">_<;% ]', email_contents)

    for token in tokens:
        token = re.sub('[^a-zA-Z0-9]', '', token)
        token = stemmer.stem(token)

        if len(token) < 1:
            continue

        # ===================== Your Code Here =====================
        # Instructions : Fill in this function to add the index of token to
        #                word_indices if it is in the vocabulary. At this point
        #                of the code, you have a stemmed word frome email in
        #                the variable token. You should look up token in the
        #                vocab_list. If a match exists, you should add the
        #                index of the word to the word_indices nparray.
        #                Concretely, if token == 'action', then you should
        #                look up the vocabulary list the find where in vocab_list
        #                'action' appears. For example, if vocab_list[18] == 'action'
        #                then you should add 18 to the word_indices array.



        # ==========================================================

        print(token)

    print('==================')

    return word_indices


def get_vocab_list():
    vocab_dict = {}
    with open('vocab.txt') as f:
        for line in f:
            (val, key) = line.split()
            vocab_dict[int(val)] = key

    return vocab_dict
