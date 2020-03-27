
from __future__ import print_function # for Python3.x compatibility

import numpy as np

from math import log





# INPUT / OUTPUT



def read_patient_ids():

    with open('stage1_sample_submission.csv') as f:

        lines = f.readlines()[1:]

        return [line.split(',')[0] for line in lines]



def prob_format(p):

    return '%e' % p



def truncate(p):

    return float(prob_format(p))



def write_submit(patient_ids, probs, file_name):

    assert len(patient_ids) == len(probs)

    with open(file_name, 'w') as f:

        f.write('id,cancer\n')

        for i, p in zip(patient_ids, probs):

            f.write('%s,%s\n' % (i, prob_format(p)))

    print('wrote %s' % file_name)



def read_scores():

    lines = open('scores.txt').readlines()

    return [s.strip() for s in lines]





# PROBABILITIES



def build_template(n, chunk_size):

    epsilon = 1.05e-5

    return 1 / (1 + np.exp(n * epsilon * 2 ** np.arange(chunk_size)))



def build_probs(n, chunk, template):

    assert template.shape == chunk.shape

    probs = np.zeros((n,))

    probs[:] = 0.5

    probs[chunk] = template

    return probs





# LABEL INFERENCE



def int_to_bin(x, size):

    s = bin(x)[2:][::-1].ljust(size, '0')

    return np.array([int(c) for c in s])



def update_labels(labels, chunk, template, score):

    assert template.shape == chunk.shape

    chunk_size = len(chunk)

    n = len(labels)

    match_count = 0

    for i in range(2**chunk_size):

        b = int_to_bin(i, chunk_size)

        score_i = ((-np.log(template) * b - np.log(1-template) * (1-b)).sum() \

                   - log(0.5) * (n-chunk_size))/n

        if score == ('%.5f' % score_i):

            match_count += 1

            new_labels = b

    assert match_count == 1 # no collisions

    print('new labels: %s' % new_labels)

    labels[chunk] = new_labels





# MAIN



def write_submit_files():

    n = 198

    np.random.seed(2017)

    idx = np.arange(n)

    np.random.shuffle(idx) # optional

    chunk_size = 15

    template = build_template(n, chunk_size)

    template = np.array([truncate(x) for x in template])



    scores = read_scores()

    labels = np.zeros((n,), dtype=np.int)

    labels[:] = -1



    patient_ids = read_patient_ids()

    chunks = [idx[i : i + chunk_size] for i in range(0, len(idx), chunk_size)]

    for i, chunk in enumerate(chunks):

        t = template[:len(chunk)]

        probs = build_probs(n, chunk, t)

        write_submit(patient_ids, probs, 'submissions/submission_%02d.csv' % i)

        if i < len(scores):

            update_labels(labels, chunk, t, scores[i])

            if i+1 == len(chunks):

                write_submit(patient_ids, labels, 'submissions/submission_fin.csv')



if __name__ == '__main__':

    write_submit_files()