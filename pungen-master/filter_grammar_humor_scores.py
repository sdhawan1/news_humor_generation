# Run this file after scoring
# Use this code to keep only the best scoring alternate headlines.

import csv
import numpy as np


def write_best_lines(rows, joke_writer):
    grammar_scores = [r[5] for r in rows]
    ratio_scores = [r[6] for r in rows]
    gind = np.argmax(grammar_scores)
    rind = np.argmax(ratio_scores)
    # write gth sentence
    sent_id, sent, mod_word, orig_word, wid, orig_verb, grammar, ratio, _, _, _ = rows[gind]
    joke_writer.writerow([sent_id, sent, mod_word, orig_word, wid, orig_verb, grammar, ratio])
    # write rth sentence.
    if rind != gind:
        sent_id, sent, mod_word, orig_word, wid, orig_verb, grammar, ratio, _, _, _ = rows[rind]
        joke_writer.writerow([sent_id, sent, mod_word, orig_word, wid, orig_verb, grammar, ratio])


def filter_jokes(input_path, output_path):
    rows = []
    with open(output_path, 'w') as fout:
        joke_writer = csv.writer(fout)
        joke_writer.writerow(['id', 'sentence', 'mod_word', 'orig_word', 'word_id', 'orig_verb',
                              'grammar_score', 'surprise_ratio'])
        with open(input_path) as f:
            joke_reader = csv.reader(f)
            next(joke_reader)
            prev_id = None
            '''
            ['id', 'sentence', 'mod_word', 'orig_word', 'word_id', 'orig_verb',
                                             'grammar', 'ratio', 'local_exp', 'global_exp', 'orig_sent_grammar']
            '''
            for line in joke_reader:
                sent_id = line[0]
                rows.append(line)
                if prev_id is None:
                    prev_id = sent_id
                elif prev_id and (prev_id != sent_id):
                    write_best_lines(rows, joke_writer)
                    rows = []
                    prev_id = sent_id


# --------------------------
# Now, run the above code.
# --------------------------

input_path = 'scorer_output.csv'
output_path = 'filtered_scorer_output.csv'
filter_jokes(input_path, output_path)