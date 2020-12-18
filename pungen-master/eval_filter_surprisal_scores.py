"""
The goal:
    - Very simply... start with a test sentence and a replacement word.
    - Replace the replacement word with the new word.
    - Pass the original and modified sentence into the "surprisal" scorer and see what happens.
"""

import re
from pungen.scorer import SurprisalScorer, UnigramModel, LMScorer
import argparse
import csv


"""
    Needed arguments:
    - orig_sentence: the original sentence with the replacement word indicated.
    - word_id: the index of the word we want to replace.
    - rep_word: the word that we plan to replace.
    - lm_path: path for the language model (use "fairseq" abspath by default)
    - word_counts_path: path for the word counts of the "wikitext" model (also use "fairseq" dict by default)
    - local_window_size: ("local" area size) 2 by default
    - oov_prob: (???) 0.03 by default.
"""
def parse_args():
    parser = argparse.ArgumentParser(description="run the 'surprisal scoring' metric", conflict_handler="resolve")
    parser.add_argument("--orig-sentence", default="thousands of gay and bisexual men convicted of long abolished sexual must are posthumously pardoned",
                        help="The non-humorous sentence to be augmented")
    parser.add_argument("--word-id", default=11, type=int, help="index of the word to augment")
    parser.add_argument("--rep-word", default="offences", help="the word to replace")

    # allow to read from file as well... this will speed things up.
    parser.add_argument("--from-file", default=False, type=bool, help="Flag to parse from file")
    parser.add_argument("--sents-csv-path", help="Path to csv file with format 'sent','word-id','rep-word'")
    parser.add_argument("--outfile", default="scorer_output.csv", help="Path to output the results")

    parser.add_argument("--lm-path", default="/Users/sidharth/Documents/ucla/quarter4/NLG/final_project/code/pungen-master/fairseq/fairseq/models/wikitext/wiki103.pt",
                        help="path of the language model")
    parser.add_argument("--word-counts-path", default="/Users/sidharth/Documents/ucla/quarter4/NLG/final_project/code/pungen-master/fairseq/fairseq/models/wikitext/dict.txt",
                        help="path of language model word counts dict")
    parser.add_argument("--local-window-size", default=2, type=int, help="size of 'local' area in tokens")
    parser.add_argument("--oov-prob", default=0.03, type=float, help="Just set it to 0.03")
    return parser.parse_args()


# the tokenizer: split the sentence and return constituent tokens.
# TODO: can replace this with keras / or pre-tokenize...
def tokenize(sent):
    sent_tokens = sent.lower().split(' ')
    return sent_tokens


# perform the preparations of the sentences for scoring
def invert_sentence(orig_sentence, word_id, rep_word):
    humor_sent = tokenize(orig_sentence)
    print(humor_sent)
    orig_word = humor_sent[word_id]
    del (humor_sent[word_id])
    humor_sent.insert(word_id, rep_word)
    return humor_sent, orig_word


# perform the final scoring.
# take the original sentence and swap word, as well as the inverse sentence & swap word
# Todo: change these names if time, they're confusing...
def score_sentences(scorer, orig_sentence, word_id, rep_word, inverse_sent, orig_word=''):
    sin = tokenize(orig_sentence)  # let's make this tokenized.
    print("\n\n***** Performing scoring...")
    print(f"Original sentence: {sin}; Replaced word: {rep_word}")
    # this will return 4 scores.
    scores = scorer.analyze(sin, word_id, rep_word)
    print(scores)

    print("\n\n***** Performing 'inverse' scoring...")
    sin = inverse_sent
    print(f"Original sentence: {sin}; Replaced word: {orig_word}")
    inv_scores = scorer.analyze(sin, word_id, orig_word)
    print(inv_scores)
    return scores, inv_scores


# get the sentences, prepare the scorer, and perform scoring.
def main(args):

    # prep the scorer.
    lm = LMScorer.load_model(args.lm_path)
    unigram_model = UnigramModel(args.word_counts_path, args.oov_prob)
    scorer = SurprisalScorer(lm, unigram_model, local_window_size= args.local_window_size)
    line_ctr = 0

    # if crawling from file, prep the data.
    if not args.from_file:
        inverse_sent, orig_word = invert_sentence(args.orig_sentence, args.word_id, args.rep_word)
        score_sentences(scorer, args.orig_sentence, args.word_id, args.rep_word, inverse_sent, orig_word)
    else:
        # crawl from file and send to the scorer
        with open(args.outfile, 'w') as outf:
            with open(args.sents_csv_path) as f:
                out_writer = csv.writer(outf)
                # write out the header.
                out_writer.writerow(['id', 'sentence', 'word_id', 'mod_word', 'grammar', 'ratio', 'local_exp',
                                     'global_exp', 'inv_grammar', 'inv_ratio', 'inv_local_exp', 'inv_global_exp'])
                sent_reader = csv.reader(f)
                next(sent_reader)  # skip header.
                for line in sent_reader:
                    line_ctr += 1
                    if line_ctr >= 200:
                        break
                    # change this line depending on which file is being analyzed.
                    id, humor_sent, aug_word, orig_word, word_id, aug_verb, orig_verb = line

                    # preprocessing...
                    if word_id == '':
                        print("skipping! Invalid word id")
                        continue
                    word_id = int(word_id)

                    orig_sent, h = invert_sentence(humor_sent, word_id, orig_word)  # h should equal aug_word.
                    scores, orig_sent_scores = score_sentences(scorer, humor_sent, word_id, orig_word, orig_sent, h)

                    # get rid of some sentences before saving them
                    keep_sent = True
                    # condition 1: if grammar scores decrease by more than 0.9, throw out the sentence
                    # [Note]: this might be too strong.
                    if orig_sent_scores['grammar'] - scores['grammar'] >= 0.9:
                        keep_sent = False
                    # condition 2: if both local and global surprisal are too low, reject the sentence.
                    elif (scores['local'] < 0) and (scores['global'] < 0):
                        keep_sent = False
                    # [possible cond. 3, measure verb-level grammar.]

                    if keep_sent:
                        out_writer.writerow([id, humor_sent, word_id, orig_word, scores['grammar'], scores['ratio'],
                                             scores['local'], scores['global']])
                    else:
                        print("*** sentence rejected!")

    """
        Note on these 4 scores:
            i. grammar
            ii. ratio: this needs to be as high as possible.
            iii. local expectancy
            iv. global expectancy.
            
        IDEA:
            - language modeller "grammar" scores are needed
            - problem: "language model" scorer doesn't score grammaticality, it just scores based on co-occurrence of words.
    """


if __name__ == '__main__':
    args = parse_args()
    print(args)
    main(args)
