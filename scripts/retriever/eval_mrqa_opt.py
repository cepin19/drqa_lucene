#!/usr/bin/env python3
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Evaluate the accuracy of the DrQA retriever module."""

import regex as re
import logging
import argparse
import json
import time
import os
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from multiprocessing import Pool as ProcessPool
from multiprocessing.util import Finalize
from functools import partial
from drqa import retriever, tokenizers
from drqa.retriever import utils
import numpy as np
from hyperopt import hp
from hyperopt import fmin, tpe

class BM25:
    def __init__(self, tokenizer=None):
        self.corpus_size = 0
        self.avgdl = 0
        self.num_doc=0
        self.doc_freqs = []
        self.idf = {}
        self.doc_len = []
        self.tokenizer = tokenizer


        self.nd={}

    def add(self,text):
        self.corpus_size+=len(text)
      #  nd = {}  # word -> number of documents with word
        for document in text:
            self.doc_len.append(len(document))
            self.num_doc += len(document)

            frequencies = {}
            for word in document:
                hash=retriever.utils.hash(word, hash_size)

                if hash not in frequencies:
                    frequencies[hash] = 0
                frequencies[hash] += 1
            self.doc_freqs.append(frequencies)

            for hash, freq in frequencies.items():
                if hash not in self.nd:
                    self.nd[hash] = 0
                self.nd[hash] += 1

        #return nd

    def finish(self):
        self.avgdl = self.num_doc / self.corpus_size

        self._calc_idf(self.nd)
        del self.nd
        self.nd={}
        gc.collect()
    def _initialize(self, corpus):
        nd = {}  # word -> number of documents with word
        num_doc = 0
        for document in corpus:
            self.doc_len.append(len(document))
            num_doc += len(document)

            frequencies = {}
            for word in document:
                hash=retriever.utils.hash(word, hash_size)
                if hash not in frequencies:
                    frequencies[hash] = 0
                frequencies[hash] += 1
            self.doc_freqs.append(frequencies)

            for hash, freq in frequencies.items():
                if hash not in nd:
                    nd[hash] = 0
                nd[hash] += 1

        self.avgdl = num_doc / self.corpus_size
        return nd

    def _tokenize_corpus(self, corpus):
        pool = Pool(cpu_count())
        tokenized_corpus = pool.map(self.tokenizer, corpus)
        return tokenized_corpus

    def _calc_idf(self, nd):
        raise NotImplementedError()

    def get_scores(self, query):
        raise NotImplementedError()

    def get_top_n(self, query, documents, n=5):

        assert self.corpus_size == len(documents), "The documents given don't match the index corpus!"

        scores = self.get_scores(query)
        top_n = np.argsort(scores)[::-1][:n]
        return [documents[i] for i in top_n]


class BM25Okapi(BM25):
    def __init__(self,tokenizer=None, k1=1.5, b=0.75, epsilon=0.25):
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon
        super().__init__(tokenizer)

    def _calc_idf(self, nd):
        """
        Calculates frequencies of terms in documents and in corpus.
        This algorithm sets a floor on the idf values to eps * average_idf
        """
        # collect idf sum to calculate an average idf for epsilon value
        idf_sum = 0
        # collect words with negative idf to set them a special epsilon value.
        # idf can be negative if word is contained in more than half of documents
        negative_idfs = []
        for word, freq in nd.items():
            idf = math.log(self.corpus_size - freq + 0.5) - math.log(freq + 0.5)
            self.idf[word] = idf
            idf_sum += idf
            if idf < 0:
                negative_idfs.append(word)
        self.average_idf = idf_sum / len(self.idf)

        eps = self.epsilon * self.average_idf
        for word in negative_idfs:
            self.idf[word] = eps

    def get_scores(self, query):
        """
        The ATIRE BM25 variant uses an idf function which uses a log(idf) score. To prevent negative idf scores,
        this algorithm also adds a floor to the idf value of epsilon.
        See [Trotman, A., X. Jia, M. Crane, Towards an Efficient and Effective Search Engine] for more info
        :param query:
        :return:
        """
        score = np.zeros(self.corpus_size)
        doc_len = np.array(self.doc_len)
        for q in query:
            q_freq = np.array([(doc.get(q) or 0) for doc in self.doc_freqs])
            score += (self.idf.get(q) or 0) * (q_freq * (self.k1 + 1) /
                                               (q_freq + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)))
        return score


# ------------------------------------------------------------------------------
# Multiprocessing target functions.
# ------------------------------------------------------------------------------

PROCESS_TOK = None
PROCESS_DB = None


def init(tokenizer_class, tokenizer_opts, db_class, db_opts):
    global PROCESS_TOK, PROCESS_DB
    PROCESS_TOK = tokenizer_class(**tokenizer_opts)
    Finalize(PROCESS_TOK, PROCESS_TOK.shutdown, exitpriority=100)
    PROCESS_DB = db_class(**db_opts)
    Finalize(PROCESS_DB, PROCESS_DB.close, exitpriority=100)


def regex_match(text, pattern):
    """Test if a regex pattern is contained within a text."""
    try:
        pattern = re.compile(
            pattern,
            flags=re.IGNORECASE + re.UNICODE + re.MULTILINE,
        )
    except BaseException:
        return False
    return pattern.search(text) is not None

def get_len(answer,doc_id):
    #logger.info("ASDSD")
    global PROCESS_DB, PROCESS_TOK
    text = PROCESS_DB.get_doc_text(doc_id)
    text = utils.normalize(text)
    #logger.warning(text)

    #logger.info("Length")
    #logger.info(len(text))

def has_answer(answer, doc_id, match):
    """Check if a document contains an answer string.

    If `match` is string, token matching is done between the text and answer.
    If `match` is regex, we search the whole text with the regex.
    """
    global PROCESS_DB, PROCESS_TOK
    if not doc_id:
        logger.warning("NO DOCID")
        return False
    text = PROCESS_DB.get_doc_text(doc_id)
    if not text:
        logger.warning("NO TEXT")
        return False
    text = utils.normalize(text)
    #logger.info(text)
    if match == 'string':
        # Answer is a list of possible strings
        text = PROCESS_TOK.tokenize(text).words(uncased=True)
        for single_answer in answer:
            single_answer = utils.normalize(single_answer)
            single_answer = PROCESS_TOK.tokenize(single_answer)
            single_answer = single_answer.words(uncased=True)
            for i in range(0, len(text) - len(single_answer) + 1):
                if single_answer == text[i: i + len(single_answer)]:
                    return True
    elif match == 'regex':
        # Answer is a regex
        single_answer = utils.normalize(answer[0])
        if regex_match(text, single_answer):
            return True
    return False


def get_score(answer_doc, match):
    """Search through all the top docs to see if they have the answer."""
    answer, (doc_ids, doc_scores) = answer_doc
    for doc_id in doc_ids:
        if has_answer(answer, doc_id, match):
            return 1
    return 0


# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------

def run(sampled_args):
    k1, b=sampled_args
    print ("running with k:{} b:{}".format(k1,b))

    logger.info('Ranking...')
    closest_docs = ranker.batch_closest_docs(
        questions, k1=k1,b=b,k=args.n_docs, num_workers=args.num_workers
    )
    answers_docs = zip(answers, closest_docs)

    # define processes
    tok_class = tokenizers.get_class(args.tokenizer)
    tok_opts = {}
    db_class = retriever.DocDB
    db_opts = {'db_path': args.doc_db}
    processes = ProcessPool(
        processes=args.num_workers,
        initializer=init,
        initargs=(tok_class, tok_opts, db_class, db_opts)
    )
    # logger.info(closest_docs)
    # compute the scores for each pair, and print the statistics
    logger.info('Retrieving and computing scores...')
    get_score_partial = partial(get_score, match=args.match)
    scores = processes.map(get_score_partial, answers_docs)

    lens = []
    quest_lens = []

    for quest, doc_ids in zip(questions, closest_docs):
        #logger.info(doc_ids[0])
        for doc_id in doc_ids[0]:
         #   logger.info(doc_id)
            text = PROCESS_DB.get_doc_text(doc_id)
            if not text:
                logger.warning("NO TEXT")
                continue
            text = utils.normalize(text)
            # logger.warning(text)
            # logger.info("Length")
            # logger.info(len(text))
            lens.append(len(text))
            quest_lens.append(len(quest))

    filename = os.path.basename(args.dataset)
    stats = (
        "\n" + "-" * 50 + "\n" +
        "{filename}\n" +
        "Examples:\t\t\t{total}\n" +
        "Matches in top {k}:\t\t{m}\n" +
        "Match % in top {k}:\t\t{p:2.2f}\n" +
        "Total time:\t\t\t{t:2.4f} (s)\n"
    ).format(
        filename=filename,
        total=len(scores),
        k=args.n_docs,
        m=sum(scores),
        p=(sum(scores) / len(scores) * 100),
        t=time.time() - start,
    )
    print(f"Avg len of docs in top {args.n_docs}: {sum(lens) / len(lens)} characters")
    plt.scatter(quest_lens, lens,s=0.05)
    plt.ylim(0, 250000)
    print(f"Correlation coeff between paragraph lengths and question lengths {np.corrcoef(lens,quest_lens)}")
    plt.savefig('lucene_wikik1{}b{}.png'.format(k1,b))

    print(stats)
    processes.close()
    processes.join()
    return  -(sum(scores) / len(scores) * 100)
if __name__ == '__main__':
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s: [ %(message)s ]',
                            '%m/%d/%Y %I:%M:%S %p')
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)

   # PROCESS_TOK = tok_class(**tok_opts)
    #Finalize(PROCESS_TOK, PROCESS_TOK.shutdown, exitpriority=100)

    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str, default=None)
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--doc-db', type=str, default=None,
                        help='Path to Document DB')
    parser.add_argument('--tokenizer', type=str, default='regexp')
    parser.add_argument('--n-docs', type=int, default=5)
    parser.add_argument('--num-workers', type=int, default=None)
    parser.add_argument('--match', type=str, default='string',
                        choices=['regex', 'string'])
    args = parser.parse_args()

    # start time
    start = time.time()

    # read all the data and store it
    logger.info('Reading data ...')
    questions = []
    answers = []
    with open(args.dataset) as d:
        j=d.read()
       # print(json.loads(j))

    for q in json.loads(j):
       # print (q["paragraphs"])
        #data = (line)
        for qas in q['qas']:
            #print (qas)
            #sštitles.append("")
            question = qas['question']
            answer = [a for a in qas['answers']]
            questions.append(question)
            answers.append(answer)
    #questions=questions[:100]
    #answers=answers[:100]

    # get the closest docs for each question.
    logger.info('Initializing ranker...')
    ranker = retriever.get_class('lucene')(lucene_path=args.model)

    # minimize the objective over the space
#[hp.uniform('k1', 0, 2), hp.uniform('b', 0, 1)]
    db_class = retriever.DocDB
    db_opts = {'db_path': args.doc_db}
    PROCESS_DB = db_class(**db_opts)
    Finalize(PROCESS_DB, PROCESS_DB.close, exitpriority=100)
    best = fmin(run, space=[hp.uniform('k1', 0, 2), hp.uniform('b', 0, 1)], algo=tpe.suggest, max_evals=100)
    print (best)