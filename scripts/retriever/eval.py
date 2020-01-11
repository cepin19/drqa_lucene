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
import pickle
mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from multiprocessing import Pool as ProcessPool
from multiprocessing.util import Finalize
from functools import partial
from drqa import retriever, tokenizers
from drqa.retriever import utils
import numpy as np
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
    logger.info("ASDSD")
    global PROCESS_DB, PROCESS_TOK
    text = PROCESS_DB.get_doc_text(doc_id)
    text = utils.normalize(text)
    logger.warning(text)

    logger.info("Length")
    logger.info(len(text))

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
    #print(list(answer_doc))
    answer, (doc_ids, doc_scores) = answer_doc
    for doc_id in doc_ids:
        if has_answer(answer, doc_id, match):
            return (1, len(utils.normalize(PROCESS_DB.get_doc_text(doc_id))))
    return (0, None)


# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------


if __name__ == '__main__':
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s: [ %(message)s ]',
                            '%m/%d/%Y %I:%M:%S %p')
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)

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
    with open("squad_topics_to_wiki.pkl",'rb') as idx:
        squad_to_wiki=pickle.load(idx)
    with open("nqid_to_wiki16.pkl", 'rb') as idx:
        nq_to_wiki16 = pickle.load(idx)
    titles=[]
   # print(squad_to_wiki)
    with open(args.dataset) as d:
        j=d.read()
       # print(json.loads(j))

    for q in json.loads(j)["data"]:
       # print (q["paragraphs"])
        #data = (line)
        for p in q['paragraphs']:
            for qas in p['qas']:
                #print (qas)
                titles.append(q["title"])
                question = qas['question']
                answer = [a["text"] for a in qas['answers']]
                questions.append(question)
                answers.append(answer)
    #questions=questions[:100]
    #answers=answers[:100]
    #titles=titles[:100]

    # get the closest docs for each question.
    logger.info('Initializing ranker...')
    ranker = retriever.get_class('lucene')(lucene_path=args.model)
#    ranker = retriever.get_class('tfidf')(tfidf_path=args.model)

    logger.info('Ranking...')
    closest_docs = ranker.batch_closest_docs(
       questions, k=args.n_docs,  num_workers=args.num_workers)#,	k1=1.81,b=0.2295)#k1=1.87,b=0.084)
    #closest_docs = ranker.batch_closest_docs(
     #   questions, k=args.n_docs,  num_workers=args.num_workers,b=0.2295, k1=1.810)
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
    #logger.info(closest_docs)
    # compute the scores for each pair, and print the statistics
    logger.info('Retrieving and computing scores...')
    get_score_partial = partial(get_score, match=args.match)
    scores_and_lens = processes.map(get_score_partial, answers_docs)
    scores=list(zip(*scores_and_lens))[0]
    correct_lens=[l for l in list(zip(*scores_and_lens))[1] if l]

    PROCESS_TOK = tok_class(**tok_opts)
    Finalize(PROCESS_TOK, PROCESS_TOK.shutdown, exitpriority=100)
    PROCESS_DB = db_class(**db_opts)
    Finalize(PROCESS_DB, PROCESS_DB.close, exitpriority=100)
    lens=[]
    quest_lens=[]


    for quest,doc_ids in zip(questions,closest_docs):
        #logger.info(doc_ids[0])
        for doc_id in doc_ids[0]:
         #   logger.info(doc_id)
            text = PROCESS_DB.get_doc_text(doc_id)
            if not text:
                logger.warning("NO TEXT")
                continue
            text = utils.normalize(text)
            #logger.warning(text)
            #logger.info("Length")
            #logger.info(len(text))
            lens.append(len(text))
        quest_lens.append(len(quest))
    #all_lens=[]
    #for i,id in enumerate(PROCESS_DB.get_doc_ids()):
     #   if (i%100000==0 and i>0):
      #      print(f"reading {i}/{len(PROCESS_DB.get_doc_ids())}")
            #break
     #   text = PROCESS_DB.get_doc_text(id)
      #  text = utils.normalize(text)

        #all_lens.append(len(text))
    correct_wiki=[]
    correct_wiki_lens=[]
    docs_wiki_index=[]
  #  for quest,ans, title in zip(questions,answers, titles):
   #     docs_wiki_index.append(([all_docs[squad_to_wiki[title]]],[100]))
       # print (title)
        #print (all_docs[squad_to_wiki[title]])
    #    correct=PROCESS_DB.get_doc_text(all_docs[squad_to_wiki[title]])
        #print(correct)
     #   correct=utils.normalize(correct)
      #  correct_wiki_lens.append(len(correct))
       # correct_wiki.append(correct)
        #if len(correct)>120000:
         #   print (title)
          #  print (len(correct))
           # print(correct)
      #      print("-------------------------------------------------------")
    #print(docs_wiki_index)
   # print(titles)
   # get_score_partial = partial(get_score, match=args.match)
   # scores_and_lens_wiki = processes.map(get_score_partial, zip(answers,docs_wiki_index))
   # scores_wiki = list(zip(*scores_and_lens_wiki))[0]
   # print ("wiki index score: {}".format((sum(scores_wiki) / len(scores_wiki) * 100)))
    all_docs=PROCESS_DB.get_doc_ids()
    for id,(quest,ans) in enumerate(zip(questions,answers)):
        docs_wiki_index.append(([all_docs[nq_to_wiki16[id]]],[100]))
    correct=PROCESS_DB.get_doc_text(all_docs[nq_to_wiki16[id]])
    print(correct)
    correct=utils.normalize(correct)
    correct_wiki_lens.append(len(correct))
    correct_wiki.append(correct)
    if len(correct)>120000:
        print (id)
        print (len(correct))
    print(correct)
    #      print("-------------------------------------------------------")
    # print(docs_wiki_index)
    # print(titles)
    get_score_partial = partial(get_score, match=args.match)
    scores_and_lens_wiki = processes.map(get_score_partial, zip(answers,docs_wiki_index))
    scores_wiki = list(zip(*scores_and_lens_wiki))[0]
    print ("wiki index score: {}".format((sum(scores_wiki) / len(scores_wiki) * 100)))
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

    #sns.distplot(np.asarray(lens), hist = True, kde = False,
      #           kde_kws = {'linewidth': 1},label="retrieved lens")
   # print(all_lens)
    #print(lens)

#    sns.distplot(np.asarray(all_lens),hist = True, kde = False,norm_hist=True,
   #              kde_kws = {'linewidth': 1},label="all lens")

 #   sns.distplot(np.asarray(correct_lens),hist = True, kde = False,norm_hist=True,
  #               kde_kws = {'linewidth': 1},label="correct lens")
  #  plt.scatter(quest_lens, lens,s=0.05)
    #np.save("nq_correct_lens_wiki16.npy",correct_lens)
#    np.save("correct_lens_wiki_idx.npy", correct_wiki_lens)
    #np.save()
    #np.save("nq_all_lens_wiki16.npy",all_lens)
    savefn="nq_lucene_unigram_default_wiki16.npy"
    np.save(savefn,lens)
    print(f"saved to {savefn}")

 #   plt.ylim(0, 250000)
   # print(f"Correlation coeff between correct article lengths and question lengths {np.corrcoef(correct_wiki_lens,quest_lens)}")
    #ret_plot.get_figure().savefig('distribution1.png')
    #all_plot.get_figure().savefig('distribution2.png')
   # plt.legend()
   # plt.xlim(-10000, 120000)
   # plt.xlabel('Length')
   # plt.ylabel('Density')
    #plt.savefig("asdfg1.png", dpi=1600)
    print(stats)
