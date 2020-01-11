#!/usr/bin/env python3
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""A script to build the tf-idf document matrices for retrieval."""

import numpy as np
import scipy.sparse as sp
import argparse
import os
import math
import logging
import sys

from multiprocessing import Pool as ProcessPool
from multiprocessing.util import Finalize
from functools import partial
from collections import Counter

from drqa import retriever
from drqa import tokenizers
#from rank_bm25 import BM25Okapi
import gc
hash_size=int(math.pow(2, 24))
logger = logging.getLogger()
logger.setLevel(logging.INFO)
fmt = logging.Formatter('%(asctime)s: [ %(message)s ]', '%m/%d/%Y %I:%M:%S %p')
console = logging.StreamHandler()
console.setFormatter(fmt)
logger.addHandler(console)

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
# Multiprocessing functions
# ------------------------------------------------------------------------------

DOC2IDX = None
PROCESS_TOK = None
PROCESS_DB = None


def init(tokenizer_class, db_class, db_opts):
    global PROCESS_TOK, PROCESS_DB
    PROCESS_TOK = tokenizer_class()
    Finalize(PROCESS_TOK, PROCESS_TOK.shutdown, exitpriority=100)
    PROCESS_DB = db_class(**db_opts)
    Finalize(PROCESS_DB, PROCESS_DB.close, exitpriority=100)


def fetch_text(doc_id):
    global PROCESS_DB
    return PROCESS_DB.get_doc_text(doc_id)


def tokenize(text):
    global PROCESS_TOK
    return PROCESS_TOK.tokenize(text)


# ------------------------------------------------------------------------------
# Build article --> word count sparse matrix.
# ------------------------------------------------------------------------------

def count(ngram, hash_size, doc_id):
    """Fetch the text of a document and compute hashed ngrams counts."""
    global DOC2IDX
    row, col, data = [], [], []
    # Tokenize
    tokens = tokenize(retriever.utils.normalize(fetch_text(doc_id)))

    # Get ngrams from tokens, with stopword/punctuation filtering.
    ngrams = tokens.ngrams(
        n=ngram, uncased=True, filter_fn=retriever.utils.filter_ngram
    )

    # Hash ngrams and count occurences
    counts = Counter([retriever.utils.hash(gram, hash_size) for gram in ngrams])

    # Return in sparse matrix data format.
    row.extend(counts.keys())
    col.extend([DOC2IDX[doc_id]] * len(counts))
    data.extend(counts.values())
    return row, col, data


def get_bm25_index(args,  db, db_opts):
    """Form a sparse word to document count matrix (inverted index).

    M[i, j] = # times word i appears in document j.
    """
    # Map doc_ids to indexes
    global DOC2IDX
    db_class = retriever.get_class(db)
    logging.info("Getting doc ids")

    with db_class(**db_opts) as doc_db:
        doc_ids = doc_db.get_doc_ids()
        DOC2IDX = {doc_id: i for i, doc_id in enumerate(doc_ids)}

        # Setup worker pool
        tok_class = tokenizers.get_class(args.tokenizer)
        #workers = ProcessPool(
         #   args.num_workers,
          #  initializer=init,
           # initargs=(tok_class, db_class, db_opts)
        #)
        logging.info(f"Documents ({len(doc_ids)}) are being tokenized")
        tok=tok_class()
        tokenized_corpus=[]
        bm25 = BM25Okapi()

        for i,doc_id in enumerate(doc_ids):
            #logging.info(doc_db.get_doc_text(doc_id))
            #logging.info(tok.tokenize(retriever.utils.normalize(doc_db.get_doc_text(doc_id))).words())i
            tokens=  tok.tokenize(retriever.utils.normalize(doc_db.get_doc_text(doc_id)))
            ngrams = tokens.ngrams(n=args.ngram, uncased=True, filter_fn=retriever.utils.filter_ngram)


            tokenized_corpus.append(ngrams)

            if i%10000==0:
                logging.info("%s/%s"%(i,len(doc_ids)))
                bm25.add(tokenized_corpus)
                logging.info(f"sizeof tokenized_corpus {sys.getsizeof(tokenized_corpus)}")
                logging.info(f"sizeof doc_freqs {sys.getsizeof(bm25.doc_freqs)}")
                logging.info(f"sizeof nd  {sys.getsizeof(bm25.nd)}")
                logging.info(f"sizeof num_doc {sys.getsizeof(bm25.num_doc)}")

                del tokenized_corpus
                tokenized_corpus = []
                gc.collect()

            if i==20000:
                pass
                #break
        bm25.add(tokenized_corpus)
        logging.info(f"sizeof tokenized_corpus {sys.getsizeof(tokenized_corpus)}")
        logging.info(f"sizeof doc_freqs {sys.getsizeof(bm25.doc_freqs)}")
        logging.info(f"sizeof nd  {sys.getsizeof(bm25.nd)}")
        logging.info(f"num_doc {bm25.num_doc}")

        logging.info("Tokenization done")
        #logging.info(tokenized_corpus)
        bm25.finish()



    return bm25 , (DOC2IDX, doc_ids)


# ------------------------------------------------------------------------------
# Transform count matrix to different forms.
# ------------------------------------------------------------------------------


def get_tfidf_matrix(cnts):
    """Convert the word count matrix into tfidf one.

    tfidf = log(tf + 1) * log((N - Nt + 0.5) / (Nt + 0.5))
    * tf = term frequency in document
    * N = number of documents
    * Nt = number of occurences of term in all documents
    """
    Ns = get_doc_freqs(cnts)
    idfs = np.log((cnts.shape[1] - Ns + 0.5) / (Ns + 0.5))
    idfs[idfs < 0] = 0
    idfs = sp.diags(idfs, 0)
    tfs = cnts.log1p()
    tfidfs = idfs.dot(tfs)
    return tfidfs


def get_doc_freqs(cnts):
    """Return word --> # of docs it appears in."""
    binary = (cnts > 0).astype(int)
    freqs = np.array(binary.sum(1)).squeeze()
    return freqs


# ------------------------------------------------------------------------------
# Main.
# ------------------------------------------------------------------------------


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('db_path', type=str, default=None,
                        help='Path to sqlite db holding document texts')
    parser.add_argument('out_dir', type=str, default=None,
                        help='Directory for saving output files')
    parser.add_argument('--ngram', type=int, default=2,
                        help=('Use up to N-size n-grams '
                              '(e.g. 2 = unigrams + bigrams)'))
    parser.add_argument('--hash-size', type=int, default=int(math.pow(2, 24)),
                        help='Number of buckets to use for hashing ngrams')
    parser.add_argument('--tokenizer', type=str, default='simple',
                        help=("String option specifying tokenizer type to use "
                              "(e.g. 'corenlp')"))
    parser.add_argument('--num-workers', type=int, default=None,
                        help='Number of CPU processes (for tokenizing, etc)')
    args = parser.parse_args()
    logger.info('Making BM25 index...')
    bm25, doc_dict = get_bm25_index(
        args, 'sqlite', {'db_path': args.db_path}
    )




    basename = os.path.splitext(os.path.basename(args.db_path))[0]
    basename += ('-bm25-ngram=%stokenizer=%s' %
                 (args.ngram, args.tokenizer))
    filename = os.path.join(args.out_dir, basename)
    filename_doc = os.path.join(args.out_dir, "doc_dict")

    logger.info('Saving to %s.npz' % filename)
    metadata = {
        'tokenizer': args.tokenizer,
        'doc_dict': doc_dict
    }
    #d={'idf':bm25.idf,'doc_freqs':bm25.doc_freqs,'doc_len':bm25.doc_len,'avgdl':bm25.avgdl}
    retriever.utils.save_bm25(filename, bm25, metadata)
    #retriever.utils.save_bm25(filename, d, metadata)
