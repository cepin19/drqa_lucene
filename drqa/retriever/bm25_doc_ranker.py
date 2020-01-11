#!/usr/bin/env python3
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Rank documents with TF-IDF scores"""

import logging
import numpy as np
import scipy.sparse as sp

from multiprocessing.pool import ThreadPool
from functools import partial

from . import utils
from . import DEFAULTS
from .. import tokenizers
from drqa import retriever

logger = logging.getLogger(__name__)
#from rank_bm25 import BM25Okapi

db_class = retriever.get_class('sqlite')
db_opts={'db_path': '/home/large/data/models/marian/encz_exp/test/QA/.wikipedia/docs.db'}
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




class Bm25DocRanker(object):
    """Loads a pre-weighted inverted index of token/document terms.
    Scores new queries by taking sparse dot products.
    """

    def __init__(self, bm25_path=None, strict=True):
        """
        Args:
            tfidf_path: path to saved model file
            strict: fail on empty queries or continue (and return empty result)
        """
        # Load from disk
        bm25_path = bm25_path or DEFAULTS['bm25_path']
        logger.info('Loading %s' % bm25_path)
        #idf,doc_freqs,doc_len, avgdl,metadata = utils.load_bm25(bm25_path)
        self.bm25,metadata= utils.load_bm25(bm25_path)
        self.tokenizer = tokenizers.get_class(metadata['tokenizer'])()
        #self.doc_freqs = metadata['doc_freqs'].squeeze()
        self.doc_dict = metadata['doc_dict']
        self.num_docs = len(self.doc_dict[0])
        self.strict = strict
        #self.bm25=BM25Okapi('Nothing')
        #self.bm25.idf=idf
        #self.bm25.avgdl=avgdl
        #logging.info(idf)
        #self.bm25.doc_freqs=doc_freqs
        #self.bm25.doc_len=doc_len

    def get_doc_index(self, doc_id):
        """Convert doc_id --> doc_index"""
        return self.doc_dict[0][doc_id]

    def get_doc_id(self, doc_index):
        """Convert doc_index --> doc_id"""
        return self.doc_dict[1][doc_index]

    def closest_docs(self, query, k=1):
        """Closest docs by dot product between query and documents
        in tfidf weighted word vector space.
        """
        #logging.info(f'Getting closest docs for {query}')

        words = self.parse(utils.normalize(query))
        wids = [utils.hash(w, 2**24) for w in words]

        res = self.bm25.get_scores(wids)
#        logging.info(res)
 #       logging.info(self.bm25.doc_freqs)
  #      logging.info(self.bm25.doc_len)
   #     logging.info(self.bm25.corpus_size)

        if len(res.data) <= k:
            o_sort = np.argsort(-res)
        else:
#            o_sort=np.argsort(-res)#[0:k]
         #   logging.info(f"o_sort: {o_sort}")

            o = np.argpartition(-res, k)[0:k]
            #logging.info(f"O: {o}")
            o_sort = o[np.argsort(res[o])]


        doc_scores = res[o_sort]
        #logging.info("Scores")
        #logging.info(doc_scores)
        doc_ids = [self.get_doc_id(i) for i in o_sort]
 #       logging.info("Ids")
  #      logging.info(doc_ids)
   #     logging.info(doc_scores)
#logging.info("Text:")

        #doc_ids = [self.get_doc_id(i) for i in res.indices[o_sort]]

        return doc_ids, doc_scores

    def batch_closest_docs(self, queries, k=1, num_workers=None):
        """Process a batch of closest_docs requests multithreaded.
        Note: we can use plain threads here as scipy is outside of the GIL.
        """
        with ThreadPool(num_workers) as threads:
            closest_docs = partial(self.closest_docs, k=k)
            results = threads.map(closest_docs, queries)
        return results

    #    for q in queries:
     #       self.closest_docs(q,k=1)
      #  return self.closest_docs(q,k=1)

    def parse(self, query):
        """Parse the query into tokens (either ngrams or tokens)."""
        tokens = self.tokenizer.tokenize(query)
        return tokens.ngrams(n=1, uncased=True,
                             filter_fn=utils.filter_ngram)

    def text2spvec(self, query):
        """Create a sparse tfidf-weighted word vector from query.

        tfidf = log(tf + 1) * log((N - Nt + 0.5) / (Nt + 0.5))
        """
        # Get hashed ngrams
        words = self.parse(utils.normalize(query))
        wids = [utils.hash(w, self.hash_size) for w in words]

        if len(wids) == 0:
            if self.strict:
                raise RuntimeError('No valid word in: %s' % query)
            else:
                logger.warning('No valid word in: %s' % query)
                return sp.csr_matrix((1, self.hash_size))

        # Count TF
        wids_unique, wids_counts = np.unique(wids, return_counts=True)
        tfs = np.log1p(wids_counts)

        # Count IDF
        Ns = self.doc_freqs[wids_unique]
        idfs = np.log((self.num_docs - Ns + 0.5) / (Ns + 0.5))
        idfs[idfs < 0] = 0

        # TF-IDF
        data = np.multiply(tfs, idfs)

        # One row, sparse csr matrix
        indptr = np.array([0, len(wids_unique)])
        spvec = sp.csr_matrix(
            (data, wids_unique, indptr), shape=(1, self.hash_size)
        )

        return spvec
