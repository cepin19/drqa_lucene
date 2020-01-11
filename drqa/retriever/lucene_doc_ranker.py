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

import sys, lucene, unittest
import os, shutil
from org.apache.lucene.analysis.standard import StandardAnalyzer, StandardTokenizer
from org.apache.lucene.analysis.shingle import ShingleFilter
from java.io import StringReader
from java.lang import System
from java.nio.file import Path, Paths
from org.apache.lucene.analysis.core import WhitespaceAnalyzer
from org.apache.lucene.analysis.miscellaneous import LimitTokenCountAnalyzer
from org.apache.lucene.analysis.standard import StandardAnalyzer
from org.apache.lucene.document import \
    Document, Field, StoredField, StringField, TextField
from org.apache.lucene.index import \
    IndexOptions, IndexWriter, IndexWriterConfig, DirectoryReader, \
    FieldInfos, MultiFields, MultiTerms, Term
from org.apache.lucene.search import BooleanQuery
from org.apache.lucene.util import PrintStreamInfoStream
from org.apache.lucene.queryparser.classic import \
    MultiFieldQueryParser, QueryParser
from org.apache.lucene.search import BooleanClause, IndexSearcher, TermQuery
from org.apache.lucene.store import MMapDirectory, SimpleFSDirectory
from org.apache.lucene.util import BytesRefIterator

from org.apache.lucene.search import BooleanClause, BooleanQuery, TermQuery
from org.apache.lucene.search.similarities import BM25Similarity,ClassicSimilarity

from org.apache.pylucene.analysis import PythonAnalyzer
class FooAnalyzer(PythonAnalyzer):
    def createComponents(self, fieldName):

        source = StandardTokenizer()
        filter = ShingleFilter(source)
        filter.setMaxShingleSize(2)
        filter.setOutputUnigrams(False)
        #filter = StopFilter(True, filter, StopAnalyzer.ENGLISH_STOP_WORDS_SET)
        #exit(0)
        return self.TokenStreamComponents(source, filter)

    def initReader(self, fieldName, reader):
        return reader

logger = logging.getLogger(__name__)





class LuceneDocRanker(object):
    """Loads a pre-weighted inverted index of token/document terms.
    Scores new queries by taking sparse dot products.
    """

    def __init__(self, lucene_path=None, strict=True):
        """
        Args:
            tfidf_path: path to saved model file
            strict: fail on empty queries or continue (and return empty result)
        """
        # Load from disk
        self.STORE_DIR=lucene_path
        self.strict = strict
        lucene.initVM()
        self.store = self.openStore()
        self.searcher = self.getSearcher(self.store)

    def getAnalyzer(self):
        return FooAnalyzer()

    def getSearcher(self, store):
        return IndexSearcher(DirectoryReader.open(store))
    def openStore(self):
        logger.info('Loading %s' % self.STORE_DIR)

        return SimpleFSDirectory(Paths.get(self.STORE_DIR))
    def get_doc_index(self, doc_id):
        """Convert doc_id --> doc_index"""
        return self.doc_dict[0][doc_id]

    def get_doc_id(self, doc_index):
        """Convert doc_index --> doc_id"""
        return self.doc_dict[1][doc_index]

    def closeStore(self, store, *args):
        for arg in args:
            if arg is not None:
                arg.close()

        store.close()

    def closest_docs(self, query, k=1,k1=1.2,b=0.75):
        """Closest docs by dot product between query and documents
        in tfidf weighted word vector space.
        """
        vm_env = lucene.getVMEnv()
        vm_env.attachCurrentThread()

        searcher = self.searcher
        self.searcher.setSimilarity(BM25Similarity(k1,b))
        #self.searcher.setSimilarity(ClassicSimilarity())

        #logging.info(query)
        BooleanQuery.setMaxClauseCount(40000)
        luceneQuery = QueryParser("content", StandardAnalyzer()).parse(QueryParser.escape(query))
        #luceneQuery2 = QueryParser("content2", FooAnalyzer()).parse(QueryParser.escape(query))

        b = BooleanQuery.Builder()
        b.add(BooleanClause(luceneQuery, BooleanClause.Occur.SHOULD))
        #b.add(BooleanClause(luceneQuery2, BooleanClause.Occur.SHOULD))
        q = b.build()

        #topDocs = searcher.search(luceneQuery, k)
        #topDocs2 = searcher.search(luceneQuery2, k)
        topDocs = searcher.search(q, k)

        #logging.info(f"QUERY {q}")
       # logging.info(f"docs: {len(topDocs.scoreDocs)}")
        doc_ids=[]
        doc_scores=[]

        for hit in topDocs.scoreDocs:
        #    logging.info("hit")
         #   logging.info((hit.score, hit.doc, hit.toString()))
            doc = searcher.doc(hit.doc)
            #logging.info(doc)
          #  logging.info(doc.get("content"))
           # logging.info(doc.get("docid"))
            doc_scores.append(hit.score)
            doc_ids.append(doc.get("docid"))

      #  for hit in topDocs2.scoreDocs:
            #logging.info("hit")
            #logging.info((hit.score, hit.doc, hit.toString()))
       #     doc = searcher.doc(hit.doc)
            #logging.info(doc)
            #logging.info(doc.get("content"))
            #logging.info(doc.get("docid"))
        #    if doc.get("docid") in doc_ids:
               # logging.info(f"adding bigram score for doc {doc.get('docid')} to doc_score index {doc_ids.index(doc.get('docid'))}" )
         #       doc_scores[doc_ids.index(doc.get("docid"))]+=hit.score
          #  else:
                #doc_scores.append(hit.score)
                #doc_ids.append(doc.get("docid"))

        #logging.info(f"QUERY: {query}")
        #logging.info(doc_ids)
        #logging.info(doc_scores)
        #if not doc_ids:
         #   doc_ids=[1]
          #  doc_scores=[1.0]
        #logging.info(sorted(zip(doc_ids,doc_scores), key=lambda x: x[1])[:k])
       # s=sorted(zip(doc_ids,doc_scores), key=lambda x: x[1], reverse=True)[:k]
        #doc_ids=[x[0] for x in s]
        #doc_scores=[x[1] for x in s]

        return doc_ids,doc_scores

    def batch_closest_docs(self, queries, k=1,k1=1.2,b=0.75, num_workers=None):
        """Process a batch of closest_docs requests multithreaded.
        Note: we can use plain threads here as scipy is outside of the GIL.
        """
       # with ThreadPool(num_workers) as threads:
       #     closest_docs = partial(self.closest_docs, k=k)
        #    results = threads.map(closest_docs, queries)
        #return results
        results=[]
        for q in queries:
            results.append(self.closest_docs(q,k1=k1,b=b,k=k))
        return results