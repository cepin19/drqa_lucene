
import logging
import math
import lucene, argparse
from drqa import retriever
from drqa import tokenizers
from java.util import HashMap

from java.io import StringReader
from java.lang import System
from java.nio.file import Path, Paths
from org.apache.lucene.analysis.core import WhitespaceAnalyzer
from org.apache.lucene.analysis.miscellaneous import LimitTokenCountAnalyzer
from org.apache.lucene.analysis.standard import StandardAnalyzer, StandardTokenizer
from org.apache.lucene.analysis.shingle import ShingleFilter
from org.apache.lucene.document import \
    Document, Field, StoredField, StringField, TextField
from org.apache.lucene.index import \
    IndexOptions, IndexWriter, IndexWriterConfig, DirectoryReader, \
    FieldInfos, MultiFields, MultiTerms, Term
from org.apache.lucene.util import PrintStreamInfoStream
from org.apache.lucene.queryparser.classic import \
    MultiFieldQueryParser, QueryParser
from org.apache.lucene.search import BooleanClause, IndexSearcher, TermQuery
from org.apache.lucene.store import MMapDirectory, SimpleFSDirectory
from org.apache.lucene.util import BytesRefIterator
from org.apache.lucene.document import Document, Field, FieldType
from org.apache.lucene.analysis.tokenattributes import CharTermAttribute

from org.apache.pylucene.analysis import PythonAnalyzer
from org.apache.lucene.analysis.miscellaneous import PerFieldAnalyzerWrapper
from lucene import *

class FooAnalyzer(PythonAnalyzer):
    def createComponents(self, fieldName):

        source = StandardTokenizer()
        filter = ShingleFilter(source)
        #filter.setMinShingleSize(1)
        filter.setMaxShingleSize(2)
        filter.setOutputUnigrams(True)
        #setMinShingleSize
        return self.TokenStreamComponents(source, filter)

    def initReader(self, fieldName, reader):
        return reader

logger = logging.getLogger()

logger.setLevel(logging.INFO)
fmt = logging.Formatter('%(asctime)s: [ %(message)s ]', '%m/%d/%Y %I:%M:%S %p')
console = logging.StreamHandler()
console.setFormatter(fmt)
logger.addHandler(console)

def indexDocument(self):
    store = self.openStore()
    writer = None
    try:
        analyzer = self.getAnalyzer()
        writer = self.getWriter(store, analyzer, True)

        doc = Document()
        doc.add(Field("title", "value of testing",
                      TextField.TYPE_STORED))
        doc.add(Field("docid", str(1),
                      StringField.TYPE_NOT_STORED))
        doc.add(Field("owner", "unittester",
                      StringField.TYPE_STORED))
        doc.add(Field("search_name", "wisdom",
                      StoredField.TYPE))
        doc.add(Field("meta_words", "rabbits are beautiful",
                      TextField.TYPE_NOT_STORED))

        # using a unicode body cause problems, which seems very odd
        # since the python type is the same regardless affter doing
        # the encode
        body_text = "hello world" * 20
        body_reader = StringReader(body_text)
        doc.add(Field("content", body_reader, TextField.TYPE_NOT_STORED))

        writer.addDocument(doc)
    finally:
        self.closeStore(store, writer)


if __name__ == "__main__":

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
    logger.info('Making lucene index...')
    db_opts= {'db_path': args.db_path}

    db_class = retriever.get_class("sqlite")
    logging.info("Getting doc ids")
    lucene.initVM()
    #store=SimpleFSDirectory(Paths.get('.'))
    #analyzer = WhitespaceAnalyzer()
    #analyzer = LimitTokenCountAnalyzer(analyzer, 10000)
    #config = IndexWriterConfig(analyzer)
    #    # config.setInfoStream(PrintStreamInfoStream(System.out))
    #config.setOpenMode(IndexWriterConfig.OpenMode.CREATE)
    #writer = IndexWriter(store, config)



    store = SimpleFSDirectory(Paths.get(args.out_dir))
    #analyzer = LimitTokenCountAnalyzer(FooAnalyzer(), 1048576)
    a=HashMap()
    a.put("content",StandardAnalyzer())
    a.put("content2",FooAnalyzer())
    aWrapper =PerFieldAnalyzerWrapper(StandardAnalyzer(),a)
    #aWrapper.addAnalyzer("content", StandardAnalyzer())
    #aWrapper.addAnalyzer("content2", FooAnalyzer());

    config = IndexWriterConfig(aWrapper)
    config.setOpenMode(IndexWriterConfig.OpenMode.CREATE)
    writer = IndexWriter(store, config)
    test = "This is how we do it."



    with db_class(**db_opts) as doc_db:
        doc_ids = doc_db.get_doc_ids()
        DOC2IDX = {doc_id: i for i, doc_id in enumerate(doc_ids)}

        #print ("%d docs in index" % writer.numDocs())

        t1 = FieldType()
        t1.setStored(True)
        t1.setTokenized(False)
        t1.setIndexOptions(IndexOptions.DOCS_AND_FREQS)

        t2 = FieldType()
        t2.setStored(True)
        t2.setTokenized(True)
        t2.setIndexOptions(IndexOptions.DOCS_AND_FREQS_AND_POSITIONS)
        for i,doc_id in enumerate(doc_ids):
            text= doc_db.get_doc_text(str(doc_id))
            doc = Document()
            doc.add(Field("content", text, t2))
            doc.add(Field("content2", text, t2))

            doc.add(Field("docid", str(doc_id),t1))
            writer.addDocument(doc)
            if i%100000==0:
                writer.commit()
                logging.info(f"{i}/{len(doc_ids)}")
   # print ("Closing index of %d docs..." % writer.numDocs())
    writer.close()
