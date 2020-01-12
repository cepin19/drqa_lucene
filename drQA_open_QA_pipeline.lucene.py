import json
import logging
import os
import socket
import sys
from multiprocessing.pool import ThreadPool

import numpy as np
import regex
import torch
from drqa import retriever
from drqa.retriever import DocDB
from torchtext.data import BucketIterator
from torch.nn import CrossEntropyLoss
from tqdm import tqdm

from datasets.SQUAD_dataset_BIDAF_QA import SquadDatasetBidaf
from scripts.bidaf.bidaf_trainer import BidafModelFramework
from scripts.common.evaluate_squad import evaluate
from scripts.common.util import setup_logging
from tokenizers.spacy_tokenizer import tokenize


def process_paragraphs(paragraphs, word_field, char_field, device):
    # def process_text(t):
    #     tokenized = tokenize_and_join(t)  # TODO:  this can be moved into preprocessing step
    #     w = word_field.process(tokenized)
    #     c = char_field.process(tokenized)
    #     return w, c

    # TODO: I have some errors when forking new process,
    # probably because main process takes a lot of memory and forked process wants to have this memory too?
    # with Pool(num_workers) as pool:
    #     return pool.map(process_text, paragraphs)
    # for p in paragraphs:
    #     r.append(process_text(p))
    # return r

    tokenized = []
    positions = []
    for p in paragraphs:
        tokens, text_tokens = tokenize(p)
        # lowercase
        tokenized.append([str.lower(t) for t in text_tokens])
        positions.append([[token.idx, token.idx + len(token.text)] for token in tokens])
    # preprocess characters
    char_tokenized_preprocessed = [char_field.preprocess(k) for k in tokenized]
    w = word_field.process(tokenized).to(device)
    c = char_field.process(char_tokenized_preprocessed).to(device)
    return w, c, positions


def get_docs_from_db(top_K_document_ids, db: DocDB, num_workers=4):
    with ThreadPool(num_workers) as threads:
        return threads.map(db.get_doc_text, top_K_document_ids)

def split_document_into_paragraphs(doc, group_length=0):
    """
    Taken from drQA: https://github.com/facebookresearch/DrQA/blob/167b6401cf02ee5440c1bc6278a6eeaf72739ee7/drqa/pipeline/drqa.py#L149
    :param doc: document to split into paragraphs
    :param group_length:  Target size for squashing short paragraphs together.
                          0 = read every paragraph independently
                          infinity = read all paragraphs together
    """
    curr = []
    curr_len = 0
    # split on multiple eolns
    if not doc:
        logging.warning("Missing doc???")
        logging.warning(doc)
        return ''
    for split in regex.split(r'\n+', doc):
        split = split.strip()
        # skip if empty
        if len(split) == 0:
            continue
        # group paragraph together if they are shorted than group_limit
        # setting group_limit to 0 disables this
        if len(curr) > 0 and curr_len + len(split) > group_length:
            yield ' '.join(curr)
            curr = []
            curr_len = 0
        curr.append(split)
        curr_len += len(split)
    # if there is something in curr and group_limit has not been reached,
    # group it and return anyway
    if len(curr) > 0:
        yield ' '.join(curr)


@torch.no_grad()
def validate_bigram_hashing():
    K = 5
    #RANKER_PATH = ".retrievers/drqa_tfidf/docs-tfidf-ngram=2-hash=16777216-tokenizer=simple.npz"
    #RANKER_PATH = "./scripts/openqa/DrQA/scripts/retriever/squad-tfidf-ngram=2-hash=16777216-tokenizer=simple.npz"
    #RANKER_PATH = "./scripts/openqa/DrQA/scripts/retriever/wiki18_index_trigram/"
    #RANKER_PATH = "./scripts/openqa/DrQA/scripts/retriever/docs-tfidf-ngram=1-hash=16777216-tokenizer=simple.npz"
    RANKER_PATH = "./squad_index"
    #RANKER_PATH = "./scripts/openqa/DrQA/scripts/retriever//nq_val-tfidf-ngram=2-hash=16777216-tokenizer=simple.npz"
    #DB_PATH = "/mnt/data/ijon/docs.db"

    #DB_PATH = "/mnt/data/ijon/enwiki_20181220.db"
    DB_PATH="squad.db"
    #DB_PATH="/mnt/minerva1/nlp/projects/counterfactual/martin_qa/QA/scripts/openqa/DrQA/scripts/retriever/nq_val.db"
    MODEL_PATH = "saved/bidaf_EM_65.22_F1_75.01_L_2.16608_2019-08-14_14:08_pcfajcik.pt"
    #MODEL_PATH="saved/bidaf_nq.pt"
    #MODEL_PATH="saved/nq_filtered_multiple_answers_train_and_val.pt"
    #MODEL_PATH="saved/nq_filtered_multiple_answers_train_only.pt"
    #MODEL_PATH="saved/nq_filtered_multiple_answers_train_to_single.pt"
    #MODEL_PATH="saved/nq_filtered_multiple_answers_train_to_single_long_run.pt"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #DEV_DATA = "/mnt/minerva1/nlp/projects/counterfactual/martin_qa/QA/scripts/openqa/DrQA/scripts/retriever/v1.0-simplified-nq-val_paragraphs_squad_formatwid.jsonl"
    DEV_DATA = "dev-v1.1.json"


    WORKERS = 4
    BATCH_SIZE = 16

    CLOSED_DOMAIN_CHECK = False

    doc_db = None

    logging.info('Loading model...')
    model = torch.load(MODEL_PATH, map_location=device)
    model = model.eval()
    model.encoder.rnn.flatten_parameters()
    model.modeling_layer.rnn.flatten_parameters()
    model.output_layer.rnn.flatten_parameters()

    logging.info('Loading data/iterator...')
    fields = SquadDatasetBidaf.prepare_fields_char_noanswerprocessing(include_lengths=False)
    #val = SquadDatasetBidaf(DEV_DATA, fields, cachedir='.data/nq', compact=True)

    val = SquadDatasetBidaf(DEV_DATA, fields, cachedir='.data/squad', compact=True)
    fields = dict(fields)
    # fields["question_char"].build_vocab(val, max_size=260)
    fields["question"].vocab = model.embedder.vocab
    fields["question_char"].nesting_field.vocab = model.char_embedder.vocab

    val_iter = BucketIterator(val, sort_key=lambda x: -(len(x.question)), sort=True,
                              batch_size=BATCH_SIZE,
                              repeat=False,
                              device=device)
    if not CLOSED_DOMAIN_CHECK:
        logging.info('Loading ranker...')
        ranker = retriever.get_class('lucene')(
           lucene_path=RANKER_PATH)
        #ranker = retriever.get_class('tfidf')(
         #   tfidf_path=RANKER_PATH)
        logging.info('Connecting to database...')
        doc_db = DocDB(db_path=DB_PATH)
    ids = []
    lossvalues = []
    spans = []
    gt_spans = []
    span_scores = []
    best_doc_ids = []
    lossfunction = CrossEntropyLoss(reduction='none')
    total_its = len(val_iter.data()) // val_iter.batch_size + 1
    pbar = tqdm(total=total_its)

    for iteration, batch in enumerate(val_iter):
        ids += batch.id
        gt_spans += batch.gt_answer
        # TODO: fix iterator, to do this implicitly
        # question lengths are aligned here, but they should be aligned when constructing true batch
        processed_question_w = batch.question
        processed_question_c = batch.question_char

        if CLOSED_DOMAIN_CHECK:
            # FIXME: broken for now, assumes old data format, and old way of evaluating results
            logprobs_S, logprobs_E = model(batch)
            loss_s = lossfunction(logprobs_S, batch.a_start)
            loss_e = lossfunction(logprobs_E, batch.a_end)
            loss = loss_s + loss_e
            lossvalues += loss.tolist()

            best_span_probs, i_candidates = model.decode(logprobs_S, logprobs_E)
            span_scores += best_span_probs.tolist()  # prob = prob_start * prob_end
            spans += BidafModelFramework.get_spans(batch, i_candidates)
        else:
            best_span_probs_topK, candidates_topK, lossvalues_topK = [], [], []

            ####
            # Part 1. Document Retrieval
            ###
            # TODO: This part can be moved into preprocessing

            # Retrieve closest document ids
            # TODO: this PROBABLY ALWAYS returns title of article, is it a problem?
            # sometimes, retrieval can return less than K values, if there is zero overlap with query
            ranked = ranker.batch_closest_docs(batch.raw_question, k=K, num_workers=WORKERS)#, k1=1.35,b=0.0093)
            top_K_document_ids, top_K_document_scores = zip(*ranked)

            # top_K_document_index = []
            # for ex_idx, retrieved_docs in enumerate(top_K_document_ids):
            #     top_K_document_index += [ex_idx] * len(retrieved_docs)
            # top_K_document_index = np.array(top_K_document_index)  # np.tile(range(len(top_K_document_ids)), [2, 1]).T

            # Remove duplicates
            # TODO: disabled for now
            # top_K_document_ids = np.array(list({d for docids in top_K_document_ids for d in docids}))
            top_K_document_ids_and_indices = [(e_id, doc_id) for e_id, doclist in enumerate(top_K_document_ids) for
                                              doc_id in doclist]
            top_K_document_index, top_K_document_ids = zip(
                *top_K_document_ids_and_indices)  # top_K_document_ids_and_indices[:,0].astype(int), top_K_document_ids_and_indices[:, 1]
            top_K_document_index = np.array(top_K_document_index)
            top_K_document_ids = np.array(top_K_document_ids)
            top_K_documents = get_docs_from_db(top_K_document_ids, num_workers=WORKERS, db=doc_db)

            # Split and flatten documents. Maintain a mapping from doc (index in
            # flat list) to split (index in flat list).
            # e.g. first element didx2sidx[0] contains double (0,x), which means
            # paragraphs from first document are located from index 0 to index x
            flat_paragraphs = []
            paragraph_mask = []
            d2p = []
            for text_i, text in zip(top_K_document_index, top_K_documents):
                paragraphs = list(split_document_into_paragraphs(text))
                paragraph_mask += [text_i] * len(paragraphs)
                for split in paragraphs:
                    flat_paragraphs.append(split)
            last = 0
            last_i = 0
            for m_i, m in enumerate(paragraph_mask):
                if m != last:
                    d2p.append((last_i, m_i))
                    last_i = m_i
                    last = m
                if m_i == len(paragraph_mask) - 1:
                    d2p.append((last_i, m_i + 1))

            # paragraphs_per_doc = [split_document_into_paragraphs(text) for text in raw_docs]
            flat_positions = []
            iterations = len(flat_paragraphs) // batch.batch_size if len(flat_paragraphs) % batch.batch_size == 0 else \
                len(flat_paragraphs) // batch.batch_size + 1
            for i in range(iterations):
                paragraphs = flat_paragraphs[i * batch.batch_size:(i + 1) * batch.batch_size]
                mask = torch.LongTensor(paragraph_mask[i * batch.batch_size:(i + 1) * batch.batch_size]).to(device)
                batch.document, batch.document_char, positions = process_paragraphs(paragraphs, fields["question"],
                                                                                    fields["question_char"], device)
                flat_positions += positions
                batch.question = torch.index_select(processed_question_w, 0, mask)
                batch.question_char = torch.index_select(processed_question_c, 0, mask)

                #######
                # Part 2, processing with model
                #######

                logprobs_S, logprobs_E = model(batch, mask_rnns=False)

                best_span_probs, candidates = model.decode(logprobs_S, logprobs_E, score="logprobs")

                best_span_probs_topK += best_span_probs.cpu().tolist()
                candidates_topK += list(zip(candidates[0].cpu().tolist(), candidates[1].cpu().tolist()))

            #####
            # Part 3. processing results for each paragraph in K retrieved documents, picking the best answer
            #####
            span_indices = []
            batch.raw_document_context = []
            batch.document_token_positions = []

            # Check there is answer for each paragraph
            assert len(best_span_probs_topK) == len(candidates_topK) == len(flat_paragraphs)
            for predictions_i_start, predictions_i_end in d2p:
                best_pred_scores = best_span_probs_topK[predictions_i_start:predictions_i_end]
                i_candidates = candidates_topK[predictions_i_start:predictions_i_end]
                best_pred_argmax = np.argmax(best_pred_scores)
                best_pred_max = best_pred_scores[best_pred_argmax]
                candidate = i_candidates[best_pred_argmax]
                span_indices.append(candidate)
                span_scores.append(best_pred_max)
                batch.raw_document_context.append(
                    flat_paragraphs[predictions_i_start:predictions_i_end][best_pred_argmax])
                batch.document_token_positions.append(
                    flat_positions[predictions_i_start:predictions_i_end][best_pred_argmax])

            span_indices = np.array(span_indices).T
            spans += BidafModelFramework.get_spans(batch, span_indices)
            pbar.update(1)

    ####
    # Part 4. Evaluating the results
    ####
    scores = dict()
    results = dict()

    for id, span, span_prob in zip(ids, spans, span_scores):
        results[id] = span
        scores[id] = span_prob

    prediction_file = f".data/nq/dev_results_{socket.gethostname()}.json"
    with open(prediction_file, "w") as f:
        json.dump(results, f)

    dataset_file = ".data/squad/dev-v1.1.json"
    #dataset_file = "/mnt/minerva1/nlp/projects/counterfactual/martin_qa/QA/scripts/openqa/DrQA/scripts/retriever/v1.0-simplified-nq-val_paragraphs_squad_formatwid.jsonl"
    #dataset_file = "/mnt/minerva1/nlp/projects/counterfactual/martin_qa/QA/scripts/openqa/DrQA/scripts/retriever/v1.0-simplified-nq-val_paragraphs_squad_formatwid_head100.jsonl"

    expected_version = '1.1'
    with open(dataset_file) as dataset_file:
        dataset_json = json.load(dataset_file)
        if (dataset_json['version'] != expected_version):
            logging.info('Evaluation expects v-' + expected_version +
                         ', but got dataset with v-' + dataset_json['version'],
                         file=sys.stderr)
        dataset = dataset_json['data']
    with open(prediction_file) as prediction_file:
        predictions = json.load(prediction_file)
    result = evaluate(dataset, predictions)

    # logging.info(loss)
    logging.info(json.dumps(result))

    # close the DB connection
    if doc_db is not None:
        doc_db.close()


if __name__ == "__main__":
    setup_logging(os.path.basename(sys.argv[0]).split(".")[0],
                  logpath="logs/",
                  config_path="configurations/logging.yml")
    validate_bigram_hashing()
    # count_bigramhasing_paragraphs()
