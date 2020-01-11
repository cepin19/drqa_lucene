import pickle,json
idx=[]
with open("v1.0-simplified-nq-val_paragraphs.jsonl") as f:
    for line in f:
        idx.append(json.loads(line)["doc_idx"])

with open("nqid_to_wiki16.pkl","wb") as f:
    pickle.dump(idx,f)
