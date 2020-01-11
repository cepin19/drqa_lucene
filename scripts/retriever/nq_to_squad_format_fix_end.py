import json,sys

squad_format = []

id=0
for line in sys.stdin:
    line=json.loads(line)
    answers=[]
    for a_start,a_end in zip(line["start"],line["end"]):

        #print(f"{a_start}:{a_end}")
        #print(line)
        #print(line["paragraph"].split(' ')[a_start:a_end])
        answers.append({"answer_start":a_start,"text":' '.join(line["paragraph"].split(' ')[a_start:a_end])})
    question=line["question"]
    paragraphs=[{"context":line["paragraph"],"qas":[{"answers":answers,"question":question,'id':str(id)}]}]
    squad_format.append({"title":"","paragraphs":paragraphs})
    id+=1

print(json.dumps({"data":squad_format,"version": "1.1"}))