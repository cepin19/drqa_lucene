import json,sys
with open(sys.argv[1]) as f:
    docs=json.load(f)['data']
    print (json.dumps({'data':docs[:int(sys.argv[2])],"version":"1.1"}))