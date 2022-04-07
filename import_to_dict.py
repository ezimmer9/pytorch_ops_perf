import json

'''
    this is reference how to get perf results output to python dict
'''
with open('build/perf_results.txt') as f:
    data = f.read()
    print(data)
    js = json.loads(data)
    print(js)