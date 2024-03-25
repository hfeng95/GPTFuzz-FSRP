import json
import os

results_dir = 'results/'

files = [j for j in os.listdir(results_dir) if j.endswith('.json')]

mutator_info = {}

for f_name in files:
    with open(os.path.join(results_dir,f_name),'r') as f:
        js = json.load(f)
        for k in js:
            mutator = js[k]['mutator']
            if mutator == 'NoneType':
                continue
            if mutator not in mutator_info:
                mutator_info[mutator] = {
                        'num_attempt' : 0,
                        'num_jailbreak' : 0,
                        'num_reject' : 0
                        }
            mutator_info[mutator]['num_attempt'] += 1
            mutator_info[mutator]['num_jailbreak'] += js[k]['num_jailbreak']
            mutator_info[mutator]['num_reject'] += js[k]['num_reject']

print('===== mutator totals =====')
for k in mutator_info:
    print(k)
    print('num attempts: ',mutator_info[k]['num_attempt'])
    print('num jailbreaks: ',mutator_info[k]['num_jailbreak'])
    print('num rejects: ',mutator_info[k]['num_reject'])

initial = list(range(9))    # keys of initial seeds
prompts_info = {}
for f_name in files:
    with open(os.path.join(results_dir,f_name),'r') as f:
        js = json.load(f)
        for i in initial:
            k = str(i)
            queue = js[k]['children'][:]
            while queue:
                n = str(queue.pop(0))
                if k not in prompts_info:
                    prompts_info[k] = {
                            'num_attempt' : 0,
                            'num_jailbreak' : 0,
                            'num_reject' : 0
                            }
                prompts_info[k]['num_attempt'] += 1
                prompts_info[k]['num_jailbreak'] += js[n]['num_jailbreak']
                prompts_info[k]['num_reject'] += js[n]['num_reject']
                if 'children' in js[n]:
                    queue.extend(js[n]['children'])
            
print('===== prompt totals =====')
for k in prompts_info:
    print('prompt ',k)
    print('num attempts: ',prompts_info[k]['num_attempt'])
    print('num jailbreaks: ',prompts_info[k]['num_jailbreak'])
    print('num rejects: ',prompts_info[k]['num_reject'])
