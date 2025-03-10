import os
from allennlp.common import Params
import myutils


def makeConfig(strategy, seed, runNonSmoothed):
    udPath = 'data/ud-treebanks-v2.7.noEUD/'
    fullConfig = {}
    for UDdir in os.listdir(udPath):
        train, dev, test = myutils.getTrainDevTest(udPath + UDdir)
        if train != '':
            if not myutils.hasColumn(train, 1):
                continue
            config = {}
            config['train_data_path'] = train
            if dev != '':
                config['validation_data_path'] = dev
            config['word_idx'] = 1
            config['tasks'] = {}
            if strategy in ['concat', 'datasetEmbeds']:
                config['tasks']['upos'] = {'task_type':'seq', 'column_idx':3}
                if myutils.hasColumn(train, 2):
                    config['tasks']['lemma'] = {'task_type':'string2string', 'column_idx':2}
                config['tasks']['feats'] = {'task_type':'seq', 'column_idx':5}
                config['tasks']['dependency'] = {'task_type':'dependency', 'column_idx':6}
                if strategy == 'datasetEmbeds':
                    config['dataset_embed_idx'] = -1
            elif strategy == 'sepDec':
                config['tasks']['upos-' + UDdir] = {'task_type':'seq', 'column_idx':3}
                if myutils.hasColumn(train, 2):
                    config['tasks']['lemma-' + UDdir] = {'task_type':'string2string', 'column_idx':2}
                config['tasks']['feats-' + UDdir] = {'task_type':'seq', 'column_idx':5}
                config['tasks']['dependency-' + UDdir] = {'task_type':'dependency', 'column_idx':6}
            fullConfig[UDdir] =  config
    
    allennlpConfig = Params(fullConfig)
    jsonPath = 'configs/tmp/fullUD' + strategy + '.json'
    allennlpConfig.to_file(jsonPath)
    cmd = 'python3 train.py --dataset_config ' + jsonPath 
    if strategy == 'datasetEmbeds':
        cmd += ' --parameters_config configs/params.datasetEmbeds.json'
    else:
        cmd += ' --parameters_config configs/params.smoothSampling.json'
    cmd += ' --name fullUD' + strategy + '.smoothed.' + seed + ' --seed ' + seed
    print(cmd)
    if runNonSmoothed:
        cmd = 'python3 train.py --dataset_config ' + jsonPath + ' --name fullUD' + strategy + '.' + seed + ' --seed ' + seed
        print(cmd)


# multi-dataset models
#TODO not efficient, hasColumn is expensive and done 3 times for each check!
for seed in myutils.seeds:
    makeConfig('concat', seed, True)
    makeConfig('sepDec', seed, False)
    makeConfig('datasetEmbeds', seed, False)

if not os.path.isdir('configs/tmp/'):
    os.mkdir('configs/tmp/')

# single-dataset models
udPath = 'data/ud-treebanks-v2.7.noEUD/'
for UDdir in os.listdir(udPath):
    train, dev, test = myutils.getTrainDevTest(udPath + UDdir)
    
    if train != '':
        if not myutils.hasColumn(train, 1):
            continue
        config = {}
        config['train_data_path'] = train
        if dev != '':
            config['validation_data_path'] = dev
        config['word_idx'] = 1
        config['tasks'] = {}
        config['tasks']['upos'] = {'task_type':'seq', 'column_idx':3}
        if myutils.hasColumn(train, 2):
            config['tasks']['lemma'] = {'task_type':'string2string', 'column_idx':2}
        config['tasks']['feats'] = {'task_type':'seq', 'column_idx':5}
        config['tasks']['dependency'] = {'task_type':'dependency', 'column_idx':6}
    
        allennlpConfig = Params({UDdir: config})
        jsonPath = 'configs/tmp/' + UDdir + '.json'
        allennlpConfig.to_file(jsonPath)
        for seed in myutils.seeds:
            print('python3 train.py --dataset_config ' + jsonPath + ' --seed ' + seed + ' --name ' + UDdir + '.' + seed)


