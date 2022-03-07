from glob import glob
import os, re, json, sys

#TODO: do something smarter about this
paths = glob(sys.argv[1] + "*")
outfile_json = sys.argv[2]
include = [] #take everything
# 13 lang
include = ['ar_padt', 'eu_bdt', 'zh_gsd', 'en_ewt', 'fi_tdt', 'he_htb', 'hi_hdtb', 'it_isdt', 'ja_gsd', 'ko_gsd', 'ru_syntagrus', 'sv_talbanken', 'tr_imst']
# germanic 
include = ['nl_alpino', 'sv_talbanken', 'en_ewt', 'de_hdt', 'af_afribooms', 'da_ddt', 'got_proiel', 'is_icepahc', 'no_bokmaal' ]
# slavic
include = ['ru_syntagrus', 'cs_pdt', 'pl_lfg','sk_snk','cu_proiel', 'orv_torot','sr_set', 'uk_iu' ]
# romance
include = ['fr_gsd', 'pt_gsd', 'it_isdt', 'es_ancora', 'ro_rrt', 'eu_bdt']
isos = {}

for path in paths:
    treebank_name = os.path.split(path)[1]
    devfiles = glob(path + "/*-ud-dev.conllu") # returns list
    trainfiles = glob(path + "/*-ud-train.conllu")
    #if devfiles and trainfiles:
    has_train_or_dev = False
    if devfiles:
        has_train_or_dev = True
        devfile = devfiles[0]
        m = re.match(r'(.*)-ud-dev.conllu',os.path.basename(devfile))
    else:
        devfile = None
    if trainfiles:
        has_train_or_dev = True
        trainfile = trainfiles[0]
        if not devfile:
            m = re.match(r'(.*)-ud-train.conllu',os.path.basename(trainfile))
    else:
        trainfile = None
    if has_train_or_dev:
        iso = m.group(1)
        if not include or iso in include:
            isos[treebank_name] = {}
            isos[treebank_name]['iso'] = iso
            isos[treebank_name]['devfile'] = devfile
            isos[treebank_name]['trainfile'] = trainfile
    #else:
    #    print(f"No data found for {treebank_name}")

config = open(outfile_json, 'w')
config.write('{ \n')
#TODO: add option to have other tasks
#TODO: I should not print anything out for devfile or trainfile if none
# right now I am just going to manually delete them

for treebank_iso in isos:
    if not treebank_iso.endswith('POS'): #hack
        tbank_param = (
                f'    \"{treebank_iso}\" : {{ \n'
                f'        \"train_data_path\": \"{isos[treebank_iso]["trainfile"]}\", \n'
                f'        \"validation_data_path\": \"{isos[treebank_iso]["devfile"]}\", \n'
                f'        \"word_idx\": 1, \n'
                f'        \"dataset_embed_idx\": -1, \n'
                f'        \"copy_other_columns\": true, \n'
                f'        \"tasks": {{ \n'
                f'            \"dependency\": {{ \n'
                f'                \"task_type\": \"dependency\", \n'
                f'                \"column_idx\": 6 \n'
                f'             }} \n          }} \n     }}, \n'
                )
    else:
        tbank_param = (
                f'    \"{treebank_iso}\" : {{ \n'
                f'        \"train_data_path\": \"{isos[treebank_iso]["trainfile"]}\", \n'
                f'        \"validation_data_path\": \"{isos[treebank_iso]["devfile"]}\", \n'
                f'        \"word_idx\": 1, \n'
                f'        \"dataset_embed_idx\": -1, \n'
                f'        \"copy_other_columns\": true, \n'
                f'        \"tasks": {{ \n'
                f'            \"upos\": {{ \n'
                f'                \"task_type\": \"seq\", \n'
                f'                \"column_idx\": 3 \n'
                f'             }} \n          }} \n     }}, \n'
                )
    config.write(tbank_param)

config.write('}')
config.close()
