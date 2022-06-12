import argparse
from lxml import etree
import json
import os
# Create the parser and add arguments
parser = argparse.ArgumentParser()
parser.add_argument("--Yahoo_data_path", type=str, help="path to load Yahoo data")
parser.add_argument("--CHQ_summ_path", type=str, help="path to load CHQ-Summ dataset")



# Parse and print the results
args = parser.parse_args()

print(args)

def clean(text):
    '''
    to clean the text by removing undesirable spaces.
    :param text:
    :return:
    '''
    text = text.replace('\n', ' ')
    text = text.replace('\t', ' ')
    text = text.replace('  ', ' ')
    text = text.replace('   ', ' ')
    return text.strip()

def process_chq_summ(yahoo_data_items, chq_summ_data, mode, path_to_save):
    with open(chq_summ_data, 'r') as rfile:
        chq_dataset = json.load(rfile)

    sources=[]
    targets=[]
    ids = []
    for chq_data_item in chq_dataset:
        if chq_data_item['id'] not in yahoo_data_items:
            continue
        meta_data = yahoo_data_items[chq_data_item['id']]
        sources.append(meta_data['question']+' '+meta_data['content'])
        targets.append(chq_data_item['human_summary'])

    with open(os.path.join(path_to_save, mode+".id"), 'w') as f_id,\
            open(os.path.join(path_to_save, mode+".source"), 'w') as f_src, open(os.path.join(path_to_save, mode+".target"), 'w') as f_tgt:
        for id, src, tgt in zip(ids, sources, targets):
            src = clean(src)
            tgt = clean(tgt)
            f_id.write(id.strip()+'\n')
            f_src.write(src+'\n')
            f_tgt.write(tgt+'\n')
    print(f"Saved file in : {path_to_save}")


def read_yahoo_data(yahooPath='data/dataset/Yahoo-L6/FullOct2007.xml'):
    data_items = {}
    ctr = 0
    for event, iter in etree.iterparse(yahooPath, tag="vespaadd", encoding='utf-8', recover=True):
        doc = iter.find('document')
        try:
            meta_data = {}
            q_id = doc.findtext('uri')
            question = doc.findtext('subject')
            content = doc.findtext('content')
            meta_data['id'] = q_id
            meta_data['question'] = question
            meta_data['content'] = content
            data_items[meta_data['id']]=meta_data
            ctr += 1
            if ctr%1000==0:
                print("Read Questions: ", ctr)
        except:
            print('ERROR')
    return data_items



def main(args):
    yahoo_data_items = read_yahoo_data(args.Yahoo_data_path)
    process_chq_summ(yahoo_data_items, os.path.join(args.CHQ_summ_path, 'train.json'),
                     mode='train',
                      path_to_save=args.CHQ_summ_path)
    #
    process_chq_summ(yahoo_data_items, os.path.join(args.CHQ_summ_path, 'val.json'),
                     mode='val',
                     path_to_save=args.CHQ_summ_path)

    process_chq_summ(yahoo_data_items, os.path.join(args.CHQ_summ_path, 'test.json'),
                     mode='test',
                     path_to_save=args.CHQ_summ_path)

main(args)