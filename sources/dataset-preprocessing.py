
import pandas as pd
import numpy as np
import string
import json
import shutil

from pathlib import Path
import os
from os import listdir, mkdir
from os.path import isfile, isdir, join, exists, abspath
from keras.preprocessing import image
from keras.applications.resnet import ResNet152, preprocess_input
from sklearn.model_selection import train_test_split
! pip install git+https://github.com/crazyfrogspb/RedditScore.git > /dev/null
def _globalMaxPool1D(tensor):
    _,_,_,size = tensor.shape
    return [tensor[:,:,:,i].max() for i in range(size)]

def _getImageFeatures(model, img_path):
    img = image.load_img(img_path, target_size=None)

    img_data = image.img_to_array(img)
    img_data = np.expand_dims(img_data, axis=0)
    img_data = preprocess_input(img_data)

    feature_tensor = model.predict(img_data)
    get_img_id = lambda p: p.split('/')[-1].split('.')[0]
    return {
        "id": get_img_id(img_path),
        "features": _globalMaxPool1D(feature_tensor),
    }

def _getJSON(path):
    with open(path) as json_file:
        return json.loads(json.load(json_file))
    
def _clean_text(text):
    text = text.replace("\n", " ")
    # onyshchak: only checking first 1000 characters, will need to extract summary propely
    text = text[:1000].rsplit(' ', 1)[0]
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

def _getTextFeatures(text_path):
    data = _getJSON(text_path)
    
    return {
        'id': data['id'],
        'text': _clean_text(data['text']),
        'title': data['title']
    }

def _getImagesMeta(path):
    data = _getJSON(path)['img_meta']
    for x in data:
        x['description'] = _clean_text(x['description'])
        x['title'] = _clean_text(x['title'])
    return data

def _getValidImagePaths(article_path):
    img_path = join(article_path, 'img/')
    return [join(img_path, f) for f in listdir(img_path) if isfile(join(img_path, f)) and f[-4:].lower() == ".jpg"]

def _dump(path, data):
    with open(path, 'w', encoding='utf8') as outfile:
        json.dump(data, outfile, indent=2, ensure_ascii=False)

def GetArticleData(article_path):
    article_data = _getTextFeatures(join(article_path, 'text.json'))
    article_data["img"] = _getImagesMeta(join(article_path, 'img/', 'meta.json'))
    
    return article_data

def ReadArticles(data_path, offset=0, limit=None):
    print("Reading in progress...")
    article_paths = [join(data_path, f) for f in listdir(data_path) if isdir(join(data_path, f))]
    limit = limit if limit else len(article_paths) - offset
    
    articles = []
    for i in range(offset, offset + limit):
        path = article_paths[i]
        if (i - offset + 1) % 100 == 0: print(i - offset, "articles have been read")
        article_data = GetArticleData(path)
        articles.append(article_data)
        if len(articles) >= limit: break  # useless?
        
    print(limit, "articles have been read")
    return articles

def GenerateVisualFeatures(data_path, offset=0, limit=None, model=None):
    article_paths = [join(data_path, f) for f in listdir(data_path) if isdir(join(data_path, f))]
    limit = limit if limit else len(article_paths) - offset
    model = model if model else ResNet152(weights='imagenet', include_top=False) 
    
    for i in range(offset, offset + limit):
        path = article_paths[i]
        print(i, path)
    
        meta_path = join(path, 'img/', 'meta.json')
        meta_arr = _getImagesMeta(meta_path)
        for meta in meta_arr:
            if 'features' in meta: continue
            if meta['filename'][-4:].lower() != ".jpg": continue
                
            img_path =  join(path, 'img/', meta['filename'])
            try:
                features = _getImageFeatures(model, img_path)['features']
                meta['features'] = [str(f) for f in features]
            except Exception as e:
                print("exception", str(e))
                print(img_path)
                continue
                
        _dump(meta_path, json.dumps({"img_meta": meta_arr}))
%%time
articles = ReadArticles('/kaggle/input/extended-wikipedia-multimodal-dataset/data/', offset=0, limit=None)
dataset_name = 'data_w2vv'
dataset_path = join('./', dataset_name)
if exists(dataset_path):
    shutil.rmtree(dataset_path)
    
mkdir(dataset_path)
subsets = {
    "train": {},
    "val": {},
    "test": {},
}

for k, v in subsets.items():
    v['name'] = dataset_name + k
    v['path'] = join(dataset_path, v['name'])
    mkdir(v['path'])
    
    v['feature_data_path'] = join(v['path'], 'FeatureData')
    if k == 'train':
        mkdir(v['feature_data_path'])
    else:
        dst = v['feature_data_path']
        os.symlink(os.path.relpath(subsets['train']['feature_data_path'], Path(dst).parent), dst)

    v["image_sets_path"] = join(v['path'], 'ImageSets')
    mkdir(v["image_sets_path"])

    v["text_data_path"] = join(v['path'], 'TextData')
    mkdir(v["text_data_path"])
def to_file(arr, filepath):
    with open(filepath, 'w') as f:
        for x in arr:
            f.write("%s\n" % x)
            
# map_data = lambda func: [func(a, i) for a in articles for i in a['img'] if 'features' in i]
def map_data():
    seen = set()
    res = []
    for a in articles:
        for i in a['img']:
            if 'features' not in i: continue
                
            img_id = os.path.splitext(i['filename'])[0]  # removing file extention
            if img_id in seen:
                # onyshchak: if image used in 2 articles, we only take the first one for simplicity
                # TODO: use all the infomation without breaking the model
                continue
                
            seen.add(img_id)
            res.append({
                "filename": img_id,
                'article_title': a['title'],
                "title": os.path.splitext(i['title'])[0],
                "description": i['description'],
                "text": a['text'],
                "features": i['features'],
            })
            
    return res

data = map_data()
del articles
list2str = lambda l: " ".join([str(x) for x in l])

img_features = ['{} {}'.format(x['filename'], list2str(x['features'])) for x in data]

raw_features_file_path = join(subsets['train']["feature_data_path"], subsets['train']['name'] + ".features.txt")
to_file(img_features, raw_features_file_path)
subsets['train']['data'], subsets['test']['data'] = train_test_split(
    data, test_size=1000, random_state=1234
)

subsets['train']['data'], subsets['val']['data'] = train_test_split(
    subsets['train']['data'], test_size=1000, random_state=1234
)

del data
for v in subsets.values():
    ids = [x['filename'] for x in v['data']]
    to_file(ids, join(v["image_sets_path"], v['name'] + ".txt"))
from redditscore.tokenizer import CrazyTokenizer

tokenizer = CrazyTokenizer(hashtags='split')

def split_hashtag_words(text):
    global tokenizer
    res = []
    for x in text.split():
        res += tokenizer.tokenize("#" + x)
        
    return " ".join(res)

def remove_auxiliary_words(text):
    aux = ["jpeg", "jpg", "png"] # only 1 valid word can contain png, other - image extention trash
    for a in aux:
        if a == text[-len(a):]:
            text = text[:-len(a)]
            break
                    
    return text

def camel_case_split(str): 
    words = [[str[0]]] 
  
    for c in str[1:]: 
        l = words[-1][-1]
        if (l.islower() and c.isupper()) or (l.isdigit() and not c.isdigit()) or (not l.isdigit() and c.isdigit()): 
            words.append(list(c)) 
        else: 
            words[-1].append(c) 
  
    return ' '.join([''.join(word) for word in words])

process_title = lambda x: split_hashtag_words(remove_auxiliary_words(camel_case_split(x)))
%%time
# onyshchak: originally ID also contained file extention e.g. *.jpg. but not in image_sets_path
get_description = lambda z: z['description'] if z['description'] else process_title(z['title'])

for v in subsets.values():
    text_data = sorted(
        ['{}#enc#0 {}'.format(x['filename'], x['text']) for x in v['data']] +
        ['{}#enc#1 {}'.format(x['filename'], get_description(x)) for x in v['data']]
    )

    to_file(text_data, join(v["text_data_path"], v['name'] + ".caption.txt"))
for k,v in subsets.items():
    del v['data']
! apt install --assume-yes python-pip > /dev/null
! python2 -m pip install --user numpy scipy matplotlib ipython jupyter pandas nose > /dev/null
IS_FILE_LIST = 0
FEATURE_DIMENTION = 2048
feature_data_path = subsets['train']["feature_data_path"]
bin_features_path = join(feature_data_path, "pyresnet152-pool5os/")

! python2 /kaggle/input/w2vv-scripts/simpleknn/txt2bin.py $FEATURE_DIMENTION $raw_features_file_path $IS_FILE_LIST $bin_features_path --overwrite 1
! mv $dataset_name/* ./
! rmdir $dataset_name