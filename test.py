import json

def read_jsonl(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def process_data(data):
    path_dict = {}
    for d in data:
        class_name = d['class_name']
        if class_name not in path_dict:
            path_dict[class_name] = {}
            path_dict[class_name]['class_id'] = d['class_id']
            path_dict[class_name]['class_name'] = d['class_name']
            path_dict[class_name]['image_path'] = []
            path_dict[class_name]['caption'] = []
        samples = d['samples']
        neighbors = d['neighbors'] 
        # import pdb;pdb.set_trace()
        for sample in samples:
            path_dict[class_name]['image_path'].append(sample)
        for neighbor in neighbors:
            neg_class_id = neighbor[0]
            neg_class_name = neighbor[1]
            if neg_class_name not in path_dict:
                path_dict[neg_class_name] = {}
                path_dict[neg_class_name]['class_id'] = neg_class_id
                path_dict[neg_class_name]['class_name'] = neg_class_name
                path_dict[neg_class_name]['image_path'] = []
                path_dict[neg_class_name]['caption'] = []
            path_dict[neg_class_name]['image_path'].append(neighbor[-1])
    for key in path_dict.keys():
        if len(path_dict[key]) != len(set(path_dict[key])):
            print('重複あり')
            path_dict[key] = list(set(path_dict[key]))
    return path_dict

data = read_jsonl("/home/oshita/vlm/LLaVA/train900_pairs.jsonl")
name = []
for d in data:
    name.append(d['class_name'])

import pdb;pdb.set_trace()
data = process_data(data)
# dataの順番を各キーの保有するclass_idでソートする
data = dict(sorted(data.items(), key=lambda x: int(x[1]['class_id'])))
# dictをjsonlとして保存する
with open('test_imagenet_cap.jsonl', 'w') as f:
    for key in data.keys():
        f.write(json.dumps(data[key]) + '\n')


