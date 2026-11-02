import os
import errno
import shutil
import pickle
import torch
import numpy as np
from pathlib import Path
from collections import defaultdict

DATA_PATH = "./data"


def parse_triple(line):
    items = line.strip().split('\t')
    if len(items) == 3:
        return items

    items = line.strip().split()
    if items and items[-1] == '.':
        items = items[:-1]
    if len(items) != 3:
        return None
    return items


def read_id_file(path):
    mapping = {}
    with open(path, 'r') as to_read:
        for line in to_read:
            line = line.strip()
            if not line:
                continue
            key, idx = line.rsplit(maxsplit=1)
            mapping[key] = int(idx)
    return mapping


def write_id_file(mapping, path):
    with open(path, 'w+') as to_write:
        for key, idx in sorted(mapping.items(), key=lambda x: x[1]):
            to_write.write("{}\t{}\n".format(key, idx))


def copy_feature_file(src, dst, n_entities):
    features = torch.load(src, map_location='cpu')
    if not hasattr(features, 'shape') or len(features.shape) == 0:
        raise ValueError("{} must contain a tensor feature matrix".format(src))
    if features.shape[0] != n_entities:
        raise ValueError("{} has {} rows, expected {}".format(src, features.shape[0], n_entities))
    shutil.copyfile(src, dst)


def create_filters_and_probas(out_path, files, n_entities, n_relations):
    print("creating filtering lists")

    to_skip = {'lhs': defaultdict(set), 'rhs': defaultdict(set)}
    for f in files:
        examples = pickle.load(open(Path(out_path) / (f + '.pickle'), 'rb'))
        for lhs, rel, rhs in examples:
            to_skip['lhs'][(rhs, rel + n_relations)].add(lhs)  # reciprocals
            to_skip['rhs'][(lhs, rel)].add(rhs)

    to_skip_final = {'lhs': {}, 'rhs': {}}
    for kk, skip in to_skip.items():
        for k, v in skip.items():
            to_skip_final[kk][k] = sorted(list(v))

    out = open(Path(out_path) / 'to_skip.pickle', 'wb')
    pickle.dump(to_skip_final, out)
    out.close()

    examples = pickle.load(open(Path(out_path) / 'train.pickle', 'rb'))
    counters = {
        'lhs': np.zeros(n_entities),
        'rhs': np.zeros(n_entities),
        'both': np.zeros(n_entities)
    }

    for lhs, rel, rhs in examples:
        counters['lhs'][lhs] += 1
        counters['rhs'][rhs] += 1
        counters['both'][lhs] += 1
        counters['both'][rhs] += 1
    for k, v in counters.items():
        counters[k] = v / np.sum(v)
    out = open(Path(out_path) / 'probas.pickle', 'wb')
    pickle.dump(counters, out)
    out.close()


def prepare_multimodal_dataset(path, name):
    files = ['train', 'valid', 'test']
    entities_to_id = read_id_file(os.path.join(path, 'entity2id.txt'))
    relations_to_id = read_id_file(os.path.join(path, 'relation2id.txt'))
    n_entities = max(entities_to_id.values()) + 1
    n_relations = max(relations_to_id.values()) + 1

    print("{} entities and {} relations".format(n_entities, n_relations))
    out_path = os.path.join(DATA_PATH, name)
    os.makedirs(out_path, exist_ok=True)
    write_id_file(entities_to_id, os.path.join(out_path, 'ent_id'))
    write_id_file(relations_to_id, os.path.join(out_path, 'rel_id'))

    for f in files:
        file_path = os.path.join(path, f)
        examples = []
        skipped = 0
        with open(file_path, 'r') as to_read:
            for line in to_read:
                triple = parse_triple(line)
                if triple is None:
                    skipped += 1
                    continue
                lhs, rel, rhs = triple
                if lhs not in entities_to_id or rhs not in entities_to_id or rel not in relations_to_id:
                    skipped += 1
                    continue
                examples.append([entities_to_id[lhs], relations_to_id[rel], entities_to_id[rhs]])

        if len(examples) == 0:
            raise ValueError("No valid triples found in {}".format(file_path))
        if skipped > 0:
            print("{} skipped {} invalid triples".format(f, skipped))

        out = open(Path(out_path) / (f + '.pickle'), 'wb')
        pickle.dump(np.array(examples).astype('uint64'), out)
        out.close()

    for feature_file in ['img_features.pth', 'text_features.pth', 'numeric_features.pth']:
        src = os.path.join(path, feature_file)
        if os.path.exists(src):
            copy_feature_file(src, os.path.join(out_path, feature_file), n_entities)

    create_filters_and_probas(out_path, files, n_entities, n_relations)

def prepare_dataset(path, name):
    files = ['train', 'valid', 'test']
    entities, relations = set(), set()
    for f in files:
        file_path = os.path.join(path, f)
        to_read = open(file_path, 'r')
        for line in to_read.readlines():
            items = parse_triple(line)
            if items is None:
                continue
            lhs, rel, rhs = items
            entities.add(lhs)
            entities.add(rhs)
            relations.add(rel)
        to_read.close()

    entities_to_id = {x: i for (i, x) in enumerate(sorted(entities))}
    relations_to_id = {x: i for (i, x) in enumerate(sorted(relations))}
    print("{} entities and {} relations".format(len(entities), len(relations)))
    n_relations = len(relations)
    n_entities = len(entities)
    os.makedirs(os.path.join(DATA_PATH, name), exist_ok=True)
    for (dic, f) in zip([entities_to_id, relations_to_id], ['ent_id', 'rel_id']):
        ff = open(os.path.join(DATA_PATH, name, f), 'w+')
        for (x, i) in dic.items():
            ff.write("{}\t{}\n".format(x, i))
        ff.close()

    for f in files:
        file_path = os.path.join(path, f)
        to_read = open(file_path, 'r')
        examples = []
        for line in to_read.readlines():
            items = parse_triple(line)
            if items is None:
                continue
            lhs, rel, rhs = items
            try:
                examples.append([entities_to_id[lhs], relations_to_id[rel], entities_to_id[rhs]])
            except ValueError:
                continue
        out = open(Path(DATA_PATH) / name / (f + '.pickle'), 'wb')
        pickle.dump(np.array(examples).astype('uint64'), out)
        out.close()

    create_filters_and_probas(os.path.join(DATA_PATH, name), files, n_entities, n_relations)


if __name__ == "__main__":
    datasets = ['WN18RR', 'FB237', 'YAGO3-10', 'Atomic', 'conceptnet-100k', 'MKG-W', 'MKG-Y', 'DB15K']
    multimodal_datasets = {'MKG-W', 'MKG-Y', 'DB15K'}
    for d in datasets:
        print("Preparing dataset {}".format(d))
        try:
            src_path = os.path.join('./src_data', d)
            if not os.path.exists(os.path.join(src_path, 'train')):
                print("{} does not exist. skipping...".format(src_path))
                continue
            if d in multimodal_datasets:
                prepare_multimodal_dataset(src_path, d)
            else:
                prepare_dataset(src_path, d)
        except OSError as e:
            if e.errno == errno.EEXIST:
                print(e)
                print("File exists. skipping...")
            else:
                raise
