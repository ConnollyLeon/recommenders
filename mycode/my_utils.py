def generate_entity_relations_dict(dir_path='wikidata-graph'):
    f = open(dir_path+'wikidata-graph.tsv')

    line = f.readline()
    entity_relations_dict = {}
    while line:
        head, r, tail = line.strip().split('\t')
        if head in entity_relations_dict:
            entity_relations_dict[head].add((r, 1))
        else:
            entity_relations_dict[head] = {(r, 1)}
        if tail in entity_relations_dict:
            entity_relations_dict[tail].add((r, -1))
        else:
            entity_relations_dict[tail] = {(r, -1)}
        line = f.readline()

    import pickle

    pickle.dump(entity_relations_dict, open(dir_path+'entity_relations_dict.pkl', 'wb'))




def generate_entity_embeddings():
    '''generate all entity embeddings, and entity2index.txt
    '''
    # Combine train, valid, test entity_embedding.vec
    embedding_dict = {}
    f = open('train_entity_embedding.vec')
    lines = f.readlines()
    f.close()
    for line in lines:
        embedding_dict[line.strip().split()[0]] = line.strip().split()[1:]

    f = open('test_entity_embedding.vec')
    lines = f.readlines()
    f.close()
    for line in lines:
        embedding_dict[line.strip().split()[0]] = line.strip().split()[1:]

    f = open('valid_entity_embedding.vec')
    lines = f.readlines()
    f.close()
    for line in lines:
        embedding_dict[line.strip().split()[0]] = line.strip().split()[1:]
    import numpy as np
    for key in embedding_dict:
        embedding_dict[key] = np.asarray(list(map(float, embedding_dict[key])))
    entity2index = {}
    entity_index = 1
    for key in embedding_dict:
        entity2index[key] = entity_index
        entity_index += 1
    entity_embeddings = np.zeros([entity_index, 100])
    for entity in entity2index:
        entity_embeddings[entity2index[entity]] = embedding_dict[entity]
    entity_embedding_path = 'entity_embeddings_5w_100_all.npy'
    np.save(entity_embedding_path, entity_embeddings)
    f = open('entity2index.txt', 'w')
    for key in entity2index:
        f.write(key + '\t' + str(entity2index[key]) + '\n')
    f.close()
    return entity_embeddings, entity2index


def generate_context_embedding_vec():
    '''
    Precondition: 1. need to get entity_relations_dict first
    2. Need to run generate entity_embedding_vec first. For getting entity2index.txt

    '''
    f = open("wikidata-graph/entity_relations_dict.pkl",'rb')
    import pickle
    entity_relations_dict = pickle.load(f)
    f.close()

    f = open('relation_embedding.vec', 'r')
    lines = f.readlines()
    f.close()

    entity_embeddings, entity2index = generate_entity_embeddings()

    import numpy as np
    relation_embedding_dict = {}
    for line in lines:
        relation_embedding_dict[line.strip().split()[0]] = line.strip().split()[1:]
    for key in relation_embedding_dict:
        relation_embedding_dict[key] = np.asarray(list(map(float, relation_embedding_dict[key])))
    context_embeddings = np.copy(entity_embeddings)
    for key in entity2index:
        for relation, pos_neg in entity_relations_dict[key]:
            context_embeddings[entity2index[key]] += relation_embedding_dict[relation] * pos_neg
