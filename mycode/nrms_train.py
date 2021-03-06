import sys

sys.path.append("../../")
from reco_utils.recommender.newsrec.newsrec_utils import prepare_hparams
from reco_utils.recommender.newsrec.models.nrms import NRMSModel
from reco_utils.recommender.newsrec.io.mind_iterator import MINDIterator
import os

print("System version: {}".format(sys.version))

epochs = 8
seed = 42
MIND_type = 'large'

data_path = './'

train_news_file = os.path.join(data_path, 'train', r'news.tsv')
train_behaviors_file = os.path.join(data_path, 'train', r'behaviors.tsv')
valid_news_file = os.path.join(data_path, 'valid', r'news.tsv')
valid_behaviors_file = os.path.join(data_path, 'valid', r'behaviors.tsv')

wordEmb_file = os.path.join(data_path, "utils", "embedding_all.npy")
userDict_file = os.path.join(data_path, "utils", "uid2index.pkl")
wordDict_file = os.path.join(data_path, "utils", "word_dict_all.pkl")
entityDict_file = os.path.join(data_path, "utils", "entity_dict_all.pkl")
# entity_embedding_file = os.path.join(data_path, "utils", "entity_embeddings_5w_100_all.npy")
# context_embedding_file = os.path.join(data_path, "utils", "context_embeddings_5w_100_all.npy")
entity_embedding_file = None
context_embedding_file = None
yaml_file = os.path.join(data_path, "utils", r'nrms.yaml')

# mind_url, mind_train_dataset, mind_dev_dataset, mind_utils = get_mind_data_set(MIND_type)
#
# if not os.path.exists(train_news_file):
#     download_deeprec_resources(mind_url, os.path.join(data_path, 'train'), mind_train_dataset)
#
# if not os.path.exists(valid_news_file):
#     download_deeprec_resources(mind_url, \
#                                os.path.join(data_path, 'valid'), mind_dev_dataset)
# if not os.path.exists(yaml_file):
#     download_deeprec_resources(r'https://recodatasets.blob.core.windows.net/newsrec/', \
#                                os.path.join(data_path, 'utils'), mind_utils)
#
hparams = prepare_hparams(yaml_file, wordEmb_file=wordEmb_file, \
                          wordDict_file=wordDict_file, userDict_file=userDict_file, \
                          epochs=epochs, entityEmb_file=entity_embedding_file, contextEmb_file=context_embedding_file, \
                          entityDict_file=entityDict_file,
                          show_step=10)
print(hparams)

iterator = MINDIterator
model = NRMSModel(hparams, iterator, seed=seed)
print(model.model.summary())

# from tensorflow.keras.utils import plot_model
# plot_model(model.model, to_file='model.png',show_shapes=True,show_layer_names=True)

# 读取模型再接着训练看看性能会不会再好一点
# model_path = os.path.join(data_path, "model")
# model.model.load_weights(os.path.join(model_path, "base_nrms_ckpt"))

model.fit(train_news_file, train_behaviors_file, valid_news_file, valid_behaviors_file)

#
# res_syn = model.run_eval(valid_news_file, valid_behaviors_file)
# print(res_syn)
model_path = os.path.join(data_path, "model")
os.makedirs(model_path, exist_ok=True)

model.model.save_weights(os.path.join(model_path, "base_nrms_ckpt_epoch_8"))

del model

model = NRMSModel(hparams, iterator, seed=seed)
model.model.load_weights(os.path.join(model_path, "base_nrms_ckpt_epoch_8"))

test_news_file = os.path.join(data_path, 'test', r'news.tsv')
test_behaviors_file = os.path.join(data_path, 'test', r'behaviors.tsv')
group_impr_indexes, group_labels, group_preds = model.run_fast_eval(test_news_file, test_behaviors_file, test=1)
#
import numpy as np
from tqdm import tqdm

with open(os.path.join(data_path, 'base_prediction.txt'), 'w') as f:
    for impr_index, preds in tqdm(zip(group_impr_indexes, group_preds)):
        impr_index += 1
        pred_rank = (np.argsort(np.argsort(preds)[::-1]) + 1).tolist()
        pred_rank = '[' + ','.join([str(i) for i in pred_rank]) + ']'
        f.write(' '.join([str(impr_index), pred_rank]) + '\n')

import zipfile

f = zipfile.ZipFile(os.path.join(data_path, 'base_prediction.zip'), 'w', zipfile.ZIP_DEFLATED)
f.write(os.path.join(data_path, 'base_prediction.txt'), arcname='base_prediction.txt')
f.close()
