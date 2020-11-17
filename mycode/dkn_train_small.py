import sys
sys.path.append("../")

import os
from tempfile import TemporaryDirectory
import logging
import tensorflow as tf

from reco_utils.dataset.download_utils import maybe_download
from reco_utils.dataset.mind import (download_mind,
                                     extract_mind,
                                     read_clickhistory,
                                     get_train_input,
                                     get_valid_input,
                                     get_user_history,
                                     get_words_and_entities,
                                     generate_embeddings)
from reco_utils.recommender.deeprec.deeprec_utils import prepare_hparams
from reco_utils.recommender.deeprec.models.dkn import DKN
from reco_utils.recommender.deeprec.io.dkn_iterator import DKNTextIterator

print(f"System version: {sys.version}")
print(f"Tensorflow version: {tf.__version__}")

# Temp dir
tmpdir = './'

# Logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stderr)
formatter = logging.Formatter("%(asctime)s %(levelname)s: %(message)s", datefmt='%I:%M:%S')
handler.setFormatter(formatter)
logger.handlers = [handler]

# Mind parameters
MIND_SIZE = "small"

# DKN parameters
epochs = 10
history_size = 50
batch_size = 100

# Paths
data_path = os.path.join(tmpdir, "mind-dkn")
train_file = os.path.join(data_path, "train_mind.txt")
valid_file = os.path.join(data_path, "valid_mind.txt")
user_history_file = os.path.join(data_path, "user_history.txt")
infer_embedding_file = os.path.join(data_path, "infer_embedding.txt")

train_zip, valid_zip = download_mind(size=MIND_SIZE, dest_path=data_path)
train_path, valid_path = extract_mind(train_zip, valid_zip)

train_session, train_history = read_clickhistory(train_path, "behaviors.tsv")
valid_session, valid_history = read_clickhistory(valid_path, "behaviors.tsv")
get_train_input(train_session, train_file)
get_valid_input(valid_session, valid_file)
get_user_history(train_history, valid_history, user_history_file)

train_news = os.path.join(train_path, "news.tsv")
valid_news = os.path.join(valid_path, "news.tsv")
news_words, news_entities = get_words_and_entities(train_news, valid_news)

train_entities = os.path.join(train_path, "entity_embedding.vec")
valid_entities = os.path.join(valid_path, "entity_embedding.vec")
news_feature_file, word_embeddings_file, entity_embeddings_file = generate_embeddings(
    data_path,
    news_words,
    news_entities,
    train_entities,
    valid_entities,
    max_sentence=10,
    word_embedding_dim=100,
)


# yaml_file = maybe_download(url="https://recodatasets.blob.core.windows.net/deeprec/deeprec/dkn/dkn_MINDsmall.yaml",
#                            work_directory=data_path)
# hparams = prepare_hparams(yaml_file,
#                           news_feature_file=news_feature_file,
#                           user_history_file=user_history_file,
#                           wordEmb_file=word_embeddings_file,
#                           entityEmb_file=entity_embeddings_file,
#                           epochs=epochs,
#                           history_size=history_size,
#                           batch_size=batch_size)
# model = DKN(hparams, DKNTextIterator)
#
# model.fit(train_file, valid_file)
#
# res = model.run_eval(valid_file)
# print(res)
