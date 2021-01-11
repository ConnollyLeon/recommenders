# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import tensorflow.keras as keras
from tensorflow.keras import layers
import numpy as np

from reco_utils.recommender.newsrec.models.base_model import BaseModel
from reco_utils.recommender.newsrec.models.layers import AttLayer2, SelfAttention

__all__ = ["NRMSModel"]


class NRMSModel(BaseModel):
    """NRMS model(Neural News Recommendation with Multi-Head Self-Attention)

    Chuhan Wu, Fangzhao Wu, Suyu Ge, Tao Qi, Yongfeng Huang,and Xing Xie, "Neural News
    Recommendation with Multi-Head Self-Attention" in Proceedings of the 2019 Conference 
    on Empirical Methods in Natural Language Processing and the 9th International Joint Conference 
    on Natural Language Processing (EMNLP-IJCNLP)

    Attributes:
        word2vec_embedding (numpy.array): Pretrained word embedding matrix.
        hparam (obj): Global hyper-parameters.
    """

    def __init__(
            self, hparams, iterator_creator, seed=None,
    ):
        """Initialization steps for NRMS.
        Compared with the BaseModel, NRMS need word embedding.
        After creating word embedding matrix, BaseModel's __init__ method will be called.
        
        Args:
            hparams (obj): Global hyper-parameters. Some key setttings such as head_num and head_dim are there.
            iterator_creator_train(obj): NRMS data loader class for train data.
            iterator_creator_test(obj): NRMS data loader class for test and validation data
        """
        self.word2vec_embedding = self._init_embedding(hparams.wordEmb_file)
        if hparams.entityEmb_file is not None:
            self.entity2vec_embedding = self._init_embedding(hparams.entityEmb_file)
        if hparams.contextEmb_file is not None:
            self.context2vec_embedding = self._init_embedding(hparams.contextEmb_file)

        super().__init__(
            hparams, iterator_creator, seed=seed,
        )

    def _get_input_label_from_iter(self, batch_data):
        """ get input and labels for trainning from iterator

        Args:
            batch data: input batch data from iterator

        Returns:
            list: input feature fed into model (clicked_title_batch & candidate_title_batch)
            array: labels
        """
        input_feat = [
            batch_data["clicked_title_batch"],
            batch_data["candidate_title_batch"],
            batch_data["clicked_title_entity_batch"],
            batch_data["candidate_title_entity_batch"]

        ]
        input_label = batch_data["labels"]
        return input_feat, input_label

    def _get_user_feature_from_iter(self, batch_data):
        """ get input of user encoder
        Args:
            batch_data: input batch data from user iterator

        Returns:
            array: input user feature (clicked title batch)
        """
        input_feat = [
            batch_data["clicked_title_batch"],
            batch_data["clicked_title_entity_batch"]
        ]
        return input_feat

    def _get_news_feature_from_iter(self, batch_data):
        """ get input of news encoder
        Args:
            batch_data: input batch data from news iterator

        Returns:
            array: input news feature (candidate title batch)
        """
        input_feat = [
            batch_data["candidate_title_batch"],
            batch_data["candidate_title_entity_batch"]
        ]

        return input_feat

    def _build_graph(self):
        """Build NRMS model and scorer.

        Returns:
            obj: a model used to train.
            obj: a model used to evaluate and inference.
        """
        hparams = self.hparams
        model, scorer = self._build_nrms()
        return model, scorer

    def _build_userencoder(self, titleencoder, entityencoder, contextencoder):
        """The main function to create user encoder of NRMS.

        Args:
            titleencoder(obj): the news encoder of NRMS. 

        Return:
            obj: the user encoder of NRMS.
        """
        hparams = self.hparams
        his_input_title = keras.Input(
            shape=(hparams.his_size, hparams.title_size), dtype="int32"
        )
        click_title_presents = layers.TimeDistributed(titleencoder, name='news_time_distributed')(his_input_title)
        y = SelfAttention(hparams.head_num, hparams.head_dim, seed=self.seed)(
            [click_title_presents] * 3
        )
        if entityencoder is not None:
            his_input_title_entity = keras.Input(shape=(hparams.his_size, hparams.title_size), dtype="int32")
            click_title_entity_presents = layers.TimeDistributed(entityencoder, name='entity_time_distributed')(
                his_input_title_entity)
            entity_y = SelfAttention(hparams.head_num, hparams.head_dim, seed=self.seed)(
                [click_title_entity_presents] * 3
            )
            if contextencoder is not None:
                click_title_context_presents = layers.TimeDistributed(contextencoder, name='context_time_distributed')(
                    his_input_title_entity)
                context_y = SelfAttention(hparams.head_num, hparams.head_dim, seed=self.seed)(
                    [click_title_context_presents] * 3
                )
                y = layers.Concatenate()([y, entity_y, context_y])
            else:
                y = layers.Concatenate()([y, entity_y])

        user_present = AttLayer2(hparams.attention_hidden_dim, seed=self.seed)(y)
        if entityencoder is not None:
            model = keras.Model(inputs=[his_input_title, his_input_title_entity], outputs=user_present,
                                name="user_encoder")
        else:
            model = keras.Model(his_input_title, user_present, name="user_encoder")
        return model

    def _build_newsencoder(self, embedding_layer, name):
        """The main function to create news encoder of NRMS.

        Args:
            embedding_layer(obj): a word embedding layer.
        
        Return:
            obj: the news encoder of NRMS.
        """
        hparams = self.hparams
        sequences_input_title = keras.Input(shape=(hparams.title_size,), dtype="int32")
        embedded_sequences_title = embedding_layer(sequences_input_title)

        y = layers.Dropout(hparams.dropout)(embedded_sequences_title)
        y = SelfAttention(hparams.head_num, hparams.head_dim, seed=self.seed)([y, y, y])
        y = layers.Dropout(hparams.dropout)(y)

        pred_title = AttLayer2(hparams.attention_hidden_dim, seed=self.seed)(y)

        model = keras.Model(sequences_input_title, pred_title, name=name)
        return model

    def news(self, batch_news_input):
        news_input = self._get_news_feature_from_iter(batch_news_input)
        news_vec = self.newsencoder.predict_on_batch(news_input[0])  # title
        if self.entityencoder:
            entity_vec = self.entityencoder.predict_on_batch(news_input[1])
            news_vec = np.concatenate([news_vec, entity_vec], -1)
            if self.contextencoder:
                context_vec = self.contextencoder.predict_on_batch(news_input[1])
                news_vec = np.concatenate([news_vec, context_vec], -1)
        news_index = batch_news_input["news_index_batch"]
        # news_vec = np.concatenate([news_vec, entity_vec, context_vec], -1)
        return news_index, news_vec

    def _build_nrms(self):
        """The main function to create NRMS's logic. The core of NRMS
        is a user encoder and a news encoder.
        
        Returns:
            obj: a model used to train.
            obj: a model used to evaluate and inference.
        """
        hparams = self.hparams

        his_input_title = keras.Input(
            shape=(hparams.his_size, hparams.title_size), dtype="int32", name='his_input_title'
        )
        pred_input_title = keras.Input(
            shape=(hparams.npratio + 1, hparams.title_size), dtype="int32", name='pred_input_title'
        )
        pred_input_title_one = keras.Input(
            shape=(1, hparams.title_size,), dtype="int32", name='pred_input_title_one'
        )
        pred_title_one_reshape = layers.Reshape((hparams.title_size,))(
            pred_input_title_one
        )

        embedding_layer = layers.Embedding(
            self.word2vec_embedding.shape[0],
            hparams.word_emb_dim,
            weights=[self.word2vec_embedding],
            trainable=True,
        )
        entity_embedding_layer = None
        context_embedding_layer = None
        if hparams.entityEmb_file is not None:
            his_input_title_entity = keras.Input(
                shape=(hparams.his_size, hparams.title_size), dtype="int32", name='his_input_title_entity'
            )
            pred_input_title_entity = keras.Input(
                shape=(hparams.npratio + 1, hparams.title_size), dtype="int32", name='pred_input_title_entity'
            )
            pred_input_title_one_entity = keras.Input(
                shape=(1, hparams.title_size,), dtype="int32", name='pred_input_title_one_entity'
            )
            pred_title_one_reshape_entity = layers.Reshape((hparams.title_size,))(
                pred_input_title_one_entity
            )
            entity_embedding_layer = layers.Embedding(
                self.entity2vec_embedding.shape[0],
                self.entity2vec_embedding.shape[1],
                weights=[self.entity2vec_embedding],
                trainable=True,
                name='entity_embedding_layer'
            )

            if hparams.contextEmb_file is not None:
                context_embedding_layer = layers.Embedding(
                    self.context2vec_embedding.shape[0],
                    self.context2vec_embedding.shape[1],
                    weights=[self.context2vec_embedding],
                    trainable=True,
                    name='context_embedding_layer'
                )

        titleencoder = self._build_newsencoder(embedding_layer, 'news_encoder')
        if hparams.entityEmb_file:
            entity_encoder = self._build_newsencoder(entity_embedding_layer, 'entity_encoder')
            if hparams.contextEmb_file:
                context_encoder = self._build_newsencoder(context_embedding_layer, 'context_encoder')
            else:
                context_encoder = None
        else:
            entity_encoder = None
            context_encoder = None
        self.userencoder = self._build_userencoder(titleencoder, entity_encoder, context_encoder)
        self.newsencoder = titleencoder
        self.entityencoder = entity_encoder
        self.contextencoder = context_encoder

        if hparams.entityEmb_file is not None:
            user_present = self.userencoder([his_input_title, his_input_title_entity])
            news_present = layers.TimeDistributed(self.newsencoder)(pred_input_title)
            news_present_one = self.newsencoder(pred_title_one_reshape)
            news_entity_present = layers.TimeDistributed(self.entityencoder)(pred_input_title_entity)
            news_entity_present_one = self.entityencoder(pred_title_one_reshape_entity)
            if hparams.contextEmb_file is not None:
                news_context_present = layers.TimeDistributed(self.contextencoder)(pred_input_title_entity)
                news_context_present_one = self.contextencoder(pred_title_one_reshape_entity)
                news_present = layers.Concatenate()([news_present, news_entity_present, news_context_present])
                news_present_one = layers.Concatenate()(
                    [news_present_one, news_entity_present_one, news_context_present_one])
            else:
                news_present = layers.Concatenate()([news_present, news_entity_present])
                news_present_one = layers.Concatenate()(
                    [news_present_one, news_entity_present_one])

        else:
            user_present = self.userencoder(his_input_title)
            news_present = layers.TimeDistributed(self.newsencoder)(pred_input_title)
            news_present_one = self.newsencoder(pred_title_one_reshape)

        preds = layers.Dot(axes=-1)([news_present, user_present])
        preds = layers.Activation(activation="softmax")(preds)

        pred_one = layers.Dot(axes=-1)([news_present_one, user_present])
        pred_one = layers.Activation(activation="sigmoid")(pred_one)
        if hparams.entityEmb_file is not None:
            model = keras.Model([his_input_title, pred_input_title, his_input_title_entity, pred_input_title_entity],
                                preds)
            scorer = keras.Model(
                [his_input_title, pred_input_title_one, his_input_title_entity, pred_input_title_one_entity], pred_one)
        else:
            model = keras.Model([his_input_title, pred_input_title], preds)
            scorer = keras.Model([his_input_title, pred_input_title_one], pred_one)

        return model, scorer
