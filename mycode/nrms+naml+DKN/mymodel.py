# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from bert4keras.backend import keras
from tensorflow.keras import layers
import numpy as np

from bert4keras.layers import Loss, MultiHeadAttention
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer, load_vocab
from bert4keras.optimizers import Adam
from bert4keras.snippets import sequence_padding, open
from bert4keras.snippets import DataGenerator, AutoRegressiveDecoder

from base_model import BaseModel
from layers import AttLayer2, SelfAttention

__all__ = ["NRMSModel"]

print('keras version:', keras.__version__)


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
        if hparams['use_bert']:

            # from transformers import TFBertModel
            # self.bert_model = TFBertModel.from_pretrained( '/Users/connolly/Documents/Study/Useful dataset/chinese_L-12_H-768_A-12')
            self.bert_model = build_transformer_model(
                hparams['bert_config_path'],
                hparams['bert_checkpoint_path'],
                name='BERT_encoder',
                # sequence_length=hparams['title_size']*hparams['his_size'],
                segment_vocab_size=-1,

            )
            # for i in range(10):
            #     self.bert_model.layers[i].trainable = False
            print(self.bert_model.summary())
            # from tensorflow.keras.utils import plot_model
            # plot_model(self.bert_model, to_file='bert_model.png',show_shapes=True,show_layer_names=True)

        else:
            self.word2vec_embedding = self._init_embedding(hparams['wordEmb_file'])
        if hparams['entityEmb_file'] is not None:
            self.entity2vec_embedding = self._init_embedding(hparams['entityEmb_file'])
        if hparams['contextEmb_file'] is not None:
            self.context2vec_embedding = self._init_embedding(hparams['contextEmb_file'])

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
        if self.hparams['entityEmb_file'] is not None:
            input_feat = [
                batch_data["clicked_title_batch"],
                batch_data["clicked_title_segment_batch"],  # LP add
                batch_data["clicked_title_entity_batch"],
                batch_data["clicked_ab_batch"],
                batch_data["clicked_vert_batch"],
                batch_data["clicked_subvert_batch"],
                batch_data["candidate_title_batch"],
                batch_data["candidate_title_segment_batch"],  # LP add
                batch_data["candidate_title_entity_batch"],
                batch_data["candidate_ab_batch"],
                batch_data["candidate_vert_batch"],
                batch_data["candidate_subvert_batch"],

            ]
        else:
            input_feat = [
                batch_data["clicked_title_batch"],
                batch_data["clicked_title_segment_batch"],  # LP add
                batch_data["candidate_title_batch"],
                batch_data["candidate_title_segment_batch"],  # LP add
                batch_data["clicked_ab_batch"],
                batch_data["clicked_vert_batch"],
                batch_data["clicked_subvert_batch"],
                batch_data["candidate_ab_batch"],
                batch_data["candidate_vert_batch"],
                batch_data["candidate_subvert_batch"],
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
            batch_data["clicked_title_segment_batch"],
            batch_data["clicked_title_entity_batch"],
            batch_data["clicked_ab_batch"],
            batch_data["clicked_vert_batch"],
            batch_data["clicked_subvert_batch"],
        ]
        input_feature = np.concatenate(input_feat, axis=-1)
        return input_feature

    def _get_news_feature_from_iter(self, batch_data):
        """ get input of news encoder
        Args:
            batch_data: input batch data from news iterator

        Returns:
            array: input news feature (candidate title batch)
        """
        input_feat = [
            batch_data["candidate_title_batch"],
            batch_data["candidate_title_segment_batch"],
            batch_data["candidate_title_entity_batch"],
            batch_data["candidate_ab_batch"],
            batch_data["candidate_vert_batch"],
            batch_data["candidate_subvert_batch"],
        ]

        input_feature = np.concatenate(input_feat, axis=-1)
        return input_feature

    def _build_graph(self):
        """Build NRMS model and scorer.

        Returns:
            obj: a model used to train.
            obj: a model used to evaluate and inference.
        """
        model, scorer = self._build_nrms()
        return model, scorer

    def _build_userencoder(self, newsencoder):
        """The main function to create user encoder of NRMS.

        Args:
            titleencoder(obj): the news encoder of NRMS.

        Return:
            obj: the user encoder of NRMS.
        """
        hparams = self.hparams
        his_input_title_entity_body_verts = keras.Input(
            shape=(hparams['his_size'], hparams['title_size'] * 3 + hparams['body_size'] + 2), dtype="int32",
            name='ue_his_input'
        )
        #
        # his_input_segment = keras.Input(
        #     shape=(hparams.his_size, hparams.title_size), dtype="int32", name='ue_his_input_segment'
        # )
        # embedded_sequences_title = layers.TimeDistributed(self.bert_model)(
        #     his_input_title)  # TODO shape可能有问题 (-1, 50,30,768)

        # embedded_sequences_title = keras.layers.Reshape((hparams['his_size'], hparams['title_size'], 768),
        #                                                 name='embedded_sequences_title_reshape')(
        #     embedded_sequences_title)

        # click_title_presents = layers.TimeDistributed(titleencoder,
        # name='news_time_distributed')(his_input_title_entity_body_verts)
        click_title_presents = layers.TimeDistributed(newsencoder,
                                                      name='news_time_distributed')(
            his_input_title_entity_body_verts
        )
        # y = SelfAttention(hparams['head_num'], hparams['head_dim'], seed=self.seed)(
        #     [click_title_presents] * 3
        # )

        y = SelfAttention(hparams['head_num'], hparams['head_dim'], seed=self.seed)([click_title_presents] * 3)
        # if entityencoder is not None:
        #     his_input_title_entity = keras.Input(shape=(hparams['his_size'], hparams['title_size']), dtype="int32",
        #                                          name='his_input_title_entity')
        #     click_title_entity_presents = layers.TimeDistributed(entityencoder, name='entity_time_distributed')(
        #         his_input_title_entity)
        #     entity_y = SelfAttention(hparams['head_num'], hparams['head_dim'], seed=self.seed)(
        #         [click_title_entity_presents] * 3
        #     )
        #     if contextencoder is not None:
        #         click_title_context_presents = layers.TimeDistributed(contextencoder, name='context_time_distributed')(
        #             his_input_title_entity)
        #         context_y = SelfAttention(hparams['head_num'], hparams['head_dim'], seed=self.seed)(
        #             [click_title_context_presents] * 3
        #         )
        #         y = layers.Concatenate()([y, entity_y, context_y])
        #     else:
        #         y = layers.Concatenate()([y, entity_y])

        user_present = AttLayer2(hparams['attention_hidden_dim'], seed=self.seed)(y)
        # if entityencoder is not None:
        #     model = keras.Model(inputs=[his_input_title, his_input_title_entity],
        #                         outputs=user_present,
        #                         name="user_encoder")
        # else:
        model = keras.Model(his_input_title_entity_body_verts, user_present, name="user_encoder")
        return model

    def _build_newsencoder(self, name, entity_embedding_layer, context_embedding_layer):
        """The main function to create news encoder of NRMS.

        Args:
            bert_model(obj): a bert model. # LP modified

        Return:
            obj: the news encoder of NRMS.
        """
        hparams = self.hparams
        input_title_entity_body_verts = keras.Input(
            shape=(hparams['title_size'] * 3 + hparams['body_size'] + 2), dtype="int32",
            name='ue_his_input')

        sequences_input_title = layers.Lambda(lambda x: x[:, : hparams['title_size']])(
            input_title_entity_body_verts
        )
        sequences_input_entity = layers.Lambda(lambda x: x[:, hparams['title_size'] * 2: hparams['title_size'] * 3])(
            input_title_entity_body_verts
        )

        sequences_input_body = layers.Lambda(
            lambda x: x[:, hparams['title_size'] * 3: hparams['title_size'] * 3 + hparams['body_size']]
        )(input_title_entity_body_verts)
        input_vert = layers.Lambda(
            lambda x: x[:,
                      hparams['title_size'] * 3
                      + hparams['body_size']:
                      hparams['title_size'] * 3
                      + hparams['body_size']
                      + 1, ]
        )(input_title_entity_body_verts)
        input_subvert = layers.Lambda(
            lambda x: x[:, hparams['title_size'] * 3 + hparams['body_size'] + 1:]
        )(input_title_entity_body_verts)
        # sequences_input_segment = keras.Input(shape=(hparams.title_size,), dtype="int32",
        #                                       name='sequences_input_segment')
        # embedded_sequences_title = layers.TimeDistributed(self.bert_model)(
        #     sequences_input_title)  # TODO shape可能有问题
        # embedded_sequences_title = keras.Input(shape=(hparams['title_size'],
        #                                               768,),
        #                                        name='embedded_sequences_title')  # TODO shape可能有问题 (?, 30, 768)

        title_repr = self._build_titleencoder()(sequences_input_title)
        body_repr = self._build_bodyencoder()(sequences_input_body)
        entity_repr = self._build_eneityencoder(entity_embedding_layer, 'entity_encoder')(sequences_input_entity)
        context_repr = self._build_eneityencoder(context_embedding_layer, 'context_encoder')(sequences_input_entity)
        vert_repr = self._build_vertencoder()(input_vert)
        subvert_repr = self._build_subvertencoder()(input_subvert)

        concate_repr = layers.Concatenate(axis=-2)(
            [title_repr, entity_repr, context_repr, body_repr, vert_repr, subvert_repr])

        news_repr = AttLayer2(hparams['attention_hidden_dim'], seed=self.seed)(
            concate_repr
        )

        model = keras.Model(input_title_entity_body_verts, news_repr, name='news_encoder')
        return model

    def _build_titleencoder(self):
        hparams = self.hparams
        sequences_input_title = keras.Input(shape=(hparams['title_size'],), dtype="int32")
        embedded_sequences_title = self.bert_model(sequences_input_title)

        y = layers.Dropout(hparams['dropout'])(embedded_sequences_title)
        y = SelfAttention(hparams['head_num'], hparams['head_dim'], seed=self.seed)(
            [y, y, y])
        y = layers.Dropout(hparams['dropout'])(y)

        pred_title = AttLayer2(hparams['attention_hidden_dim'], seed=self.seed)(y)
        pred_title = layers.Reshape((1, hparams['filter_num']))(pred_title)
        model = keras.Model(sequences_input_title, pred_title, name="title_encoder")
        return model

    def _build_bodyencoder(self):
        hparams = self.hparams
        sequences_input_body = keras.Input(shape=(hparams['title_size'],), dtype="int32")
        embedded_sequences_body = self.bert_model(sequences_input_body)

        y = layers.Dropout(hparams['dropout'])(embedded_sequences_body)
        y = SelfAttention(hparams['head_num'], hparams['head_dim'], seed=self.seed)(
            [y, y, y])
        y = layers.Dropout(hparams['dropout'])(y)

        pred_body = AttLayer2(hparams['attention_hidden_dim'], seed=self.seed)(y)
        pred_body = layers.Reshape((1, hparams['filter_num']))(pred_body)
        model = keras.Model(sequences_input_body, pred_body, name="body_encoder")
        return model

    def _build_eneityencoder(self, embedding_layer, name):
        """The main function to create news encoder of NRMS.

        Args:
            embedding_layer(obj): embedding layer. # LP modified

        Return:
            obj: the news encoder of NRMS.
        """
        hparams = self.hparams
        sequences_input_entity = keras.Input(shape=(hparams['title_size'],), dtype="int32",
                                             name='sequences_input_title')
        embedded_sequences_entity = embedding_layer(sequences_input_entity)  # TODO shape可能有问题

        y = layers.Dropout(hparams['dropout'])(embedded_sequences_entity)
        y = SelfAttention(hparams['head_num'], hparams['head_dim'], seed=self.seed)([y, y, y])
        y = layers.Dropout(hparams['dropout'])(y)

        pred_entity = AttLayer2(hparams['attention_hidden_dim'], seed=self.seed)(y)
        pred_entity = layers.Reshape((1, hparams['filter_num']))(pred_entity)
        model = keras.Model(sequences_input_entity, pred_entity, name=name)
        return model

    def _build_vertencoder(self):
        """build vert encoder of NAML news encoder.

        Return:
            obj: the vert encoder of NAML.
        """
        hparams = self.hparams
        input_vert = keras.Input(shape=(1,), dtype="int32")
        vert_embedding = layers.Embedding(
            hparams['vert_num'], hparams['vert_emb_dim'], trainable=True
        )
        vert_emb = vert_embedding(input_vert)
        pred_vert = layers.Dense(
            hparams['filter_num'],
            activation=hparams['dense_activation'],
            bias_initializer=keras.initializers.Zeros(),
            kernel_initializer=keras.initializers.glorot_uniform(seed=self.seed),
        )(vert_emb)
        pred_vert = layers.Reshape((1, hparams['filter_num']))(pred_vert)

        model = keras.Model(input_vert, pred_vert, name="vert_encoder")
        return model

    def _build_subvertencoder(self):
        """build subvert encoder of NAML news encoder.

        Return:
            obj: the subvert encoder of NAML.
        """
        hparams = self.hparams
        input_subvert = keras.Input(shape=(1,), dtype="int32")

        subvert_embedding = layers.Embedding(
            hparams['subvert_num'], hparams['subvert_emb_dim'], trainable=True
        )

        subvert_emb = subvert_embedding(input_subvert)
        pred_subvert = layers.Dense(
            hparams['filter_num'],
            activation=hparams['dense_activation'],
            bias_initializer=keras.initializers.Zeros(),
            kernel_initializer=keras.initializers.glorot_uniform(seed=self.seed),
        )(subvert_emb)
        pred_subvert = layers.Reshape((1, hparams['filter_num']))(pred_subvert)

        model = keras.Model(input_subvert, pred_subvert, name="subvert_encoder")
        return model

    def news(self, batch_news_input):
        news_input = self._get_news_feature_from_iter(batch_news_input)
        # embedded_news = self.bert_model.predict_on_batch(news_input[0])
        news_vec = self.newsencoder.predict_on_batch(news_input)  # title
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
            shape=(hparams['his_size'],
                   hparams['title_size'],), dtype="int32", name='his_input_title'
        )
        his_input_segment = keras.Input(
            shape=(hparams['his_size'],
                   hparams['title_size'],), dtype="int32", name="his_input_segment"  # LP add
        )
        his_input_body = keras.Input(
            shape=(hparams['his_size'], hparams['body_size']), dtype="int32"
        )
        his_input_vert = keras.Input(shape=(hparams['his_size'], 1), dtype="int32")
        his_input_subvert = keras.Input(shape=(hparams['his_size'], 1), dtype="int32")

        his_input_title_entity = keras.Input(
            shape=(hparams['his_size'], hparams['title_size']), dtype="int32", name='his_input_title_entity'
        )

        pred_input_title = keras.Input(
            shape=(hparams['npratio'] + 1,
                   hparams['title_size'],), dtype="int32", name='pred_input_title'
        )
        pred_input_segment = keras.Input(
            shape=(hparams['npratio'] + 1,
                   hparams['title_size'],), dtype="int32", name='pred_input_segment'
        )
        pred_input_title_entity = keras.Input(
            shape=(hparams['npratio'] + 1, hparams['title_size']), dtype="int32", name='pred_input_title_entity'
        )

        pred_input_body = keras.Input(
            shape=(hparams['npratio'] + 1, hparams['body_size']), dtype="int32"
        )
        pred_input_vert = keras.Input(shape=(hparams['npratio'] + 1, 1), dtype="int32")
        pred_input_subvert = keras.Input(shape=(hparams['npratio'] + 1, 1), dtype="int32")

        pred_input_title_one = keras.Input(
            shape=(1, hparams['title_size']), dtype="int32", name='pred_input_title_one'
        )

        # pred_title_one_reshape = layers.Reshape((hparams['title_size'],),
        #                                         name='pred_title_one_reshape')(pred_input_title_one)

        pred_input_title_segment_one = keras.Input(
            shape=(1, hparams['title_size']), dtype="int32", name='pred_input_title_segment_one'  # LP add
        )

        pred_input_entity_one = keras.Input(
            shape=(1, hparams['title_size'],), dtype="int32", name='pred_input_title_one_entity'
        )

        # pred_title_segment_one_reshape = layers.Reshape((hparams['title_size'],),
        #                                                 name='pred_title_segment_one_reshape')(
        #     pred_input_title_segment_one)  # LP add 

        pred_input_body_one = keras.Input(shape=(1, hparams['body_size'],), dtype="int32")
        pred_input_vert_one = keras.Input(shape=(1, 1), dtype="int32")
        pred_input_subvert_one = keras.Input(shape=(1, 1), dtype="int32")

        his_title_body_verts = layers.Concatenate(axis=-1)(
            [his_input_title, his_input_segment, his_input_title_entity, his_input_body, his_input_vert,
             his_input_subvert]
        )

        pred_title_body_verts = layers.Concatenate(axis=-1)(
            [pred_input_title, pred_input_segment, pred_input_title_entity, pred_input_body, pred_input_vert,
             pred_input_subvert]
        )

        pred_title_body_verts_one = layers.Concatenate(axis=-1)(
            [
                pred_input_title_one,
                pred_input_title_segment_one,
                pred_input_entity_one,
                pred_input_body_one,
                pred_input_vert_one,
                pred_input_subvert_one,
            ]
        )
        pred_title_body_verts_one = layers.Reshape((-1,))(pred_title_body_verts_one)

        # embedding_layer = layers.Embedding(
        #     self.word2vec_embedding.shape[0],
        #     hparams.word_emb_dim,
        #     weights=[self.word2vec_embedding],
        #     trainable=True,
        # )

        # use bert_model 来代替 word embedding

        # entity_embedding_layer = None
        # context_embedding_layer = None

        entity_embedding_layer = layers.Embedding(
            self.entity2vec_embedding.shape[0],
            self.entity2vec_embedding.shape[1],
            weights=[self.entity2vec_embedding],
            trainable=True,
            name='entity_embedding_layer'
        )

        context_embedding_layer = layers.Embedding(
            self.context2vec_embedding.shape[0],
            self.context2vec_embedding.shape[1],
            weights=[self.context2vec_embedding],
            trainable=True,
            name='context_embedding_layer'
        )

        self.newsencoder = self._build_newsencoder('news_encoder', entity_embedding_layer, context_embedding_layer)
        self.userencoder = self._build_userencoder(self.newsencoder)
        # from tensorflow.keras.utils import plot_model
        # plot_model(self.userencoder, to_file='userencoder_model.png',show_shapes=True,show_layer_names=True)

        user_present = self.userencoder(his_title_body_verts)
        news_present = layers.TimeDistributed(self.newsencoder)(pred_title_body_verts)
        news_present_one = self.newsencoder(pred_title_body_verts_one)

        preds = layers.Dot(axes=-1)([news_present, user_present])
        preds = layers.Activation(activation="softmax")(preds)

        pred_one = layers.Dot(axes=-1)([news_present_one, user_present])
        pred_one = layers.Activation(activation="sigmoid")(pred_one)

        model = keras.Model(
            [
                his_input_title,
                his_input_segment,
                his_input_title_entity,
                his_input_body,
                his_input_vert,
                his_input_subvert,
                pred_input_title,
                pred_input_segment,
                pred_input_title_entity,
                pred_input_body,
                pred_input_vert,
                pred_input_subvert,
            ],
            preds,
        )

        scorer = keras.Model(
            [
                his_input_title,
                his_input_segment,
                his_input_title_entity,
                his_input_body,
                his_input_vert,
                his_input_subvert,
                pred_input_title_one,
                pred_input_title_segment_one,
                pred_input_entity_one,
                pred_input_body_one,
                pred_input_vert_one,
                pred_input_subvert_one,
            ],
            pred_one,
        )

        return model, scorer
