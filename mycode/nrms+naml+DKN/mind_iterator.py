# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import pickle

import numpy as np
import tensorflow as tf
from bert4keras.snippets import DataGenerator, AutoRegressiveDecoder
from bert4keras.tokenizers import Tokenizer, load_vocab
from bert4keras.snippets import sequence_padding,  to_array
from iterator import BaseIterator
from newsrec_utils import word_tokenize, newsample

__all__ = ["MINDIterator"]

# dict_path = '/vocab.txt'
dict_path = '/Users/connolly/Documents/Study/Useful dataset/chinese_L-12_H-768_A-12/vocab.txt'
maxlen = 256
batch_size = 16
tokenizer = Tokenizer(dict_path, do_lower_case=True)


class data_generator(DataGenerator):
    """数据生成器
    """

    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, (text, label) in self.sample(random):
            token_ids, segment_ids = tokenizer.encode(text, maxlen=maxlen)
            token_ids, segment_ids = to_array([token_ids,segment_ids])

            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_labels.append([label])
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = sequence_padding(batch_labels)
                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []


class MINDIterator(BaseIterator):
    """Train data loader for NAML model.
    The model require a special type of data format, where each instance contains a label, impresion id, user id,
    the candidate news articles and user's clicked news article. Articles are represented by title words,
    body words, verts and subverts. 

    Iterator will not load the whole data into memory. Instead, it loads data into memory
    per mini-batch, so that large files can be used as input data.

    Attributes:
        col_spliter (str): column spliter in one line.
        ID_spliter (str): ID spliter in one line.
        batch_size (int): the samples num in one batch.
        title_size (int): max word num in news title.
        his_size (int): max clicked news num in user click history.
        npratio (int): negaive and positive ratio used in negative sampling. -1 means no need of negtive sampling.
    """

    def __init__(
            self, hparams, npratio=-1, col_spliter="\t", ID_spliter="%",
    ):
        """Initialize an iterator. Create necessary placeholders for the model.
        
        Args:
            hparams (obj): Global hyper-parameters. Some key setttings such as head_num and head_dim are there.
            npratio (int): negaive and positive ratio used in negative sampling. -1 means no need of negtive sampling.
            col_spliter (str): column spliter in one line.
            ID_spliter (str): ID spliter in one line.
        """
        self.col_spliter = col_spliter
        self.ID_spliter = ID_spliter
        self.batch_size = hparams['batch_size']
        self.title_size = hparams['title_size']
        self.his_size = hparams['his_size']
        self.npratio = npratio

        self.word_dict = self.load_dict(hparams['wordDict_file'])
        self.entity_dict = self.load_dict(hparams['entityDict_file'])
        self.uid2index = self.load_dict(hparams['userDict_file'])

    def load_dict(self, file_path):
        """ load pickle file
        Args:
            file path (str): file path
        
        Returns:
            (obj): pickle load obj
        """
        with open(file_path, "rb") as f:
            return pickle.load(f)

    def init_news(self, news_file):
        """ init news information given news file, such as news_title_index and nid2index.
        Args:
            news_file: path of news file
        """

        self.nid2index = {}
        news_title = [""]
        title_entities = [""]  # every line in it: format like ['Q1', 'Q2', 'Q3']
        with tf.io.gfile.GFile(news_file, "r") as rd:
            for line in rd:
                nid, vert, subvert, title, ab, url, title_entity, _ = line.strip("\n").split(
                    self.col_spliter
                )

                if nid in self.nid2index:
                    continue

                self.nid2index[nid] = len(self.nid2index) + 1
                title = word_tokenize(title)
                news_title.append(title)

                entities = []
                title_entity = eval(title_entity)
                for dic in title_entity:
                    entities.append(dic['WikidataId'])
                title_entities.append(entities)

        self.news_title_index = np.zeros(
            (len(news_title), self.title_size), dtype="int32"
        )

        # LP code for bert tokenizer
        self.news_title_segment = np.zeros(
            (len(news_title), self.title_size), dtype="int32"
        )

        for news_index in range(len(news_title)):
            title = news_title[news_index]
            token_ids, segment_ids = tokenizer.encode(title, maxlen=self.title_size)
            if len(token_ids) > 30:
                self.news_title_index[news_index] = token_ids[:30]  # TODO 可能会有bug吗 (最大长度的问题）
                self.news_title_segment[news_index] = segment_ids[:30]
            else:
                self.news_title_index[news_index, :len(token_ids)] = token_ids
                self.news_title_segment[news_index, :len(token_ids)] = segment_ids

        # LP code for title_entity data extraction
        self.title_entity_index = np.zeros(
            (len(title_entities), self.title_size), dtype="int32"
        )

        # for news_index in range(len(news_title)):
        #     title = news_title[news_index]
        #     for word_index in range(min(self.title_size, len(title))):
        #         if title[word_index] in self.word_dict:
        #             self.news_title_index[news_index, word_index] = self.word_dict[
        #                 title[word_index].lower()
        #             ]

        for news_index in range(len(title_entities)):
            entities = title_entities[news_index]  # entities of the i_th news
            for entity_index in range(min(self.title_size, len(entities))):
                if entities[entity_index] in self.entity_dict:
                    # forming news entity matrix
                    self.title_entity_index[news_index, entity_index] = self.entity_dict[
                        entities[entity_index]
                    ]

    def init_behaviors(self, behaviors_file, test=False):
        """ init behavior logs given behaviors file.

        Args:
        behaviors_file: path of behaviors file
        """
        self.histories = []
        self.imprs = []
        self.labels = []
        self.impr_indexes = []
        self.uindexes = []

        with tf.io.gfile.GFile(behaviors_file, "r") as rd:
            impr_index = 0
            for line in rd:
                uid, time, history, impr = line.strip("\n").split(self.col_spliter)[-4:]

                history = [self.nid2index[i] for i in history.split()]
                history = [0] * (self.his_size - len(history)) + history[
                                                                 : self.his_size
                                                                 ]

                impr_news = [self.nid2index[i.split("-")[0]] for i in impr.split()]
                if not test:
                    label = [int(i.split("-")[1]) for i in impr.split()]
                else:
                    label = [1 for i in impr.split()]
                uindex = self.uid2index[uid] if uid in self.uid2index else 0

                self.histories.append(history)
                self.imprs.append(impr_news)
                self.labels.append(label)
                self.impr_indexes.append(impr_index)
                self.uindexes.append(uindex)
                impr_index += 1

    def parser_one_line(self, line):
        """Parse one behavior sample into feature values.
        if npratio is larger than 0, return negtive sampled result.
        
        Args:
            line (int): sample index.

        Returns:
            list: Parsed results including label, impression id , user id, 
            candidate_title_index, clicked_title_index.
        """

        if self.npratio > 0:
            impr_label = self.labels[line]
            impr = self.imprs[line]

            poss = []
            negs = []

            for news, click in zip(impr, impr_label):
                if click == 1:
                    poss.append(news)
                else:
                    negs.append(news)

            for p in poss:
                candidate_title_index = []
                candidate_title_segment = []
                impr_index = []
                user_index = []
                label = [1] + [0] * self.npratio

                n = newsample(negs, self.npratio)
                candidate_title_index = self.news_title_index[[p] + n]
                candidate_title_segment = self.news_title_segment[[p] + n] # LP add

                candidate_title_entity_index = self.title_entity_index[[p] + n]  # LP add
                click_title_index = self.news_title_index[self.histories[line]]
                click_title_segment = self.news_title_segment[self.histories[line]] # LP add
                click_title_entity_index = self.title_entity_index[self.histories[line]]  # LP add
                impr_index.append(self.impr_indexes[line])
                user_index.append(self.uindexes[line])

                yield (
                    label,
                    impr_index,
                    user_index,
                    candidate_title_index,
                    candidate_title_segment,
                    click_title_index,
                    click_title_segment,
                    candidate_title_entity_index,
                    click_title_entity_index,
                )
        else:
            impr_label = self.labels[line]
            impr = self.imprs[line]

            for news, label in zip(impr, impr_label):
                candidate_title_index = []
                candidate_title_segment = []
                impr_index = []
                user_index = []
                label = [label]

                candidate_title_index.append(self.news_title_index[news])
                candidate_title_segment.append(self.news_title_segment[news])
                click_title_index = self.news_title_index[self.histories[line]]
                click_title_segment = self.news_title_segment[self.histories[line]]
                impr_index.append(self.impr_indexes[line])
                user_index.append(self.uindexes[line])
                candidate_title_entity_index = self.title_entity_index[news]  # LP add
                click_title_entity_index = self.title_entity_index[self.histories[line]]  # LP add

                yield (
                    label,
                    impr_index,
                    user_index,
                    candidate_title_index,
                    candidate_title_segment,
                    click_title_index,
                    click_title_segment,
                    candidate_title_entity_index,
                    click_title_entity_index
                )

    def load_data_from_file(self, news_file, behavior_file):
        """Read and parse data from news file and behavior file.

        Args:
            news_file (str): A file contains several informations of news.
            beahaviros_file (str): A file contains information of user impressions.

        Returns:
            obj: An iterator that will yields parsed results, in the format of dict.
        """

        if not hasattr(self, "news_title_index"):
            self.init_news(news_file)

        if not hasattr(self, "impr_indexes"):
            self.init_behaviors(behavior_file)

        label_list = []
        imp_indexes = []
        user_indexes = []
        candidate_title_indexes = []
        candidate_title_segments = []
        click_title_indexes = []
        click_title_segments = []
        candidate_title_entity_indexes = []
        click_title_entity_indexes = []
        cnt = 0

        indexes = np.arange(len(self.labels))

        if self.npratio > 0:
            np.random.shuffle(indexes)

        for index in indexes:
            for (
                    label,
                    imp_index,
                    user_index,
                    candidate_title_index,
                    candidate_title_segment,  # LP add
                    click_title_index,
                    click_title_segment,  # LP add
                    candidate_title_entity_index,  # LP add
                    click_title_entity_index  # LP add
            ) in self.parser_one_line(index):
                candidate_title_indexes.append(candidate_title_index)
                candidate_title_segments.append(candidate_title_segment)  # LP add
                click_title_indexes.append(click_title_index)
                click_title_segments.append(click_title_segment)  # LP add
                imp_indexes.append(imp_index)
                user_indexes.append(user_index)
                label_list.append(label)
                candidate_title_entity_indexes.append(candidate_title_entity_index)  # LP add
                click_title_entity_indexes.append(click_title_entity_index)  # LP add

                cnt += 1
                if cnt >= self.batch_size:
                    yield self._convert_data(
                        label_list,
                        imp_indexes,
                        user_indexes,
                        candidate_title_indexes,
                        candidate_title_segments,
                        click_title_indexes,
                        click_title_segments,
                        candidate_title_entity_indexes,  # LP add
                        click_title_entity_indexes,  # LP add
                    )
                    label_list = []
                    imp_indexes = []
                    user_indexes = []
                    candidate_title_indexes = []
                    click_title_indexes = []
                    candidate_title_segments = []
                    click_title_segments = []
                    candidate_title_entity_indexes = []
                    click_title_entity_indexes = []
                    cnt = 0

        if cnt > 0:
            yield self._convert_data(
                label_list,
                imp_indexes,
                user_indexes,
                candidate_title_indexes,
                candidate_title_segments,
                click_title_indexes,
                click_title_segments,
                candidate_title_entity_indexes,  # LP add
                click_title_entity_indexes,  # LP add
            )

    def _convert_data(
            self,
            label_list,
            imp_indexes,
            user_indexes,
            candidate_title_indexes,
            candidate_title_segments,  # LP add
            click_title_indexes,
            click_title_segments,  # LP add
            candidate_title_entity_indexes,
            click_title_entity_indexes,
    ):
        """Convert data into numpy arrays that are good for further model operation.

        Args:
            label_list (list): a list of ground-truth labels.
            imp_indexes (list): a list of impression indexes.
            user_indexes (list): a list of user indexes.
            candidate_title_indexes (list): the candidate news titles' words indices.
            click_title_indexes (list): words indices for user's clicked news titles.

        Returns:
            dict: A dictionary, contains multiple numpy arrays that are convenient for further operation.
        """

        labels = np.asarray(label_list, dtype=np.float32)
        imp_indexes = np.asarray(imp_indexes, dtype=np.int32)
        user_indexes = np.asarray(user_indexes, dtype=np.int32)
        candidate_title_index_batch = np.asarray(
            candidate_title_indexes, dtype=np.int64
        )
        candidate_title_segment_batch = np.asarray(
            candidate_title_segments, dtype=np.int64  # LP add
        )

        click_title_index_batch = np.asarray(click_title_indexes, dtype=np.int64)
        click_title_segment_batch = np.asarray(click_title_segments, dtype=np.int64)  # LP add

        candidate_title_entity_index_batch = np.asarray(
            candidate_title_entity_indexes, dtype=np.int64
        )
        click_title_entity_index_batch = np.asarray(click_title_entity_indexes, dtype=np.int64)
        return {
            "impression_index_batch": imp_indexes,
            "user_index_batch": user_indexes,
            "clicked_title_batch": click_title_index_batch,
            "clicked_title_segment_batch": click_title_segment_batch,  # LP
            "candidate_title_batch": candidate_title_index_batch,
            "candidate_title_segment_batch": candidate_title_segment_batch,  # LP
            "labels": labels,
            "candidate_title_entity_batch": candidate_title_entity_index_batch,
            "clicked_title_entity_batch": click_title_entity_index_batch,
        }

    def load_user_from_file(self, news_file, behavior_file, test):
        """Read and parse user data from news file and behavior file.

        Args:
            news_file (str): A file contains several informations of news.
            beahaviros_file (str): A file contains information of user impressions.

        Returns:
            obj: An iterator that will yields parsed user feature, in the format of dict.
        """

        if not hasattr(self, "news_title_index"):
            self.init_news(news_file)

        if not hasattr(self, "impr_indexes"):
            self.init_behaviors(behavior_file, test)

        user_indexes = []
        impr_indexes = []
        click_title_indexes = []
        click_title_segments = []  # LP
        click_title_entity_indexes = []
        cnt = 0
        # TODO 这里也要添加segment哦
        for index in range(len(self.impr_indexes)):
            click_title_indexes.append(self.news_title_index[self.histories[index]])
            click_title_segments.append(self.news_title_segment[self.histories[index]])  # LP add

            click_title_entity_indexes.append(self.title_entity_index[self.histories[index]])
            user_indexes.append(self.uindexes[index])
            impr_indexes.append(self.impr_indexes[index])

            cnt += 1
            if cnt >= self.batch_size:
                yield self._convert_user_data(
                    user_indexes, impr_indexes, click_title_indexes, click_title_segments, click_title_entity_indexes
                )
                user_indexes = []
                impr_indexes = []
                click_title_indexes = []
                click_title_entity_indexes = []
                click_title_segments = []  # LP
                cnt = 0

        if cnt > 0:
            yield self._convert_user_data(
                user_indexes, impr_indexes, click_title_indexes, click_title_segments, click_title_entity_indexes
            )

    def _convert_user_data(
            self, user_indexes, impr_indexes, click_title_indexes, click_title_segments, click_title_entity_indexes
    ):
        """Convert data into numpy arrays that are good for further model operation.

        Args:
            user_indexes (list): a list of user indexes.
            click_title_indexes (list): words indices for user's clicked news titles.

        Returns:
            dict: A dictionary, contains multiple numpy arrays that are convenient for further operation.
        """

        user_indexes = np.asarray(user_indexes, dtype=np.int32)
        impr_indexes = np.asarray(impr_indexes, dtype=np.int32)
        click_title_index_batch = np.asarray(click_title_indexes, dtype=np.int64)
        click_title_segment_batch = np.asarray(click_title_segments, dtype=np.int64)
        click_title_entity_indexes = np.asarray(click_title_entity_indexes, dtype=np.int64)
        return {
            "user_index_batch": user_indexes,
            "impr_index_batch": impr_indexes,
            "clicked_title_batch": click_title_index_batch,
            "clicked_title_segment_batch": click_title_segment_batch,
            "clicked_title_entity_batch": click_title_entity_indexes,
        }

    def load_news_from_file(self, news_file):
        """Read and parse user data from news file.

        Args:
            news_file (str): A file contains several informations of news.

        Returns:
            obj: An iterator that will yields parsed news feature, in the format of dict.
        """
        if not hasattr(self, "news_title_index"):
            self.init_news(news_file)

        news_indexes = []
        candidate_title_indexes = []
        candidate_title_segments = []
        candidate_title_entity_indexes = []
        cnt = 0
        # LP 加入entity
        for index in range(len(self.news_title_index)):
            news_indexes.append(index)
            candidate_title_indexes.append(self.news_title_index[index])
            candidate_title_segments.append(self.news_title_segment[index])
            candidate_title_entity_indexes.append(self.title_entity_index[index])

            cnt += 1
            if cnt >= self.batch_size:
                yield self._convert_news_data(
                    news_indexes, candidate_title_indexes, candidate_title_segments, candidate_title_entity_indexes
                )
                news_indexes = []
                candidate_title_indexes = []
                candidate_title_segments = []
                candidate_title_entity_indexes = []
                cnt = 0

        if cnt > 0:
            yield self._convert_news_data(
                news_indexes, candidate_title_indexes, candidate_title_segments, candidate_title_entity_indexes
            )

    def _convert_news_data(
            self, news_indexes, candidate_title_indexes, candidate_title_segments, candidate_title_entity_indexes
    ):
        """Convert data into numpy arrays that are good for further model operation.

        Args:
            news_indexes (list): a list of news indexes.
            candidate_title_indexes (list): the candidate news titles' words indices.

        Returns:
            dict: A dictionary, contains multiple numpy arrays that are convenient for further operation.
        """

        news_indexes_batch = np.asarray(news_indexes, dtype=np.int32)
        candidate_title_index_batch = np.asarray(
            candidate_title_indexes, dtype=np.int32
        )
        candidate_title_segment_batch = np.asarray(
            candidate_title_segments, dtype=np.int32
        )
        candidate_title_entity_index_batch = np.asarray(
            candidate_title_entity_indexes, dtype=np.int32

        )

        return {
            "news_index_batch": news_indexes_batch,
            "candidate_title_batch": candidate_title_index_batch,
            "candidate_title_segment_batch": candidate_title_segment_batch,
            "candidate_title_entity_batch": candidate_title_entity_index_batch
        }

    def load_impression_from_file(self, behaivors_file, test=0):
        """Read and parse impression data from behaivors file.

        Args:
            behaivors_file (str): A file contains several informations of behaviros.

        Returns:
            obj: An iterator that will yields parsed impression data, in the format of dict.
        """
        if not test:
            if not hasattr(self, "histories"):
                self.init_behaviors(behaivors_file)
        else:
            self.histories = []
            self.imprs = []
            self.labels = []
            self.impr_indexes = []
            self.uindexes = []

            with tf.io.gfile.GFile(behaivors_file, "r") as rd:
                impr_index = 0
                for line in rd:
                    uid, time, history, impr = line.strip("\n").split(self.col_spliter)[-4:]

                    history = [self.nid2index[i] for i in history.split()]
                    history = [0] * (self.his_size - len(history)) + history[
                                                                     : self.his_size
                                                                     ]

                    impr_news = [self.nid2index[i.split("-")[0]] for i in impr.split()]
                    uindex = self.uid2index[uid] if uid in self.uid2index else 0
                    self.histories.append(history)
                    self.imprs.append(impr_news)
                    self.labels.append([1 for i in impr.split()])
                    self.impr_indexes.append(impr_index)
                    self.uindexes.append(uindex)
                    impr_index += 1

        indexes = np.arange(len(self.labels))

        for index in indexes:
            impr_label = np.array(self.labels[index], dtype="int32")
            impr_news = np.array(self.imprs[index], dtype="int32")

            yield (
                self.impr_indexes[index],
                impr_news,
                self.uindexes[index],
                impr_label,
            )
