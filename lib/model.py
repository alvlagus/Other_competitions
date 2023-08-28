import pandas as pd
import numpy as np
import re
import pymorphy2
import zipfile
import pandarallel

from json import load as json_load
from pandas import DataFrame, concat
from functools import lru_cache
from pandarallel import pandarallel
from catboost import CatBoostClassifier
from dataclasses import dataclass
from typing import Dict, Union, Tuple
from nltk.corpus import stopwords

regexes_path = '/app/lib/archive/regexp_dict.json'
catboost_path = '/app/lib/archive/final_model.cbm'
stopwords_path = '/app/lib/archive/stopwords.json'
archive_path = '/app/lib/archive/catboost_model.zip'
unpacking_path = '/app/lib/archive/'

# распаковываем файл модели из архива
with zipfile.ZipFile(archive_path, 'r') as zip_file:
    zip_file.extractall(unpacking_path)


# функция для открытия json файлов
def get_json(path):
    with open(path) as json_file:
        json = json_load(json_file)
        json_file.close()
    return json


def get_text_clearing(text):
    reg = re.compile('[^0-9a-zA-Zа-яА-ЯёЁ\.,\(\)]+]')
    text = re.sub(r'([^\w ])', r' \1', text)
    text = re.sub(r'([^ \w])', r'\1', text)
    text = re.sub(r' +', r' ', text)
    text = re.sub(r'^ ', r'', text)
    text = re.sub(r'[\W_]+', ' ', text)
    text = reg.sub(' ', text)
    text = ' '.join(text.split())
    text = text.lower()
    return text


# Лемматизация
@lru_cache(maxsize=100000)  # для скорости вычислений
def get_lemmatization(word, morph):
    return morph.parse(word)[0].normal_form


# функция предварительной обработки, включая очистку, удаление стоп-слов и лемматизацию
def get_text_preprocessing(text, my_stopwords, morph):
    text = get_text_clearing(str(text)).split()
    text = [word for word in text if word not in my_stopwords]
    return ' '.join(map(lambda word: get_lemmatization(word, morph), text))


# функция поиска по словарю
def get_dict_search(sample, text):
    events = []
    for name, regexp in sample.items():
        events.append(int(bool(re.search(regexp, text))))
    return events

# функция для преобразования результатов поиска выше в датафрейм
def get_df_from_list(event, regexp):
    return DataFrame((regex for regex in event), columns=regexp.keys())

# зададим классификатор CatBoost
@dataclass
class CatBoost:
    dataset: DataFrame
    regexps_path: str
    catb_path: str
    stopwords_path: str
    predictions: DataFrame = None
    my_stopwords: set = None
    regexes: Dict[str, str] = None
    model: CatBoostClassifier = None

    def get_json(self):
        self.regexps = get_json(self.regexps_path)

    def get_stopwords(self):
        additional_stopwords = get_json(self.stopwords_path)
        self.my_stopwords = set(additional_stopwords)
        self.my_stopwords.update(stopwords.words('russian'))
        self.my_stopwords.update(stopwords.words('english'))

    def get_dataset_preparing(self):
        morph = pymorphy2.MorphAnalyzer()

        # вычисления длины заголовка новости без учета пробелов
        self.dataset['description_len'] = self.dataset['description'].apply(
            lambda x: len(x) - x.count(" "))
        print('Done len')

        # замена NaN в признаке price на среднее значение. Выбросы оставим без внимания
        self.dataset['price'] = self.dataset['price'].fillna(self.dataset['price'].mean())
        print('Done price')

        # логорифмирование price
        self.dataset['price_log'] = np.log(self.dataset['price'] +1)
        print('Done price_log')

        # обработаем дату размещения, выделив месяцы и дни
        self.dataset['month'] = pd.to_datetime(self.dataset['datetime_submitted']).dt.strftime("%m").astype(int)
        self.dataset['day'] = pd.to_datetime(self.dataset['datetime_submitted']).dt.strftime("%d").astype(int)
        self.dataset = self.dataset.drop(['datetime_submitted'], axis=1)
        print('Done data')

        # производим в описании поиск регулярных выражений. Создаем из результатов поиска датасет
        regexp_search = self.dataset.description.apply(
            lambda text: get_dict_search(self.regexps, text))
        regexp_search = get_df_from_list(regexp_search, self.regexps)

        # объединяем признаки title и description и удаляем исходные
        self.dataset['title_and_description'] = self.dataset.title + ' ' + self.dataset.description
        # self.dataset.drop(['title', 'description'], axis=1, inplace=True)
        print('Done title_and_description')

        # Создадим дополнительные признаки, вытащим из 'title_and_description' отдельно текст и цифры
        self.dataset['text'] = self.dataset['title_and_description'].apply(
            lambda text: re.sub('[^A-Za-z0-9\.\@\ \-\_]', ' ', text))
        self.dataset['text'] = self.dataset['text'].apply(lambda text: re.sub(' +', ' ', text))
        print('Done text')

        self.dataset['numbers'] = self.dataset['title_and_description'].apply(
            lambda text: re.sub('[^0-9\+\(\)\-]', ' ', text))
        self.dataset['numbers'] = self.dataset['numbers'].apply(lambda text: re.sub(' +', ' ', text))
        print('Done numbers')

        # добавляем регулярный датасет в исходный
        self.dataset = concat([self.dataset, regexp_search], axis=1)
        print('Done concat')

        self.dataset.title_and_description = self.dataset.title_and_description.apply(
            lambda text: get_text_preprocessing(text, self.my_stopwords, morph))
        print('Done get_text_preprocessing')

    def get_model(self):
        model_file = self.catb_path
        self.model = CatBoostClassifier().load_model(model_file)

    def predict(self):
        self.predictions = self.model.predict_proba(self.dataset)[:, 1]
        index = range(len(self.predictions))
        self.predictions = DataFrame(
            zip(index, self.predictions),
            columns=['index', 'prediction']
        )

    def run_model(self):
        pandarallel.initialize(progress_bar=False)
        self.get_json()
        self.get_stopwords()
        self.get_dataset_preparing()
        self.get_model()
        self.dataset = self.dataset[self.model.feature_names_]
        self.predict()
        return self.predictions


def task1(test):
    catboost = CatBoost(
        dataset=test,
        regexps_path=regexes_path,
        catb_path=catboost_path,
        stopwords_path=stopwords_path)

    return catboost.run_model()


def task2(description: str) -> Union[Tuple[int, int], Tuple[None, None]]:
    description_size = len(description)
    if description_size % 2 == 0:
        return None, None
    else:
        return 0, description_size
