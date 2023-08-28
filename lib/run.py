import argparse
import logging
import os
import sys
from typing import Tuple

import pandas as pd
from model import task1, task2


class Test:
    train_csv = 'train.csv'
    val_csv = 'val.csv'
    test_csv = 'test_data.csv'
    task1_prediction = 'task1_prediction.csv'
    task2_prediction = 'task2_prediction.csv'

    def __init__(self, debug: bool = True):
        self.debug = debug
        self.logger = self._get_logger()
        self.data_dir = os.getenv('DATA_ROOT')
        self.test_data_dir = os.getenv('TEST_DATA_ROOT')
        self.user = os.getenv('USER')

    def _get_logger(self):
        logger = logging.getLogger()
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

        return logger

    def _check(
        self, test: pd.DataFrame, task1_prediction: pd.DataFrame, task2_prediction: pd.DataFrame,
    ):
        if test.shape[0] != task1_prediction.shape[0]:
            error = (
                f'Размер тестовой выборки ({test.shape[0]}) не совпадает с ответами задачи 1 '
                f'({task1_prediction.shape[0]})'
            )
            raise ValueError(error)
        if set(task1_prediction.columns) != {'index', 'prediction'}:
            error = (
                f'В ответах к задаче 1 должны быть два стобца - index и prediction '
                f'({task1_prediction.columns})'
            )
            raise ValueError(error)
        if set(range(test.shape[0])) != set(task1_prediction['index']):
            error = (
                'В ответах к задаче 1 должны быть `index` должен полностью пересекаться с тестом'
            )
            raise ValueError(error)
        if not all(task1_prediction['prediction'].apply(lambda x: 0 <= x <= 1)):
            error = 'В ответах к задаче 1 `prediction` должен лежать в диапазоне [0, 1]'
            raise ValueError(error)

        if test.shape[0] != task2_prediction.shape[0]:
            error = (
                f'Размер тестовой выборки ({test.shape[0]}) не совпадает с ответами задачи 2 '
                f'({task2_prediction.shape[0]})'
            )
            raise ValueError(error)
        if set(task2_prediction.columns) != {'index', 'start', 'finish'}:
            error = (
                f'В ответах к задаче 2 должны быть три стобца - index, start и finish'
                f'({task1_prediction.columns})'
            )
            raise ValueError(error)
        if not all(
            task2_prediction['start'].apply(
                lambda x: pd.isna(x)
                or (isinstance(x, (int, float)) and x >= 0 and int(x) == float(x)),
            ),
        ):
            error = 'В ответах к задаче 2 `start` должен быть None или индексом символа в `description`'
            raise ValueError(error)
        if not all(
            task2_prediction['finish'].apply(
                lambda x: pd.isna(x)
                or (isinstance(x, (int, float)) and x >= 0 and int(x) == float(x)),
            ),
        ):
            error = 'В ответах к задаче 2 `finish` должен быть None или индексом символа в `description`'
            raise ValueError(error)

    def train(self) -> pd.DataFrame:
        path = os.path.join(self.data_dir, self.train_csv)
        if not os.path.exists(path):
            self.logger.info(f'Файл {path} не найден')
            path = os.path.join(self.test_data_dir, self.train_csv)
        if not os.path.exists(path):
            self.logger.info(f'Файл {path} не найден')
            raise ValueError('train не найден')
        df = pd.read_csv(path)

        return df

    def val(self) -> pd.DataFrame:
        path = os.path.join(self.data_dir, self.val_csv)
        if not os.path.exists(path):
            self.logger.info(f'Файл {path} не найден')
            path = os.path.join(self.test_data_dir, self.val_csv)
        if not os.path.exists(path):
            self.logger.info(f'Файл {path} не найден')
            raise ValueError('val не найден')
        df = pd.read_csv(path)

        return df

    def test(self) -> pd.DataFrame:
        path = os.path.join(self.test_data_dir, self.val_csv if self.debug else self.test_csv)
        if not os.path.exists(path):
            self.logger.info(f'Файл {path} не найден')
            raise ValueError('test не найден')
        df = pd.read_csv(path)
        if 'is_bad' in df:
            del df['is_bad']

        return df

    def run(self):
        self.logger.info(f'Запуск модели {self.user}')
        task1_prediction, task2_prediction = self.process()
        task1_prediction.index = task1_prediction['index']
        task2_prediction.index = task2_prediction['index']
        self.logger.info('Проверка ответов')
        self._check(self.test(), task1_prediction, task2_prediction)

        task1_path = os.path.join(self.test_data_dir, f'{self.user}_{self.task1_prediction}')
        task2_path = os.path.join(self.test_data_dir, f'{self.user}_{self.task2_prediction}')
        self.logger.info(f'Сохранение результатов {task1_path}')
        task1_prediction.to_csv(task1_path, index=False)
        self.logger.info(f'Сохранение результатов {task2_path}')
        task2_prediction.to_csv(task2_path, index=False)

        self.logger.info('Готово')

    def process(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        test = self.test()

        # task1_prediction = pd.DataFrame(columns=['index', 'prediction'])
        # task1_prediction['index'] = test.index
        # task1_prediction['prediction'] = test['title'].apply(task1)

        task1_prediction = task1(test)

        task2_prediction = pd.DataFrame(columns=['index', 'start', 'finish'])
        task2_prediction['index'] = test.index
        task2_prediction[['start', 'finish']] = test['description'].apply(
            lambda x: pd.Series(task2(x)),
        )

        return task1_prediction, task2_prediction


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true')

    test = Test(debug=parser.parse_args().debug)
    test.run()
