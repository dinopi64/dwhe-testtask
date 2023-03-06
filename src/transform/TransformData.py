# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd


# main part: data training, processing
# A lot of omissions (lack of time), but everything works
# Logs replaced print and read data placed in test
class NormalizationScoreZ:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, x):
        self.mean = np.mean(x, axis=0)
        self.std = np.std(x, axis=0)

        # no log, only print =D
        print(
            "LOG. result fit() function mean data: " + str(self.mean)
        )
        print(
            "LOG. result fit() function std data: " + str(self.std)
        )

    def fit_data(self, train_data):
        print(
            "LOG. run fit_data() function."
        )
        arr_train = np.array(train_data)
        self.fit(arr_train)

    def transform(self, y):
        stand = (y - self.mean) / self.std
        print(
            "LOG. result transform() function stand data: " + str(stand)
        )
        return stand

    def fit_transform(self, test_data, output_columns, indx):
        arr_test = np.array(test_data)

        transform = pd.DataFrame(columns=output_columns)
        transform.insert(0, indx.name, indx)

        arr_transform = self.transform(arr_test)
        transform.iloc[:, 1:] = arr_transform

        print(
            "LOG. result fit_transform() function stand data: " + str(transform)
        )

        return transform

    # numpy faster and less dump if numbers
    def data_idmax_feature(self, data_feature):
        # indx_max = data_feature.idxmax(axis=1).str[-1].astype(int)
        indx_max = np.argmax(data_feature.values, axis=1)
        return indx_max


class TransformData:
    def __init__(
            self,
            train_data,
            test_data,
            dict_feature_type,
            input_columns,
            output_columns,
    ):
        self.train_data = train_data
        self.test_data = test_data
        self.dict_feature_type = dict_feature_type
        self.input_columns = input_columns
        self.output_columns = output_columns

    def preprocess_data(self):
        print(
            "LOG. run preprocess_data() function."
        )

        # map, filter, type - everything is here,
        # but usually somewhere in the hanlde class in preprocessing
        # (which also includes normalization)
        data_raw = [self.train_data, self.test_data]
        data_df = []

        for rwd in data_raw:
            pp_df = rwd[self.input_columns]
            # veeeeery long number
            # pp_df.iloc[:, 0] = pp_df.iloc[:, 0].astype(np.uint64)
            data_df.append(pp_df)

        return data_df[0], data_df[1]

    def run_job(self):
        transform = NormalizationScoreZ()
        train_df, test_df = self.preprocess_data()
        full_data_feature = []

        print(
            "LOG. result run_job() function data train: " + str(train_df)
        )
        print(
            "LOG. result run_job() function data test: " + str(test_df)
        )

        # breakdown by type, but not the best. As I understand it,
        # type '2' can beat, for example, str,
        # then you need to work with each one separately (for example, map)
        begin_feature = 1
        for feature, n in self.dict_feature_type.items():

            train_data_feature = train_df.iloc[:, begin_feature: (begin_feature + n)]
            test_data_feature = test_df.iloc[:, begin_feature: (begin_feature + n)]

            transform.fit_data(train_data_feature)
            tranform_data_feature = transform.fit_transform(
                test_data=test_data_feature,
                output_columns=self.output_columns[begin_feature: (begin_feature + n)],
                indx=test_df.iloc[:, 0]
            )

            tranform_data_feature[f'max_feature_{feature}_index'] = transform.data_idmax_feature(test_data_feature)

            full_data_feature.append(tranform_data_feature)
            begin_feature += n

            print(
                "LOG. result run_job() function data transform: " + str(tranform_data_feature)
            )

        return pd.concat(full_data_feature)


if __name__ == '__main__':
    train_df = pd.DataFrame({
        'id_job': [1, 2, 3],
        'feature_type_1_0': [2.0, 3.0, 4.0],
        'feature_type_1_1': [1.0, 5.0, 3.0],
    })
    test_df = pd.DataFrame({
        'id_job': [4, 5, 6],
        'feature_type_1_0': [1.0, 2.0, 3.0],
        'feature_type_1_1': [4.0, 5.0, 6.0],
    })
    expected_data = pd.DataFrame({
        'id_job': [4, 5, 6],
        'feature_type_1_stand_0': [-2.449490, -1.224745, 0.000000],
        'feature_type_1_stand_1': [0.612372, 1.224745, 1.837117],
        'max_feature_type_1_index': [2, 2, 2],
    })

    param = {
        'train_data': train_df,
        'test_data': test_df,
        "dict_feature_type": {'type_1': 2, },
        "input_columns": train_df.columns,
        "output_columns": expected_data.columns,
    }
    transform_data = TransformData(
        **param
    )

    pd.set_option('display.max_rows', 10)
    pd.set_option('display.max_columns', 10)
    pd.set_option('display.width', 100)

    result = transform_data.run_job()
    # assert result[0].equals(expected_data.iloc[:, :-1]), "expected and result data are diff"
