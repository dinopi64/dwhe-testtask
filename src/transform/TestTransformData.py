# -*- coding: utf-8 -*-

import pandas as pd
from transform.TransformData import TransformData


# This is a bit of a wild part (unpack_nested_dict, input/output cols),
# do not pay attention (I tried to do a cols check, manually it's worse)
class Utils:
    @staticmethod
    def read_csv(file_path, batch=None):
        if batch is not None:
            return pd.read_csv(file_path, chunksize=batch)
        elif batch is None:
            return pd.read_csv(file_path)

    @staticmethod
    def write_csv(df, file_path):
        df.to_csv(file_path, index=False)

    #
    # {a: {b: 2}, c: 0} -> [a_b_0, a_b_1, a_b_2, c] Dict[Dict...[int]] -> List[str]
    @staticmethod
    def unpack_nested_dict(nested_dict, prefix=''):
        cols_input = []
        for key, value in nested_dict.items():
            if isinstance(value, dict):
                cols_input += Utils.unpack_nested_dict(value, prefix + key + '_')
            elif isinstance(value, int):
                if value > 0:
                    for i in range(0, value):
                        cols_input.append(prefix + key + '_' + str(i))
                else:
                    cols_input.append(prefix + key)
        return cols_input


class TestTransformData(TransformData):
    def __init__(
            self,
            path_train_data,
            path_test_data,
            path_transformed_data,
            batch_train,
            dict_feature_type,
            input_columns,
            output_columns,
    ):
        super().__init__(
            train_data=None,
            test_data=None,
            dict_feature_type=dict_feature_type,
            input_columns=input_columns,
            output_columns=output_columns,
        )
        self.batch_train = batch_train
        self.path_train_data = path_train_data
        self.path_test_data = path_test_data
        self.path_transformed_data = path_transformed_data
        self.transform_test_result = None

    # very bad, no time
    def get_raw_data(self):
        raw_data = {'train_raw_data': [], 'test_raw_data': []}

        for k in raw_data.keys():
            batch_size = None
            path_data = None
            try:
                if k == 'train_raw_data':
                    batch_size = self.batch_train
                    path_data = self.path_train_data

                # 80/20 => x4
                # not sure, need to check.
                # Part of the data is lost (bad),
                # you need to take and use train / transform separately
                elif k == 'test_raw_data':
                    batch_size = self.batch_train / 4
                    path_data = self.path_test_data

                for chunk in Utils.read_csv(path_data, batch=batch_size):
                    raw_data[k] += [chunk]

                    print(
                        "LOG. run get_raw_data() function, chunk: " + str(chunk)
                    )

            except FileNotFoundError as fnfe:
                print(
                    'ERROR func get_raw_data(). Change path file: ' + str(path_data) + ". " + str(fnfe)
                )

            except Exception as exc:
                print(
                    'ERROR func get_raw_data():' + str(exc)
                )

        print(
            "LOG. run get_raw_data() function, status data test/train: " +
            "len train_data: " + str(len(raw_data['train_raw_data'])) +
            "len test_data: " + str(len(raw_data['test_raw_data']))
        )

        return raw_data['train_raw_data'], raw_data['test_raw_data']

    # lego
    def run_test(self):
        transform_data_list = []
        lst_train_data, lst_test_data = self.get_raw_data()
        print(lst_train_data)

        # very bad
        end_point = len(lst_train_data)
        if len(lst_test_data) < len(lst_train_data):
            end_point = len(lst_test_data)

        for i in range(0, end_point):
            self.train_data = lst_train_data[i]
            self.test_data = lst_test_data[i]
            df_batch_transform = self.run_job()
            transform_data_list.append(df_batch_transform)
            print(
                "LOG. result run_test() function, status data transform: " + str(df_batch_transform)
            )

        if end_point < len(lst_test_data):
            while end_point < len(lst_test_data):
                self.train_data = lst_train_data[-1]
                self.test_data = lst_test_data[end_point]
                df_batch_transform = self.run_job()
                transform_data_list.append(df_batch_transform)
                print(
                    "LOG. result run_test() function, status data transform: " + str(df_batch_transform)
                )
                end_point += 1

        self.transform_test_result = pd.concat(transform_data_list)



    def get_transform_data_csv(self):
        Utils.write_csv(self.transform_test_result, self.path_transformed_data)


if __name__ == '__main__':
    # example test type 2, 3, ...:
    # schema_cols_input = {'id_job': 0,
    #                     'feature': {'type_1': 10, 'type_2': 2, 'type_3': 0}}

    # convenient form for writing: each row is a gluing of a column through the prefix '_'
    schema_cols_input = {'id_job': 0,
                         'feature': {'type_1':10, }}
    schema_cols_output = {'id_job': 0,
                          'feature': {'type_1': {'stand': 10}, },
                          'max_feature': {'type_1': {'index': 0}, }}

    param_transform = {
        'path_train_data': '../../data/input/train.csv',
        'path_test_data': '../../data/input/test.csv',
        'path_transformed_data': '../../data/output/test_transformed.csv',
        'batch_train': 400,
        'input_columns': Utils.unpack_nested_dict(schema_cols_input),
        'output_columns': Utils.unpack_nested_dict(schema_cols_output),
        'dict_feature_type': schema_cols_input['feature'],
    }
    transform = TestTransformData(
        **param_transform
    )

    transform.run_test()
    transform.get_transform_data_csv()
