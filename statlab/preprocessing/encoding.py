import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder, LabelEncoder


def one_hot_encoder(y, sparse: bool = False):

    val = np.array(y)
    encoder = OneHotEncoder(sparse=sparse)

    if len(val.shape) == 1:
        val = val.reshape(-1, 1)

    encoded = encoder.fit(val)

    return encoded.transform(val), encoded.categories_[0]


def label_encoder(y):

    val = np.array(y).ravel()
    encoder = LabelEncoder()
    encoded = encoder.fit(val)

    return encoded.transform(val), val


def encoder_data_frame(
        df: pd.DataFrame,
        encod_columns: list,
        encoding_type: str = "OneHot",
        **kwargs
) -> pd.DataFrame:

    if encoding_type not in ("OneHot", "LabelEncoding"):
        raise TypeError("Incorret type of encoding")

    for i in encod_columns:
        if i not in df.columns:
            raise ValueError("Data frame has not encod_columns")

    encoder = {
        "OneHot": one_hot_encoder,
        "LabelEncoding": label_encoder
    }

    index = df.index
    data_frame = df.to_dict()

    for item in encod_columns:
        val = list(data_frame.pop(item).values())
        encoded = encoder[encoding_type](val, **kwargs)

        if encoding_type == "LabelEncoding":
            data_frame[item] = dict(zip(index, encoded[0]))
        else:
            for i in range(len(encoded[1])):
                data_frame[encoded[1][i]] = dict(zip(index, encoded[0][:, i]))

    data_frame = pd.DataFrame.from_dict(data_frame)
    data_frame.index = index

    return data_frame


if __name__ == '__main__':
    x = dict(sex=['male', 'female', 'male', 'female'], id=[1, 2, 3, 4])
    x = pd.DataFrame.from_dict(x)
    one_hot = encoder_data_frame(x, encod_columns=['sex'], encoding_type='OneHot')
    label = encoder_data_frame(x, encod_columns=['sex'], encoding_type='LabelEncoding')
