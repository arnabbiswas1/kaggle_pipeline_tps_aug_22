import category_encoders as ce
from sklearn.preprocessing import LabelEncoder


__all__ = [
    "do_count_encoding",
    "do_label_encoding",
    "convert_to_int",
]


def _get_encoder(encoder_name):
    """
    Returns an Encdoer Object given the name of the encoder
    """
    if encoder_name == "LabelEncoder":
        return LabelEncoder()
    elif encoder_name == "CountEncoder":
        return ce.CountEncoder()
    else:
        return None


def _do_encoding(
    encoder_name,
    source_train_df,
    source_test_df,
    target_train_df,
    target_test_df,
    categorical_features,
    feature_name_suffix,
):
    """
        Given with a type of encoding, encode set of features
        listed in categorical_features variable
        """
    for feature_name in categorical_features:
        encoder = _get_encoder(encoder_name)
        encoder.fit(
            list(source_train_df[feature_name].values)
            + list(source_test_df[feature_name].values)
        )
        if feature_name_suffix:
            target_feature_name = f"{feature_name}_{feature_name_suffix}"
            print(
                f"{encoder_name} of feature [{feature_name}] is saved at [{target_feature_name}]"
            )
        else:
            target_feature_name = feature_name
            print(f"{encoder_name} the feature [{target_feature_name}]")
        target_train_df[target_feature_name] = encoder.transform(
            list(source_train_df[feature_name].values)
        )
        target_test_df[target_feature_name] = encoder.transform(
            list(source_test_df[feature_name].values)
        )
    return target_train_df, target_test_df


def do_label_encoding(
    source_train_df,
    source_test_df,
    target_train_df,
    target_test_df,
    categorical_features,
    feature_name_suffix=None,
):
    """
    Label encode the categorical features.
    After encdoing, it appends a new set of features with name
    <original_feature_name>_label to the target dataframe
    """
    return _do_encoding(
        "LabelEncoder",
        source_train_df,
        source_test_df,
        target_train_df,
        target_test_df,
        categorical_features,
        feature_name_suffix,
    )


def do_count_encoding(
    source_train_df,
    source_test_df,
    target_train_df,
    target_test_df,
    categorical_features,
    feature_name_suffix=None,
):
    """
    Count encode the categorical features.
    After encdoing, it appends a new set of features with name
    <original_feature_name>_label to the target dataframe
    """
    return _do_encoding(
        "CountEncoder",
        source_train_df,
        source_test_df,
        target_train_df,
        target_test_df,
        categorical_features,
        feature_name_suffix,
    )


def convert_to_int(df, feature_names):
    for feature_name in feature_names:
        df.loc[:, feature_name] = df[feature_name].astype("int")
    return df
