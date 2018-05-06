import numpy
import pandas
import random
import scipy.stats
import xgboost

# Cross-validation

def load_cv_train(n_split, n_downsample, features):
    """Returns the training dataset for one of the cross-validation splits."""
    assert n_split in (1, 2, 3)
    xs, ys = [], []
    for n_day in range(1, 4):
        if n_day != n_split:
            for r in range(4):
                in_file = 'cache/feat_ds' + str(n_downsample)
                in_file += '_day' + str(n_day) + '-' + str(r) + '.h5'
                print('Loading', in_file)
                dataset = pandas.read_hdf(in_file)
                xs.append(dataset.as_matrix(columns=features))
                ys.append(dataset.as_matrix(columns=['is_attributed']))
    return numpy.concatenate(xs), numpy.concatenate(ys)

def load_cv_test(n_split, features):
    """Returns the test dataset for one of the cross-validation splits."""
    assert n_split in (1, 2, 3)
    in_file = 'cache/feat_day' + str(n_split) + '_test.h5'
    print('Loading', in_file)
    dataset = pandas.read_hdf(in_file)
    x = dataset.as_matrix(columns=features)
    y = dataset.as_matrix(columns=['is_attributed'])
    return x, y

def _score_cv_split(n_split, features, params):
    """Returns the AUROC score for each boosting round on one of the splits."""
    x_train, y_train = load_cv_train(n_split, params['n_downsample'], features)
    dtrain = xgboost.DMatrix(x_train, label=y_train)
    x_test, y_test = load_cv_test(n_split, features)
    dtest = xgboost.DMatrix(x_test, label=y_test)
    tree_params = dict(params['tree_params'])
    tree_params['eval_metric'] = 'auc'
    tree_params['silent'] = 1
    results = dict()
    if params['early_stopping'] == True:
        model = xgboost.train(
            tree_params, dtrain,
            evals=[(dtest, 'test')],
            evals_result=results,
            num_boost_round=1000,
            early_stopping_rounds=params['early_stopping_rounds'],
            verbose_eval=False)
    else:
        model = xgboost.train(
            tree_params, dtrain,
            evals=[(dtest, 'test')],
            evals_result=results,
            num_boost_round=params['num_boost_round'],
            verbose_eval=False)
    return results['test']['auc']

def top_score(scores):
    """Returns the maximum score and the associated number of boosting rounds."""
    max_score = max(scores)
    num_boost_rounds = scores.index(max_score) + 1
    return max_score, num_boost_rounds

def average_scores(all_scores):
    """Returns the average of a list of scores."""
    length = min(len(scores) for scores in all_scores)
    result = []
    for i in range(length):
        result.append(numpy.mean([scores[i] for scores in all_scores]))
    return result

def score_cv_split(n_split, features, params):
    """Returns the score and the number of boosting rounds on a split."""
    scores = _score_cv_split(n_split, features, params)
    result = top_score(scores)
    print(result)
    return result

def score_cv_all_splits(features, params):
    """Returns the average score and the number of boosting rounds on all the splits."""
    all_scores = []
    for n_split in range(1, 4):
        scores = _score_cv_split(n_split, features, params)
        print(top_score(scores))
        all_scores.append(scores)
    scores = average_scores(all_scores)
    result = top_score(scores)
    print(result)
    return result

# Submission

def load_full_train(features, n_downsample):
    """Returns the entire training dataset."""
    xs, ys = [], []
    for n_day in range(1, 4):
        for r in range(4):
            in_file = 'cache/feat_ds' + str(n_downsample)
            in_file += '_day' + str(n_day) + '-' + str(r) + '.h5'
            print('Loading', in_file)
            dataset = pandas.read_hdf(in_file)
            xs.append(dataset.as_matrix(columns=features))
            ys.append(dataset.as_matrix(columns=['is_attributed']))
    return numpy.concatenate(xs), numpy.concatenate(ys)

def load_full_test(features):
    """Returns the entire test dataset."""
    in_file = 'cache/feat_test.h5'
    print('Loading', in_file)
    dataset = pandas.read_hdf(in_file)
    x = dataset.as_matrix(columns=features)
    return x

def submit(params):
    """Creates and saves a submission."""
    x_train, y_train = load_full_train(params['n_downsample'])
    dtrain = xgboost.DMatrix(x_train, label=y_train)
    x_test = load_full_test()
    dtest = xgboost.DMatrix(x_test)
    tree_params = dict(params['tree_params'])
    tree_params['silent'] = 1
    model = xgboost.train(
        tree_params, dtrain,
        num_boost_round=params['num_boost_round'])
    y_predict = model.predict(dtest)
    y_predict = scipy.stats.rankdata(y_predict, method='ordinal') / len(y_predict)
    submission = pandas.read_hdf('cache/id_test.h5')
    submission['is_attributed'] = y_predict
    submission.to_csv('submission.csv', index=False)

# Features selection

def select_features():

    def score_features(features):
        """Returns the score of a list of features on the first split."""
        n_day = 1
        params = {
            'n_downsample': 1,
            'early_stopping': True,
            'early_stopping_rounds': 5,
            'tree_params': {
                'eta': 0.2,
                'max_depth': 11,
                'tree_method': 'exact',
            }
        }
        score = score_cv_split(n_day, features, params)[0]
        print(features)
        return score

    def read_all_features():
        """Iterates over all the features."""
        dataset = pandas.read_hdf('cache/feat_ds1_day1-0.h5')
        for feature in dataset.dtypes.keys():
            if feature != 'is_attributed':
                yield feature

    def mutate_features(features, all_features):
        """Returns a modified list of features with a random addition or removal."""
        feature = random.choice(all_features)
        features = set(features)
        if feature in features:
            features.remove(feature)
        else:
            features.add(feature)
        return list(sorted(features))

    all_features = list(read_all_features())
    while True:
        max_features = ['app', 'device', 'os', 'channel', 'day_seconds']
        max_score = score_features(max_features)
        failed_rounds = 0
        while failed_rounds < 40:
            features = mutate_features(max_features, all_features)
            score = score_features(features)
            if score > max_score:
                max_score = score
                max_features = features
                failed_rounds = 0
            else:
                failed_rounds += 1
        with open('feat_selection.txt' , 'a') as stream:
            stream.write(str((max_score, max_features)) + '\n')

#select_features()

def evaluate_features():

    features_candidates = [
        ['app', 'previous_click_per_60m_device_os', 'previous_click_per_60m_ip_device_os_app', 'previous_click_per_60m_device', 'unique_os_per_60m_ip', 'clicks_per_60m_device_os', 'unique_channel_per_60m_ip', 'unique_device_per_60m_ip', 'day_seconds', 'previous_click_per_60m_ip_device_os_channel', 'clicks_per_60m_os', 'clicks_per_60m_ip_device_app', 'previous_click_per_60m_os', 'os', 'clicks_per_60m_ip_app', 'previous_click_per_60m_device_channel', 'previous_click_per_60m_channel', 'channel', 'previous_click_per_60m_ip_os_app', 'clicks_per_60m_ip_os_app_channel', 'previous_click_per_60m_ip_device_channel', 'previous_click_per_60m_ip', 'device', 'clicks_per_60m_ip_os', 'clicks_per_60m_ip_device_os_channel', 'clicks_per_60m_ip_device_os_app', 'previous_click_per_60m_ip_app'],
        ['app', 'previous_click_per_60m_ip_device_os_app', 'clicks_per_60m_ip_device_os', 'unique_os_per_60m_ip', 'clicks_per_60m_device_os', 'previous_click_per_60m_ip_channel', 'unique_device_per_60m_ip', 'day_seconds', 'clicks_per_60m_ip_channel', 'previous_click_per_60m_os_channel', 'previous_click_per_60m_ip_os_channel', 'clicks_per_60m_ip_device_os_app_channel', 'clicks_per_60m_ip_device_app_channel', 'previous_click_per_60m_os', 'os', 'previous_click_per_60m_device_os_app', 'previous_click_per_60m_ip_os_app_channel', 'previous_click_per_60m_ip_device', 'previous_click_per_60m_device_channel', 'channel', 'clicks_per_60m_ip_device_os_app', 'device', 'clicks_per_60m_ip_device_os_channel', 'previous_click_per_60m_ip_device_app_channel', 'clicks_per_60m_ip_app_channel'],
        ['app', 'clicks_per_60m_ip_device_os', 'previous_click_per_60m_ip_device_os_app_channel', 'previous_click_per_60m_ip_device_os', 'previous_click_per_60m_app', 'clicks_per_60m_ip', 'unique_channel_per_60m_ip', 'unique_device_per_60m_ip', 'day_seconds', 'clicks_per_60m_ip_channel', 'previous_click_per_60m_ip_device_os_channel', 'clicks_per_60m_ip_device_os_app_channel', 'os', 'previous_click_per_60m_ip_device', 'channel', 'previous_click_per_60m_ip_os_app', 'clicks_per_60m_ip_device_os_app', 'previous_click_per_60m_app_channel', 'device', 'previous_click_per_60m_ip_device_channel', 'previous_click_per_60m_ip_os_app_channel', 'previous_click_per_60m_ip_app_channel'],
    ]

    def score_features(features):
        """Returns the score of a list of features on all the splits."""
        n_day = 1
        params = {
            'n_downsample': 1,
            'early_stopping': True,
            'early_stopping_rounds': 20,
            'tree_params': {
                'eta': 0.2,
                'max_depth': 11,
                'tree_method': 'exact',
            }
        }
        score = score_cv_all_splits(features, params)[0]
        print(features)
        return score

    for features in features_candidates:
        score = score_features(features)
        with open('feat_evaluation.txt' , 'a') as stream:
            stream.write(str((score, features)) + '\n')

#evaluate_features()

# Submission

def evaluate_submission():

    all_features = [
        ['app', 'previous_click_per_60m_device_os', 'previous_click_per_60m_ip_device_os_app', 'previous_click_per_60m_device', 'unique_os_per_60m_ip', 'clicks_per_60m_device_os', 'unique_channel_per_60m_ip', 'unique_device_per_60m_ip', 'day_seconds', 'previous_click_per_60m_ip_device_os_channel', 'clicks_per_60m_os', 'clicks_per_60m_ip_device_app', 'previous_click_per_60m_os', 'os', 'clicks_per_60m_ip_app', 'previous_click_per_60m_device_channel', 'previous_click_per_60m_channel', 'channel', 'previous_click_per_60m_ip_os_app', 'clicks_per_60m_ip_os_app_channel', 'previous_click_per_60m_ip_device_channel', 'previous_click_per_60m_ip', 'device', 'clicks_per_60m_ip_os', 'clicks_per_60m_ip_device_os_channel', 'clicks_per_60m_ip_device_os_app', 'previous_click_per_60m_ip_app'],
        ['app', 'previous_click_per_60m_ip_device_os_app', 'clicks_per_60m_ip_device_os', 'unique_os_per_60m_ip', 'clicks_per_60m_device_os', 'previous_click_per_60m_ip_channel', 'unique_device_per_60m_ip', 'day_seconds', 'clicks_per_60m_ip_channel', 'previous_click_per_60m_os_channel', 'previous_click_per_60m_ip_os_channel', 'clicks_per_60m_ip_device_os_app_channel', 'clicks_per_60m_ip_device_app_channel', 'previous_click_per_60m_os', 'os', 'previous_click_per_60m_device_os_app', 'previous_click_per_60m_ip_os_app_channel', 'previous_click_per_60m_ip_device', 'previous_click_per_60m_device_channel', 'channel', 'clicks_per_60m_ip_device_os_app', 'device', 'clicks_per_60m_ip_device_os_channel', 'previous_click_per_60m_ip_device_app_channel', 'clicks_per_60m_ip_app_channel'],
        ['app', 'clicks_per_60m_ip_device_os', 'previous_click_per_60m_ip_device_os_app_channel', 'previous_click_per_60m_ip_device_os', 'previous_click_per_60m_app', 'clicks_per_60m_ip', 'unique_channel_per_60m_ip', 'unique_device_per_60m_ip', 'day_seconds', 'clicks_per_60m_ip_channel', 'previous_click_per_60m_ip_device_os_channel', 'clicks_per_60m_ip_device_os_app_channel', 'os', 'previous_click_per_60m_ip_device', 'channel', 'previous_click_per_60m_ip_os_app', 'clicks_per_60m_ip_device_os_app', 'previous_click_per_60m_app_channel', 'device', 'previous_click_per_60m_ip_device_channel', 'previous_click_per_60m_ip_os_app_channel', 'previous_click_per_60m_ip_app_channel'],
    ]

    all_params = [
        {'tree_params': {'eta': 0.05, 'max_depth': 11, 'tree_method': 'exact'}, 'n_downsample': 1, 'early_stopping': True, 'early_stopping_rounds': 50},
        {'tree_params': {'scale_pos_weight': 19, 'colsample_bytree': 0.9, 'eta': 0.05, 'subsample': 1, 'max_depth': 9, 'min_child_weight': 50, 'tree_method': 'exact'}, 'n_downsample': 19, 'early_stopping': True, 'early_stopping_rounds': 50},
        {'tree_params': {'scale_pos_weight': 19, 'colsample_bytree': 0.95, 'eta': 0.05, 'subsample': 1, 'max_depth': 11, 'min_child_weight': 100, 'tree_method': 'exact'}, 'n_downsample': 19, 'early_stopping': True, 'early_stopping_rounds': 50},
        {'tree_params': {'scale_pos_weight': 19, 'colsample_bytree': 0.8, 'eta': 0.05, 'subsample': 0.95, 'max_depth': 8, 'min_child_weight': 10, 'tree_method': 'exact'}, 'n_downsample': 19, 'early_stopping': True, 'early_stopping_rounds': 50},
        {'tree_params': {'scale_pos_weight': 19, 'colsample_bytree': 0.9, 'eta': 0.05, 'subsample': 0.9, 'max_depth': 11, 'min_child_weight': 3, 'tree_method': 'exact'}, 'n_downsample': 19, 'early_stopping': True, 'early_stopping_rounds': 50},
    ]

    for features in all_features:
        for params in all_params:
            score, num_boost_rounds = score_cv_all_splits(features, params)
            with open('submit_evaluation.txt' , 'a') as stream:
                stream.write(str((score, num_boost_rounds, features, params)) + '\n')

evaluate_submission()
