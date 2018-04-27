import numpy
import pandas
import random
import scipy.stats
import xgboost

x_columns = [
    'app', 'device', 'os', 'channel',
    'day_seconds',
    'clicks_per_60m_ip',
    'clicks_per_60m_ip_device_os',
    'clicks_per_60m_ip_app_channel',
    'clicks_per_60m_ip_device_os_app_channel',
    'unique_os_per_60m_ip',
]
y_columns = ['is_attributed']

# Cross-validation

def load_cv_train(n_split, n_downsample):
    """Returns the training dataset for one of the cross-validation splits."""
    assert n_split in (1, 2, 3)
    xs, ys = [], []
    for n_day in range(1, 4):
        if n_day != n_split:
            for r in range(4):
                in_file = 'cache/feat_ds' + str(n_downsample) + '_day' + str(n_day) + '-' + str(r) + '.h5'
                print('Loading', in_file)
                dataset = pandas.read_hdf(in_file)
                xs.append(dataset.as_matrix(columns=x_columns))
                ys.append(dataset.as_matrix(columns=y_columns))
    return numpy.concatenate(xs), numpy.concatenate(ys)

def load_cv_test(n_split):
    """Returns the test dataset for one of the cross-validation splits."""
    assert n_split in (1, 2, 3)
    in_file = 'cache/feat_day' + str(n_split) + '_test.h5'
    print('Loading', in_file)
    dataset = pandas.read_hdf(in_file)
    x = dataset.as_matrix(columns=x_columns)
    y = dataset.as_matrix(columns=y_columns)
    return x, y

def score_cv_split(n_split, params, best_stop):
    """Returns the AUROC score for each boosting round on one of the cross-validation
    splits. Stops when the cross-validation score decreases if requested."""
    x_train, y_train = load_cv_train(n_split, params['n_downsample'])
    dtrain = xgboost.DMatrix(x_train, label=y_train)
    x_test, y_test = load_cv_test(n_split)
    dtest = xgboost.DMatrix(x_test, label=y_test)
    tree_params = dict(params['tree_params'])
    tree_params['eval_metric'] = 'auc'
    tree_params['silent'] = 1
    results = dict()
    if best_stop:
        model = xgboost.train(
            tree_params, dtrain,
            evals=[(dtest, 'test')],
            evals_result=results,
            num_boost_round=1000,
            early_stopping_rounds=25,
            verbose_eval=False)
    else:
        model = xgboost.train(
            tree_params, dtrain,
            evals=[(dtest, 'test')],
            evals_result=results,
            num_boost_round=params['num_boost_round'],
            verbose_eval=False)
    results = results['test']['auc']
    best_auc, best_i = 0, 0
    for i, auc in enumerate(results):
        if auc > best_auc:
            best_auc, best_i = auc, i
    best = (best_auc, best_i + 1)
    print(best)
    return results

def score_cv_all_splits(params, best_stop):
    """Returns the best average AUROC score on all of the cross-validation splits
    and the associated number of boosting rounds. Stops when the cross-validation
    score decreases if requested."""
    results = []
    for n_split in range(1, 4):
        print('Split', str(n_split))
        results.append(score_cv_split(n_split, params, best_stop))
    length = min([len(result) for result in results])
    best_auc, best_i = 0, 0
    for i in range(length):
        auc = numpy.mean([result[i] for result in results])
        if auc > best_auc:
            best_auc, best_i = auc, i
    best = (best_auc, best_i + 1)
    print(best)
    return best

# Submission

def load_full_train(n_downsample):
    """Returns the entire training dataset."""
    xs, ys = [], []
    for n_day in range(1, 4):
        for r in range(4):
            in_file = 'cache/feat_ds' + str(n_downsample) + '_day' + str(n_day) + '-' + str(r) + '.h5'
            print('Loading', in_file)
            dataset = pandas.read_hdf(in_file)
            xs.append(dataset.as_matrix(columns=x_columns))
            ys.append(dataset.as_matrix(columns=y_columns))
    return numpy.concatenate(xs), numpy.concatenate(ys)

def load_full_test():
    """Returns the entire test dataset."""
    in_file = 'cache/feat_test.h5'
    print('Loading', in_file)
    dataset = pandas.read_hdf(in_file)
    x = dataset.as_matrix(columns=x_columns)
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

# Main

# CV 0.976332
params = {'tree_params': {'scale_pos_weight': 49, 'colsample_bytree': 0.9, 'eta': 0.05, 'subsample': 0.9, 'max_depth': 11, 'min_child_weight': 3, 'tree_method': 'exact'}, 'n_downsample': 49, 'num_boost_round': 154}
submit(params)
