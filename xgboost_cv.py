import numpy
import pandas
import sklearn.metrics
import xgboost

x_columns = ['app', 'device', 'os', 'channel', 'ssm', '1m_ip', '10m_ip', '60m_ip']
y_columns = ['is_attributed']

train_files = ('feat_day1.h5', 'feat_day2.h5', 'feat_day3.h5')
test_files = ('feat_test1.h5', 'feat_test2.h5', 'feat_test3.h5')

def downsample(dataset, n):
    """Downsamples a training dataset. Selects all the attributed clicks.
    Randomly selects N times as many non-attributed clicks. Shuffles the results."""
    print('Downsampling')
    attributed = dataset[dataset['is_attributed'] == True]
    not_attributed = dataset[dataset['is_attributed'] == False].sample(n=n*len(attributed))
    return attributed.append(not_attributed).sample(frac=1)

def load_train(filename, downsample_n=0):
    """Loads a training dataset."""
    print('Loading', filename)
    dataset = pandas.read_hdf(filename)
    if downsample_n != 0:
        dataset = downsample(dataset, downsample_n)
    x = dataset.as_matrix(columns=x_columns)
    y = dataset.as_matrix(columns=y_columns)
    return x, y

def load_multi_train(filenames, downsample_n):
    """Loads and aggregates splits of a training dataset."""
    xs = []
    ys = []
    for filename in filenames:
        x, y = load_train(filename, downsample_n)
        xs.append(x)
        ys.append(y)
    return numpy.concatenate(xs), numpy.concatenate(ys)

def cv_train_dataset(n, downsample_n):
    """Returns the cross-validation training dataset for one of the splits."""
    assert n in (0, 1, 2)
    print('Loading training split', n)
    if n == 0: return load_multi_train((train_files[1], train_files[2]), downsample_n)
    if n == 1: return load_multi_train((train_files[0], train_files[2]), downsample_n)
    if n == 2: return load_multi_train((train_files[0], train_files[1]), downsample_n)

def cv_test_dataset(n):
    """Returns the cross-validation test dataset for one of the splits."""
    assert n in (0, 1, 2)
    print('Loading test split', n)
    if n == 0: return load_train(test_files[0])
    if n == 1: return load_train(test_files[1])
    if n == 2: return load_train(test_files[2])

def cv_score(n, max_depth, n_estimators, downsample_n):
    """Returns the cross-validation AUROC score on one of the splits."""
    x_train, y_train = cv_train_dataset(n, downsample_n)
    model = xgboost.XGBRegressor(
        n_jobs=4,
        max_depth=max_depth,
        n_estimators=n_estimators,
        scale_pos_weight=downsample_n)
    print('Training')
    model.fit(x_train, y_train)
    x_test, y_test = cv_test_dataset(n)
    print('Predicting')
    y_predict = model.predict(x_test)
    print('Scoring')
    score = sklearn.metrics.roc_auc_score(y_test, y_predict)
    print('Score', n, score)
    return score

score = 0
for i in range(3):
    score += cv_score(i, 10, 100, 9)
score /= 3
print('Average score', score)
