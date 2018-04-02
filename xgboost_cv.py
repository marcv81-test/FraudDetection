import numpy
import sklearn.metrics
import xgboost

train_files = ('fe_day1.npz', 'fe_day2.npz', 'fe_day3.npz')
test_files = ('fe_test1.npz', 'fe_test2.npz', 'fe_test3.npz')
assert len(day_files) == len(test_files)

def load(filename):
    """Loads a training dataset."""
    print('Loading', filename)
    npz = numpy.load(filename)
    x = npz['arr_0']
    y = npz['arr_1'].ravel()
    return x, y

def load_pair(filename1, filename2):
    """Loads and merges two training datasets."""
    x1, y1 = load(filename1)
    x2, y2 = load(filename2)
    return numpy.concatenate((x1, x2)), numpy.concatenate((y1, y2))

def cv_train(n):
    """Returns the cross-validation training dataset for one of the splits."""
    assert n in (0, 1, 2)
    print('Loading training split', n)
    if n == 0: return load_pair(train_files[1], train_files[2])
    if n == 1: return load_pair(train_files[0], train_files[2])
    if n == 2: return load_pair(train_files[0], train_files[1])

def cv_test(n):
    """Returns the cross-validation test dataset for on of the splits."""
    assert n in (0, 1, 2)
    print('Loading test split', n)
    if n == 0: return load(test_files[0])
    if n == 1: return load(test_files[1])
    if n == 2: return load(test_files[2])

def cv_score(n):
    """Returns the cross-validation AUROC score on one of the splits."""
    x_train, y_train = cv_train(n)
    model = xgboost.XGBRegressor(max_depth=10, n_estimators=100, n_jobs=4, scale_pos_weight=9)
    print('Training')
    model.fit(x_train, y_train)
    x_test, y_test = cv_test(n)
    print('Predicting')
    y_predict = model.predict(x_test)
    print('Scoring')
    score = sklearn.metrics.roc_auc_score(y_test, y_predict)
    print('Score', n, score)
    return score

score = 0
for i in range(3):
    score += cv_score(i)
score /= 3
print('Average score', score)
