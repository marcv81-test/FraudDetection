import multiprocessing
import pandas

def engineer_time(dataset):
    """Sorts the dataset by increasing click time. Creates a copy
    of the click time supporting subtraction. Calculates the seconds
    since midnight. Generates a unique identifier for each hour."""
    print('Engineering time')
    dataset.sort_values(by='click_time', inplace=True)
    dataset['raw_click_time'] = dataset['click_time'].dt.tz_localize(None)
    # Max = 60 * 60 * 24 = 86400, safe as uint32
    dataset['day_seconds'] = (
        60 * 60 * dataset['click_time'].dt.hour +
        60 * dataset['click_time'].dt.minute +
        dataset['click_time'].dt.second
    ).astype('uint32')
    # Current epoch / (60 * 60) is safe as uint32
    dataset['60m'] = (
        dataset['click_time'].astype('int64') // (60 * 60 * int(1e9))
    ).astype('uint32')

def engineer_clicks_per_group(dataset, group):
    """Aggregates the number of clicks per group of features."""
    name = 'clicks_per_' + '_'.join(group)
    print('Engineering', name)
    dataset[name] = pandas.Series(data=0, index=dataset.index, dtype='uint64')
    dataset[name] = dataset[[name] + group].groupby(group)[name].transform('count')
    # Safety check before converting to uint32
    assert dataset[name].max() < pow(2, 32)
    dataset[name] = dataset[name].astype('uint32')

def engineer_previous_click_per_group(dataset, group):
    """Calculates the time in seconds since the previous click associated to
    the same group of features. Defaults to 1 hour for the first click."""
    name = 'previous_click_per_' + '_'.join(group)
    print('Engineering', name)
    series = (
        dataset['raw_click_time'] -
        dataset[['raw_click_time'] + group].groupby(group)['raw_click_time'].shift(1))
    series.fillna(60 * 60, inplace=True)
    series = (series.astype('int64') // int(1e9))
    # Safety check before converting to uint16
    assert series.max() <= 3600
    dataset[name] = series.astype('uint16')

def engineer_unique_feature_per_group(dataset, feature, group):
    """Counts the number of unique feature values per group of features."""
    name = 'unique_' + feature + '_per_' + '_'.join(group)
    print('Engineering', name)
    series = dataset[[feature] + group].groupby(group)[feature].transform('nunique')
    # Safety check before converting to uint16
    assert series.max() < pow(2, 16)
    dataset[name] = series.astype('uint16')

def iter_group(features):
    """Iterates over all the non-empty subsets of a set of features."""
    for i in range(1, pow(2, len(features))):
        selected = []
        for j in range(len(features)):
            if i & (1 << j) > 0:
                selected.append(features[j])
        yield selected

def iter_pair(features):
    """Iterates over all the ordered pairs from a set of features."""
    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            yield features[i], features[j]
            yield features[j], features[i]

def feature_engineering(dataset):
    """Adds derived features to a dataset.
    Drops the features we should not learn from."""
    print('Feature engineering')
    engineer_time(dataset)
    for group in iter_group(['ip', 'device', 'os', 'app', 'channel']):
        group = ['60m'] + group
        engineer_clicks_per_group(dataset, group)
        engineer_previous_click_per_group(dataset, group)
    for feature, group in iter_pair(['ip', 'device', 'os', 'app', 'channel']):
        group = ['60m', group]
        engineer_unique_feature_per_group(dataset, feature, group)
    dataset.drop([
        'click_time', 'raw_click_time', '60m', 'ip',
    ], axis=1, inplace=True)
    print(dataset.dtypes)

def downsample(dataset, n):
    """Downsamples a training dataset. Selects all the attributed clicks.
    Randomly selects N times as many non-attributed clicks. Shuffles the results."""
    print('Downsampling')
    attributed = dataset[dataset['is_attributed'] == True]
    not_attributed = dataset[dataset['is_attributed'] == False].sample(n=n*len(attributed))
    return attributed.append(not_attributed).sample(frac=1)

def process(basename):
    """Opens a dataset, applies feature engineering, and saves the results."""
    in_file = 'cache/' + basename + '.h5'
    print('Loading', in_file)
    dataset = pandas.read_hdf(in_file)
    feature_engineering(dataset)
    out_file = 'cache/feat_' + basename + '.h5'
    print('Saving', out_file)
    store = pandas.HDFStore(out_file)
    store.put('dataset', dataset)
    store.close()

def process_downsample(basename, n_downsample=[1, 19, 49]):
    """Opens a dataset, applies feature engineering, and saves
    different downsampled versions of the results."""
    in_file = 'cache/' + basename + '.h5'
    print('Loading', in_file)
    dataset = pandas.read_hdf(in_file)
    feature_engineering(dataset)
    for n in n_downsample:
        sub_dataset = downsample(dataset, n)
        out_file = 'cache/feat_ds' + str(n) + '_' + basename + '.h5'
        print('Saving', out_file)
        store = pandas.HDFStore(out_file)
        store.put('dataset', sub_dataset)
        store.close()

def run(function, parameter):
    """Runs a function in a separate thread to work around the memory leaks."""
    with multiprocessing.Pool(1) as pool:
        pool.map(function, [parameter])
        pool.terminate()

for d in range(1, 4):
    run(process, 'day' + str(d) + '_test')
    for r in range(4):
        run(process_downsample, 'day' + str(d) + '-' + str(r))
run(process, 'test')
