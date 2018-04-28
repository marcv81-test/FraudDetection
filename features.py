import pandas
import gc

def engineer_time(dataset):
    """Engineers the time to seconds since midnight
    and a unique identifier for each hour since 1970."""
    print('Engineering time')
    dataset['day_seconds'] = (
        60 * 60 * dataset['click_time'].dt.hour +
        60 * dataset['click_time'].dt.minute +
        dataset['click_time'].dt.second
    ).astype('uint32')
    dataset['60m'] = (
        dataset['click_time'].astype('int64') // (60 * 60 * int(1e9))
    ).astype('uint32')

def engineer_clicks_per_group(dataset, group):
    """Engineers the number of clicks per group of other features."""
    name = 'clicks_per_' + '_'.join(group)
    print('Engineering', name)
    dataset[name] = pandas.Series(data=0, index=dataset.index, dtype='uint16')
    dataset[name] = dataset[[name] + group].groupby(group).transform('count')

def engineer_group(dataset, group):
    """Engineers additionnal groups. Group IDs will not match
    across splits; learning should not rely on such features."""
    name = '_'.join(group)
    print('Engineering', name)
    dataset[name] = dataset[group].groupby(group).grouper.group_info[0]

def engineer_unique_feature_per_group(dataset, feature, group):
    """Engineers the number of unique features per group of features."""
    name = 'unique_' + feature + '_per_' + '_'.join(group)
    print('Engineering', name)
    dataset[name] = dataset[[feature] + group].groupby(group)[feature].transform('nunique')

def feature_engineering(dataset):
    """Applies feature engineering to a dataset."""
    print('Feature engineering')
    engineer_time(dataset)
    engineer_group(dataset, ['device', 'os'])
    engineer_group(dataset, ['app', 'channel'])
    engineer_group(dataset, ['device', 'os', 'app', 'channel'])
    engineer_unique_feature_per_group(dataset, 'os', ['60m', 'ip'])
    engineer_clicks_per_group(dataset, ['60m', 'ip'])
    engineer_clicks_per_group(dataset, ['60m', 'ip', 'device_os'])
    engineer_clicks_per_group(dataset, ['60m', 'ip', 'app_channel'])
    engineer_clicks_per_group(dataset, ['60m', 'ip', 'device_os_app_channel'])
    dataset.drop([
        'click_time', '60m', 'ip',
        'device_os', 'app_channel', 'device_os_app_channel'
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
    gc.collect()

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
        gc.collect()

for d in range(1, 4):
    for r in range(4):
        process_downsample('day' + str(d) + '-' + str(r))
    process('day' + str(d) + '_test')
process('test')
