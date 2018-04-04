import pandas
import gc

def seconds_since_midnight(dataset):
    """Feature engineering: seconds since midnight."""
    return (
        (dataset['click_time'].dt.hour * 3600) +
        (dataset['click_time'].dt.minute * 60) +
        (dataset['click_time'].dt.second)
    ).astype('uint32')

def clicks_per_ip_in_time_range(dataset, minutes):
    """Feature engineering: clicks per IP in a range of N minutes."""
    assert 60 % minutes == 0
    assert minutes <= 60
    return dataset.groupby([
        dataset['click_time'].astype('int64') // (minutes * 60 * int(1e9)),
        dataset['ip'],
    ])['ip'].transform('count').astype('uint16')

def feature_engineering(dataset):
    """Applies feature engineering to a dataset."""
    print('Feature engineering')
    dataset['ssm'] = seconds_since_midnight(dataset)
    dataset['1m_ip'] = clicks_per_ip_in_time_range(dataset, 1)
    dataset['10m_ip'] = clicks_per_ip_in_time_range(dataset, 10)
    dataset['60m_ip'] = clicks_per_ip_in_time_range(dataset, 60)
    dataset.drop(['click_time', 'ip'], axis=1, inplace=True)

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

def process_downsample(basename, n_downsample=(1, 2, 3, 4, 9, 19)):
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

process_downsample('day1')
process_downsample('day2')
process_downsample('day3')

process('day1_test')
process('day2_test')
process('day3_test')
process('test')
