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

def add_features(in_file, out_file):
    """Adds features to a dataset."""
    print('Loading', in_file)
    dataset = pandas.read_hdf(in_file)
    print('Adding features')
    dataset['ssm'] = seconds_since_midnight(dataset)
    dataset['1m_ip'] = clicks_per_ip_in_time_range(dataset, 1)
    dataset['10m_ip'] = clicks_per_ip_in_time_range(dataset, 10)
    dataset['60m_ip'] = clicks_per_ip_in_time_range(dataset, 60)
    print('Saving', out_file)
    store = pandas.HDFStore(out_file)
    store.put('dataset', dataset)
    store.close()
    gc.collect()

add_features('split_day1.h5', 'feat_day1.h5')
add_features('split_day2.h5', 'feat_day2.h5')
add_features('split_day3.h5', 'feat_day3.h5')

add_features('split_test1.h5', 'feat_test1.h5')
add_features('split_test2.h5', 'feat_test2.h5')
add_features('split_test3.h5', 'feat_test3.h5')

add_features('test.h5', 'feat_test.h5')
