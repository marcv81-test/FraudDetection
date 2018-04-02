import pandas
import numpy
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

def features_train(in_file, out_file):
    """Adds features to a training dataset."""
    print('Loading', in_file)
    dataset = pandas.read_hdf(in_file)
    print('Engineering features')
    dataset['ssm'] = seconds_since_midnight(dataset)
    dataset['1m_ip'] = clicks_per_ip_in_time_range(dataset, 1)
    dataset['2m_ip'] = clicks_per_ip_in_time_range(dataset, 2)
    dataset['5m_ip'] = clicks_per_ip_in_time_range(dataset, 5)
    dataset['10m_ip'] = clicks_per_ip_in_time_range(dataset, 10)
    dataset['30m_ip'] = clicks_per_ip_in_time_range(dataset, 30)
    dataset['60m_ip'] = clicks_per_ip_in_time_range(dataset, 60)
    print('Converting to Numpy arrays')
    x = dataset.as_matrix(columns=[
        'app', 'device', 'os', 'channel', 'ssm'
        '1m_ip', '2m_ip', '5m_ip', '10m_ip', '30m_ip', '60m_ip'])
    y = dataset.as_matrix(columns=['is_attributed'])
    print('Saving', out_file)
    numpy.savez(out_file, x, y)
    gc.collect()

features_train('split_tiny1.h5', 'fe_tiny1.npz')
features_train('split_tiny2.h5', 'fe_tiny2.npz')
features_train('split_tiny3.h5', 'fe_tiny3.npz')

features_train('split_test1.h5', 'fe_test1.npz')
features_train('split_test2.h5', 'fe_test2.npz')
features_train('split_test3.h5', 'fe_test3.npz')
