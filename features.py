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

def add_features_train(in_file, out_file, downsample=False):
    """Adds features to a training dataset."""
    print('Loading', in_file)
    dataset = pandas.read_hdf(in_file)
    print('Engineering features')
    dataset['ssm'] = seconds_since_midnight(dataset)
    dataset['1m_ip'] = clicks_per_ip_in_time_range(dataset, 1)
    dataset['10m_ip'] = clicks_per_ip_in_time_range(dataset, 10)
    dataset['60m_ip'] = clicks_per_ip_in_time_range(dataset, 60)
    if downsample:
        print('Downsampling')
        dataset = downsample_train(dataset)
    print('Converting to Numpy arrays')
    x = dataset.as_matrix(columns=[
        'app', 'device', 'os', 'channel', 'ssm'
        '1m_ip', '10m_ip', '60m_ip'])
    y = dataset.as_matrix(columns=['is_attributed'])
    print('Saving', out_file)
    numpy.savez(out_file, x, y)
    gc.collect()

def downsample_train(dataset, n=9):
    """Downsamples a training dataset. Selects all the attributed clicks.
    Randomly selects N times as many non-attributed clicks. Shuffles the results."""
    attributed = dataset[dataset['is_attributed'] == True]
    not_attributed = dataset[dataset['is_attributed'] == False].sample(n=n*len(attributed))
    return attributed.append(not_attributed).sample(frac=1)

add_features_train('split_day1.h5', 'fe_day1.npz', downsample=True)
add_features_train('split_day2.h5', 'fe_day2.npz', downsample=True)
add_features_train('split_day3.h5', 'fe_day3.npz', downsample=True)

add_features_train('split_test1.h5', 'fe_test1.npz')
add_features_train('split_test2.h5', 'fe_test2.npz')
add_features_train('split_test3.h5', 'fe_test3.npz')
