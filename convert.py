import pandas
import pytz
import gc

train_columns = ['ip', 'app', 'device', 'os', 'channel', 'click_time', 'is_attributed']
test_columns = ['ip', 'app', 'device', 'os', 'channel', 'click_time']
dtypes = {
    'is_attributed': 'bool',
    'ip': 'uint64',
    'app': 'uint32',
    'device': 'uint32',
    'os': 'uint32',
    'channel': 'uint32',
}

def convert_dataset(basename, columns):
    """Loads a CSV dataset and converts the timezeone. Saves as HDF5."""
    in_file = basename + '.csv'
    print('Loading', in_file)
    dataset = pandas.read_csv(in_file, usecols=columns, dtype=dtypes)
    # Safety check before converting to uint32
    assert dataset['ip'].max() < pow(2, 32)
    dataset['ip'] = dataset['ip'].astype('uint32')
    # Safety checks before converting to uint16
    for feature in ('app', 'device', 'os', 'channel'):
        assert dataset[feature].max() < pow(2, 16)
        dataset[feature] = dataset[feature].astype('uint16')
    tz_china = pytz.timezone('Asia/Shanghai')
    dataset['click_time'] = pandas.to_datetime(dataset['click_time'])
    dataset['click_time'] = dataset['click_time'].dt.tz_localize(pytz.utc)
    dataset['click_time'] = dataset['click_time'].dt.tz_convert(tz_china)
    out_file = 'cache/' + basename + '.h5'
    print('Saving', out_file)
    store = pandas.HDFStore(out_file)
    store.put('dataset', dataset)
    store.close()
    gc.collect()

def convert_id(basename):
    """Loads the click_id field of a CSV dataset. Saves as HDF5."""
    in_file = basename + '.csv'
    print('Loading', in_file)
    dataset = pandas.read_csv(in_file, usecols=['click_id'])
    out_file = 'cache/id_' + basename + '.h5'
    print('Saving', out_file)
    store = pandas.HDFStore(out_file)
    store.put('dataset', dataset)
    store.close()
    gc.collect()

convert_dataset('train', train_columns)
convert_dataset('test', test_columns)
convert_id('test')
