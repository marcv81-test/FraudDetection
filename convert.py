import pandas
import pytz
import gc

train_columns = ['ip', 'app', 'device', 'os', 'channel', 'click_time', 'is_attributed']
test_columns = ['ip', 'app', 'device', 'os', 'channel', 'click_time']
dtypes = {
    'ip': 'uint32',
    'app': 'uint16',
    'device': 'uint16',
    'os': 'uint16',
    'channel': 'uint16',
    'is_attributed': 'bool',
}

def convert_dataset(basename, columns):
    """Loads a CSV dataset and converts the timezeone. Saves as HDF5."""
    in_file = basename + '.csv'
    print('Loading', in_file)
    dataset = pandas.read_csv(in_file, usecols=columns, dtype=dtypes)
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
