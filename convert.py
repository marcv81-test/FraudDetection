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

def csv_to_hdf5(csv_file, hdf5_file, columns):
    """Loads a CSV dataset and converts the timezeone. Saves as HDF5."""
    print('Loading', csv_file)
    dataset = pandas.read_csv(csv_file, usecols=columns, dtype=dtypes)
    tz_china = pytz.timezone('Asia/Shanghai')
    dataset['click_time'] = pandas.to_datetime(dataset['click_time'])
    dataset['click_time'] = dataset['click_time'].dt.tz_localize(pytz.utc)
    dataset['click_time'] = dataset['click_time'].dt.tz_convert(tz_china)
    print('Saving', hdf5_file)
    store = pandas.HDFStore(hdf5_file)
    store.put('dataset', dataset)
    store.close()
    gc.collect()

def click_id_to_hdf5(csv_file, hdf5_file):
    """Loads the click_id field of a CSV dataset. Saves as HDF5."""
    print('Loading', csv_file)
    dataset = pandas.read_csv(csv_file, usecols=['click_id'])
    print('Saving', hdf5_file)
    store = pandas.HDFStore(hdf5_file)
    store.put('dataset', dataset)
    store.close()
    gc.collect()

csv_to_hdf5('train.csv', 'train.h5', train_columns)
csv_to_hdf5('test.csv', 'test.h5', test_columns)
click_id_to_hdf5('test.csv', 'test_id.h5')
