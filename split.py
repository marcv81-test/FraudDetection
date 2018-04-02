import pandas
import gc

dataset = pandas.read_hdf('train.h5')

def split_day(hdf5_file, y, m, d):
    """Extracts a single day from the training dataset. Saves as HDF5."""
    print('Creating', hdf5_file)
    sub_dataset = dataset[
        (dataset['click_time'].dt.year == y) &
        (dataset['click_time'].dt.month == m) &
        (dataset['click_time'].dt.day == d)]
    store = pandas.HDFStore(hdf5_file)
    store.put('dataset', sub_dataset)
    store.close()
    gc.collect()

split_day('split_day1.h5', 2017, 11, 7)
split_day('split_day2.h5', 2017, 11, 8)
split_day('split_day3.h5', 2017, 11, 9)

def split_test_ranges(hdf5_file, y, m, d):
    """Extracts the time ranges of the challenge test dataset
    from a single day of the training dataset. Saves as HDF5."""
    print('Creating', hdf5_file)
    sub_dataset = dataset[
        (dataset['click_time'].dt.year == y) &
        (dataset['click_time'].dt.month == m) &
        (dataset['click_time'].dt.day == d) &
        (dataset['click_time'].dt.hour.isin([12, 13, 17, 18, 21, 22]))]
    store = pandas.HDFStore(hdf5_file)
    store.put('dataset', sub_dataset)
    store.close()
    gc.collect()

split_test_ranges('split_test1.h5', 2017, 11, 7)
split_test_ranges('split_test2.h5', 2017, 11, 8)
split_test_ranges('split_test3.h5', 2017, 11, 9)
