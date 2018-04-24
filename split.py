import pandas
import gc

in_file = 'cache/train.h5'
print('Loading', in_file)
dataset = pandas.read_hdf(in_file)

def split_day(basename, y, m, d):
    """Extracts a single day from the training dataset. Saves as HDF5."""
    for r in range(4):
        out_file = 'cache/' + basename + '-' + str(r) + '.h5'
        print('Creating', out_file)
        hours = [(6 * r) + i for i in range(6)]
        print('Hours', hours)
        sub_dataset = dataset[
            (dataset['click_time'].dt.year == y) &
            (dataset['click_time'].dt.month == m) &
            (dataset['click_time'].dt.day == d) &
            (dataset['click_time'].dt.hour.isin(hours))]
        store = pandas.HDFStore(out_file)
        store.put('dataset', sub_dataset)
        store.close()
        gc.collect()

split_day('day1', 2017, 11, 7)
split_day('day2', 2017, 11, 8)
split_day('day3', 2017, 11, 9)

def split_day_test(basename, y, m, d):
    """Extracts the time ranges of the challenge test dataset
    from a single day of the training dataset. Saves as HDF5."""
    out_file = 'cache/' + basename + '.h5'
    print('Creating', out_file)
    sub_dataset = dataset[
        (dataset['click_time'].dt.year == y) &
        (dataset['click_time'].dt.month == m) &
        (dataset['click_time'].dt.day == d) &
        (dataset['click_time'].dt.hour.isin([12, 13, 17, 18, 21, 22]))]
    store = pandas.HDFStore(out_file)
    store.put('dataset', sub_dataset)
    store.close()
    gc.collect()

split_day_test('day1_test', 2017, 11, 7)
split_day_test('day2_test', 2017, 11, 8)
split_day_test('day3_test', 2017, 11, 9)
