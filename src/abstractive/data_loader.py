import gc
import glob
import random

import torch

from others.logging import logger

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


class AbstractiveBatch(object):
    def _pad(self, data, height, width, pad_id):
        """ ? """
        rtn_data = [d + [pad_id] * (width - len(d)) for d in data]
        rtn_length = [len(d) for d in data]
        rtn_data = rtn_data + [[pad_id] * width] * (height - len(data))
        rtn_length = rtn_length + [0] * (height - len(data))

        return rtn_data, rtn_length

    def __init__(self, data=None, hier=False, pad_id=None, device=None, is_test=False):
        """Create a Batch from a list of examples."""
        if data is not None:
            self.batch_size = len(data)
            src = [x[0] for x in data]
            tgt = [x[1] for x in data]

            if (hier):
                max_nblock = max([len(e) for e in src])
                max_ntoken = max([max([len(p) for p in e]) for e in src])
                _src = [self._pad(e, max_nblock, max_ntoken, pad_id) for e in src]
                src = torch.stack([torch.tensor(e[0]) for e in _src])


            else:
                _src = self._pad(src, width=max([len(d) for d in src]), height=len(src), pad_id=pad_id)
                src = torch.tensor(_src[0])  # batch_size, src_len

            setattr(self, 'src', src.to(device))

            _tgt = self._pad(tgt, width=max([len(d) for d in tgt]), height=len(tgt), pad_id=pad_id)
            tgt = torch.tensor(_tgt[0]).transpose(0, 1)
            setattr(self, 'tgt', tgt.to(device))

            if (is_test):
                tgt_str = [x[2] for x in data]
                setattr(self, 'tgt_str', tgt_str)

    def __len__(self):
        return self.batch_size




def load_dataset(args, corpus_type, shuffle):
    """
    Dataset generator. Don't do extra stuff here, like printing,
    because they will be postponed to the first loading time.
    Args:
        corpus_type: 'train' or 'valid'
    Returns:
        A list of dataset, the dataset(s) are lazily loaded.
    """
    assert corpus_type in ["train", "valid", "test"]

    def _lazy_dataset_loader(pt_file, corpus_type):
        dataset = torch.load(pt_file)
        logger.info('Loading %s dataset from %s, number of examples: %d' %
                    (corpus_type, pt_file, len(dataset)))
        return dataset

    # Sort the glob output by file name (by increasing indexes).
    pts = sorted(glob.glob(args.data_path + '.' + corpus_type + '.[0-9]*.pt'))
    if pts:
        if (shuffle):
            random.shuffle(pts)

        for pt in pts:
            yield _lazy_dataset_loader(pt, corpus_type)
    else:
        # Only one inputters.*Dataset, simple!
        pt = args.data_path + '.' + corpus_type + '.pt'
        yield _lazy_dataset_loader(pt, corpus_type)


class AbstractiveDataloader(object):
    def __init__(self, args, datasets, symbols, batch_size,
                 device, shuffle, is_test):
        self.args = args
        self.datasets = datasets
        self.symbols = symbols
        self.batch_size = batch_size
        self.device = device
        self.shuffle = shuffle
        self.is_test = is_test
        self.cur_iter = self._next_dataset_iterator(datasets)
        assert self.cur_iter is not None

    def __iter__(self):
        dataset_iter = (d for d in self.datasets)
        while self.cur_iter is not None:
            for batch in self.cur_iter:
                yield batch
            self.cur_iter = self._next_dataset_iterator(dataset_iter)

    def _next_dataset_iterator(self, dataset_iter):
        try:
            # Drop the current dataset for decreasing memory
            if hasattr(self, "cur_dataset"):
                self.cur_dataset = None
                gc.collect()
                del self.cur_dataset
                gc.collect()

            self.cur_dataset = next(dataset_iter)
        except StopIteration:
            return None

        return AbstracticeIterator(args = self.args,
            dataset=self.cur_dataset, symbols=self.symbols, batch_size=self.batch_size,
            device=self.device, shuffle=self.shuffle, is_test=self.is_test)


class AbstracticeIterator(object):
    def __init__(self, args, dataset, symbols, batch_size, device=None, is_test=False,
                 shuffle=True):
        self.args = args
        self.batch_size, self.is_test, self.dataset = batch_size, is_test, dataset
        self.iterations = 0
        self.device = device
        self.shuffle = shuffle

        # self.secondary_sort_key = lambda x: len(x[0])
        # self.secondary_sort_key = lambda x: sum([len(xi) for xi in x[0]])
        # self.prime_sort_key = lambda x: len(x[1])
        self.secondary_sort_key = lambda x: sum([len(xi) for xi in x[0]])
        self.prime_sort_key = lambda x: len(x[1])
        self._iterations_this_epoch = 0


        self.symbols = symbols

    def data(self):
        if self.shuffle:
            random.shuffle(self.dataset)
        xs = self.dataset
        return xs

    def preprocess(self, ex):

        sos_id = self.symbols['BOS']
        eos_id = self.symbols['EOS']
        eot_id = self.symbols['EOT']
        eop_id = self.symbols['EOP']
        eoq_id = self.symbols['EOQ']
        src, tgt, tgt_str = ex['src'], ex['tgt'], ex['tgt_str']
        if (not self.args.hier):
            src = sum([p + [eop_id] for p in src], [])[:-1][:self.args.trunc_src_ntoken] + [
                eos_id]
            return src, tgt, tgt_str

        return src[:self.args.trunc_src_nblock], tgt, tgt_str

    def simple_batch_size_fn(self, new, count):
        src, tgt = new[0], new[1]

        global max_src_in_batch, max_tgt_in_batch
        if count == 1:
            max_src_in_batch = 0
        if (self.args.hier):
            max_src_in_batch = max(max_src_in_batch, sum([len(p) for p in src]))
        else:
            max_src_in_batch = max(max_src_in_batch, len(src))
        src_elements = count * max_src_in_batch
        return src_elements

    def get_batch(self, data, batch_size):
        """Yield elements from data in chunks of batch_size."""
        minibatch, size_so_far = [], 0
        for ex in data:
            minibatch.append(ex)
            size_so_far = self.simple_batch_size_fn(ex, len(minibatch))
            if size_so_far == batch_size:
                yield minibatch
                minibatch, size_so_far = [], 0
            elif size_so_far > batch_size:
                yield minibatch[:-1]
                minibatch, size_so_far = minibatch[-1:], self.simple_batch_size_fn(ex, 1)
        if minibatch:
            yield minibatch

    def batch_buffer(self, data, batch_size):
        minibatch, size_so_far = [], 0
        for ex in data:
            ex = self.preprocess(ex)
            minibatch.append(ex)
            size_so_far = self.simple_batch_size_fn(ex, len(minibatch))
            if size_so_far == batch_size:
                yield minibatch
                minibatch, size_so_far = [], 0
            elif size_so_far > batch_size:
                yield minibatch[:-1]
                minibatch, size_so_far = minibatch[-1:], self.simple_batch_size_fn(ex, 1)
        if minibatch:
            yield minibatch

    def create_batches(self):
        """ Create batches """
        data = self.data()
        for buffer in self.batch_buffer(data, self.batch_size * 100):
            if (self.args.mode != 'train'):
                p_batch = self.get_batch(
                    sorted(sorted(buffer, key=self.prime_sort_key), key=self.secondary_sort_key),
                    self.batch_size)
            else:
                p_batch = self.get_batch(
                    sorted(sorted(buffer, key=self.secondary_sort_key), key=self.prime_sort_key),
                    self.batch_size)

            p_batch = list(p_batch)

            if (self.shuffle):
                random.shuffle(p_batch)
            for b in p_batch:
                if(len(b)==0):
                    continue
                yield b

    def __iter__(self):

        while True:
            self.batches = self.create_batches()
            for idx, minibatch in enumerate(self.batches):
                if self._iterations_this_epoch > idx:
                    continue
                self.iterations += 1
                self._iterations_this_epoch += 1
                batch = AbstractiveBatch(minibatch, self.args.hier, self.symbols['PAD'], self.device, self.is_test)

                yield batch
            return