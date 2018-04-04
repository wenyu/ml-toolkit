import multiprocessing as mp
import sys
import numpy as np


def fault_tolerant_endless_generator(lambda_iterable):
    """
    Restarts generator upon error in iteration.

    :type lambda_iterable: function
    :return: None
    """
    gen = lambda_iterable()
    while True:
        try:
            for item in gen:
                yield item
        except KeyboardInterrupt:
            sys.stderr.write("Keyboard Interrupt\n")
            break
        except StopIteration as e:
            sys.stderr.write(str(e) + '\n')
            gen = lambda_iterable()


def parallel_map_generator(func, iterable, jobs=-1, batch_size=-1):
    """
    Perform parallel mapping on iterable.

    :type func: function
    :type iterable: iterable
    :type batch_size: int
    :type jobs: int
    :return:
    """
    if jobs < 0:
        jobs = mp.cpu_count()
    if batch_size < 0:
        batch_size = jobs

    pool = mp.Pool(jobs)
    pmap = pool.map
    batch = []

    for item in iterable:
        batch.append(item)
        if len(batch) >= batch_size:
            result = pmap(func, batch)
            for result_item in result:
                yield result_item
            batch = []
    else:
        if len(batch) > 0:
            result = pmap(func, batch)
            for result_item in result:
                yield result_item


def evenly_distributed_labeling_generator(X_data, Y_data):
    data = {}
    for x, y in zip(X_data, Y_data):
        if y in data:
            data[y].append(x)
        else:
            data[y] = [x]

    gen = map(lambda (y, x): (fault_tolerant_endless_generator(lambda:x), y), data.items())

    while True:
        for x_gen, y in gen:
            yield x_gen.next(), y


def batch_x_y_generator(generator, batch_size=32):
    X, Y = [], []
    for x, y in generator:
        X.append(x)
        Y.append(y)
        if len(Y) >= batch_size:
            yield np.array(X), np.array(Y)
            X, Y = [], []
    else:
        if len(Y) > 0:
            yield np.array(X), np.array(Y)
