'''
this file is modified from keras implemention of data process multi-threading,
see https://github.com/fchollet/keras/blob/master/keras/utils/data_utils.py
'''
import csv
import os
import time
import numpy as np
import threading
import multiprocessing
try:
    import queue
except ImportError:
    import Queue as queue


class GeneratorEnqueuer():
    """Builds a queue out of a data generator.

    Used in `fit_generator`, `evaluate_generator`, `predict_generator`.

    # Arguments
        generator: a generator function which endlessly yields data
        use_multiprocessing: use multiprocessing if True, otherwise threading
        wait_time: time to sleep in-between calls to `put()`
        random_seed: Initial seed for workers,
            will be incremented by one for each workers.
    """

    def __init__(self, generator,
                 use_multiprocessing=False,
                 wait_time=0.05,
                 random_seed=None):
        self.wait_time = wait_time
        self._generator = generator
        self._use_multiprocessing = use_multiprocessing
        self._threads = []
        self._stop_event = None
        self.queue = None
        self.random_seed = random_seed

    def start(self, workers=1, max_queue_size=10):
        """Kicks off threads which add data from the generator into the queue.

        # Arguments
            workers: number of worker threads
            max_queue_size: queue size
                (when full, threads could block on `put()`)
        """

        def data_generator_task():
            while not self._stop_event.is_set():
                try:
                    if self._use_multiprocessing or self.queue.qsize() < max_queue_size:
                        generator_output = next(self._generator)
                        self.queue.put(generator_output)
                    else:
                        time.sleep(self.wait_time)
                except Exception:
                    self._stop_event.set()
                    raise

        try:
            if self._use_multiprocessing:
                self.queue = multiprocessing.Queue(maxsize=max_queue_size)
                self._stop_event = multiprocessing.Event()
            else:
                self.queue = queue.Queue()
                self._stop_event = threading.Event()

            for _ in range(workers):
                if self._use_multiprocessing:
                    # Reset random seed else all children processes
                    # share the same seed
                    np.random.seed(self.random_seed)
                    thread = multiprocessing.Process(target=data_generator_task)
                    thread.daemon = True
                    if self.random_seed is not None:
                        self.random_seed += 1
                else:
                    thread = threading.Thread(target=data_generator_task)
                self._threads.append(thread)
                thread.start()
        except:
            self.stop()
            raise

    def is_running(self):
        return self._stop_event is not None and not self._stop_event.is_set()

    def stop(self, timeout=None):
        """Stops running threads and wait for them to exit, if necessary.

        Should be called by the same thread which called `start()`.

        # Arguments
            timeout: maximum time to wait on `thread.join()`.
        """
        if self.is_running():
            self._stop_event.set()

        for thread in self._threads:
            if thread.is_alive():
                if self._use_multiprocessing:
                    thread.terminate()
                else:
                    thread.join(timeout)

        if self._use_multiprocessing:
            if self.queue is not None:
                self.queue.close()

        self._threads = []
        self._stop_event = None
        self.queue = None

    def get(self):
        """Creates a generator to extract data from the queue.

        Skip the data if it is `None`.

        # Returns
            A generator
        """
        while self.is_running():
            if not self.queue.empty():
                inputs = self.queue.get()
                if inputs is not None:
                    yield inputs
            else:
                time.sleep(self.wait_time)


def _data_replace(image_path):
    return image_path.replace(os.path.basename(image_path).split('.')[1], 'txt')


def get_data(in_data_path):
    LIST_EXT = ['csv', 'txt']
    IMAGES_EXT = ['jpg', 'png', 'jpeg', 'JPG']

    image_lists, text_lists = [], []
    data_list = [item.strip() for item in in_data_path.split(',') if item.strip()]
    for data_path in data_list:
        data_ext = os.path.splitext(data_path)[-1].replace('.', '')

        if data_ext in LIST_EXT:
            # support csv format input
            with open(data_path, 'rt') as f:
                data = np.array(list(csv.reader(f)))

            if len(data[0]) == 1:
                '''
                image-0.jpg
                image-1.jpg
                '''
                image_list = data
                text_list = np.array(list(map(_data_replace, data)))
            else:
                '''
                image-0.jpg, bbox-0.txt, ...
                image-1.jpg, bbox-0.txt, ...
                '''
                image_list, text_list = data[:,0], data[:,1]

        elif data_ext in IMAGES_EXT:
            image_list, text_list = [data_path], [_data_replace(data_path)]

        else:
            # folder
            image_list = []
            for ext in ['jpg', 'png', 'jpeg', 'JPG']:
                image_list.extend(glob.glob(
                    os.path.join(data_path, '*.{}'.format(ext))))
            text_list = list(map(_data_replace, data))

        image_lists += list(image_list)
        text_lists += list(text_list)

    return np.array(image_lists), np.array(text_lists)


def backbone_converter(backbone):
    if 'res' in backbone:
        if '50' in backbone:
            return 'resnet_v1_50'
        elif '101' in backbone:
            return 'resnet_v1_101'
    raise Exception('Only support resnet_v1_50 and resnet_v1_101. Backbone {} is not supported'.format(backbone))

