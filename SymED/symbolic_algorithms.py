import numpy as np
from timeit import default_timer as timer

import sys
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.join(script_dir, '..')
sys.path.append(parent_dir)
import SymED
from ABBA import ABBA
import threading
import queue


class Result:
    def __init__(self, conversion, reconstruction, compression_rate_length, compression_rate_bytes, mse, times_combined,
                 times_sender, times_receiver):
        """
         Result class for symbolic conversions, contains all relevant information for evaluation.
         Parameters
         ----------
            conversion - string
                string of symbols
            reconstruction - numpy array
                reconstructed time series
            compression_rate_length - float
                dimension reduction rate in terms of length
            compression_rate_bytes - float
                compression rate in terms of bytes
            mse - float
                mean squared error between original and reconstructed time series
            times_combined - list of floats
                time spent in sender + receiver for each symbol
            times_sender - list of floats
                time spent in sender for each symbol
            times_receiver - list of floats
                time spent in receiver for each symbol
         """
        self.conversion = conversion
        self.reconstruction = reconstruction
        self.compression_rate_length = compression_rate_length
        self.compression_rate_bytes = compression_rate_bytes
        self.mse = mse
        self.times_combined = times_combined
        self.times_sender = times_sender
        self.times_receiver = times_receiver

class SymED_online_result:
    def __init__(self, i, string, centers, pieces):
        """
         Result class for SymED online conversion
         Mainly needed for evaluation purposes, not needed in a production environment to that extend
         Parameters
         ----------
            i - int
                iteration counter of the time series serving as a timestamp replacement for performance evaluation
            string - string
                string of symbols converted so far
            centers - numpy array
                centers of the clusters for reconstructing the converted symbols
            pieces - numpy array
                linear pieces of the time series
         """
        self.i = i
        self.string = string
        self.centers = centers
        self.pieces = pieces


def abba(ts, tol, verbose=0, c_method="kmeans", scl=1.0, min_k=3):
    """
    Calling original ABBA algorithm
    Parameters
    ----------
    ts - list
        time series to be converted to symbols 1D list of floats
    tol - float
        tolerance parameter determining the allowed deviation from the original time series during conversion
    verbose - int
        verbosity level
    c_method - str
        clustering method
    scl - float
        scaling parameter for weighing lengths during clustering
    min_k - int
        minimum number of clusters to be created
    Returns
    -------
    Result() - object
        object containing the results of the ABBA algorithm
    abba - object
        object containing the ABBA algorithm instance used
    centers - numpy array
        centers of the clusters for reconstructing the converted symbols (contained in Result())
    pieces - numpy array
        linear pieces of the time series
    """
    time_start = timer()
    ts_mean = np.mean(ts)
    ts_std = np.std(ts)
    ts_unit = (ts - ts_mean) / ts_std
    abba = ABBA.ABBA(tol=tol, verbose=verbose, scl=scl, c_method=c_method, min_k=min_k)

    pieces = abba.compress(ts_unit)
    string, centers = abba.digitize(pieces)

    time_sender = timer() - time_start
    compression_rate_length = round(len(string) / len(ts_unit), 3)
    # each center has 2 float values
    reconstruction_bytes = len(centers) * 4 * 2
    compression_rate_bytes = round((len(string) + reconstruction_bytes) / (len(ts_unit) * 4), 3)
    time_start = timer()
    reconstructed_abba = np.multiply(abba.inverse_transform(string, centers, ts_unit[0]), ts_std) + ts_mean
    time_receiver = timer() - time_start
    mse_abba = round(sum((ts - reconstructed_abba) ** 2) / len(ts), 2)
    return Result(conversion=string, reconstruction=reconstructed_abba, compression_rate_length=compression_rate_length,
                  compression_rate_bytes=compression_rate_bytes, mse=mse_abba,
                  times_combined=[time_sender + time_receiver],
                  times_sender=[time_sender], times_receiver=[time_receiver]), \
           abba, centers, pieces


def symed(ts, tol, min_k=3, alpha=0.01, verbose=1, c_method="kmeans", scl=1.0, ewma=None, ewmv=None,
          online_result_queue=None):
    """
    Converts a time series to symbols using the SymED online algorithm.
    This method is for evaluation purposes only. It spawns a sender and a receiver, each running in a separate thread.
    Sender transmits compressed data to receiver via queues. The receiver converts data to symbols and also does
    reconstruction. The evaluation results are returned after both threads have finished.
    Parameters
    ----------
    ts - list
        time series to be converted to symbols 1D list of floats
    tol - float
        tolerance parameter determining the allowed deviation from the original time series during conversion
    min_k - int
        minimum number of clusters to be created
    alpha - float
        online normalization weighting parameter for the exponential moving average and standard deviation
    verbose - int
        verbosity level
    c_method - str
        clustering method, currently only 'kmeans' is supported
    scl - float
        scaling parameter for weighing lengths during clustering
    ewma - float
        Optional initial value for the exponential moving average
    ewmv - float
        Optional initial value for the exponential moving standard deviation
    online_result_queue - queue
        Queue for sending the results of the online conversion to the evaluation thread
        Mainly here for evaluation purposes.

    Returns
    -------
    Result() - object
        Object containing the results of the SymED algorithm
    symed - object
        Object containing the SymED algorithm instance used
        Needed for evaluation purposes
    """
    sender_to_receiver_queue = queue.Queue()
    sender_times_queue = queue.Queue()
    receiver_result_queue = queue.Queue()
    iteration_queue = queue.Queue()

    symed_send = SymED.SymED(scl=scl, tol=tol, alpha=alpha, verbose=verbose, c_method=c_method, min_k=min_k,
                             exp_mov_avg=ewma, exp_mov_std=ewmv)

    symed_recv = SymED.SymED(scl=scl, tol=tol, alpha=alpha, verbose=verbose, c_method=c_method, min_k=min_k,
                             exp_mov_avg=ewma, exp_mov_std=ewmv)

    thread_sender = threading.Thread(target=symed_sender, args=(
    ts, symed_send, sender_to_receiver_queue, sender_times_queue, iteration_queue))
    thread_sender.start()
    thread_receiver = threading.Thread(target=symed_receiver, args=(
    ts, symed_recv, sender_to_receiver_queue, sender_times_queue, receiver_result_queue, iteration_queue, verbose,
    online_result_queue))
    thread_receiver.start()

    thread_sender.join()
    thread_receiver.join()

    return receiver_result_queue.get() + (symed_send,)


def symed_sender(ts, symed_send, sender_to_receiver_queue, times_queue, iteration_queue):
    """
    Sender part of the SymED online algorithm. Compresses the time series and sends the compressed data to the receiver.
    A queue is used for simple communication between sender and receiver, to do evaluation independent of any network or protocol.
    For use in a real-world scenario the queue (sender_to_receiver_queue) would typically be replaced by a network connection.
    Other queues mainly serve evaluation purposes and may not be needed otherwise.
    ----------
    ts - list
        time series to be converted to symbols 1D list of floats
    symed_send - object
        SymED algorithm instance used for compression
    sender_to_receiver_queue - queue
        Queue for sending the compressed data to the receiver
    times_queue - queue
        Queue for sending the times of the compression to the receiver
        Only needed for evaluation purposes, so  hat the receiver can construct the Result object including all necessary data.
    iteration_queue - queue
        Queue for sending the iteration number of the sender to the receiver, only needed for evaluation.
        Is used as a timestamp replacement, so that the receiver knows how many iterations have been processed.
        The difference between two successive iterations in the queue is representative of the time passed.
        In a real-world scenario the receiver would just measure the time between two received messages instead.
    """
    times_sender = []
    lengths_summed = 0

    time_start = timer()
    # -1 as additional entry for flush at the end, doesn't get processed
    t_prev = None
    for i, t in enumerate(np.append(ts, -1)):

        flush = i == len(ts)
        piece = symed_send.compress_online(t) if not flush else symed_send.flush_compress_online()

        if not isinstance(piece, type(None)):
            lengths_summed += piece[0]
            sender_to_receiver_queue.put(t_prev)
            iteration_queue.put(i)
            time_sender = timer() - time_start
            times_sender.append(time_sender)
            time_start = timer()

        t_prev = t

    sender_to_receiver_queue.put(None)
    iteration_queue.put(None)
    times_queue.put(times_sender)


def symed_receiver(ts, symed_recv, sender_to_receiver_queue, sender_times_queue, result_queue, iteration_queue, verbose,
                   online_result_queue=None):
    """
    Receiver part of the SymED online algorithm. Converts compressed data to symbols and reconstructs the time series.
    A queue is used for simple communication between sender and receiver, to do evaluation independent of any network or protocol.
    For use in a real-world scenario the queue (sender_to_receiver_queue) would typically be replaced by a network connection.
    Other queues mainly serve evaluation purposes and may not be needed otherwise.
    Parameters
    ----------
    ts - list
        time series to be converted to symbols 1D list of floats
    symed_recv- object
        SymED algorithm instance used for symbolic conversion
    sender_to_receiver_queue - queue
        Queue for receiving the compressed data from the sender. Is used instead of a network connection for simple evaluation.
    sender_times_queue - queue
        Queue for receiving the compression runtimes from the sender, only needed for evaluation.
    result_queue - queue
        Queue for sending the result of the whole conversion to the calling thread
    iteration_queue - queue
        Queue for receiving the iteration number of the sender, only needed for evaluation.
        Is used as a timestamp replacement, so that the receiver knows how many iterations have been processed.
        The difference between two successive iterations in the queue is representative of the time passed.
        In a real-world scenario the receiver would just measure the time between two received messages instead.
    verbose - int
        Verbosity level of the SymED algorithm [0, 1 ,2]
    online_result_queue - queue
        Queue for sending the results of the online conversion to the evaluation thread.
        Mainly here for evaluation purposes, so that metrics can be gathered during the online conversion.
    """
    ts_reconstructed_online = [ts[0]]
    string = ""
    times_receiver = []
    pieces = np.empty([0, 2])

    while True:
        t = sender_to_receiver_queue.get()
        # i is the iteration counter of the time series serving as a timestamp replacement, as this simulation does not run in real-time.
        i = iteration_queue.get()
        time_start = timer()
        if not isinstance(t, type(None)):
            # in a real-time scenario the amount of passed seconds between to messages from the sender would be used instead of i
            time_since_last_update = i - (len(ts_reconstructed_online) - 1)
            segment_length = time_since_last_update - 1  # -1 because latest datapoint is kept for next piece
            segment_inc = t - ts_reconstructed_online[-1]
            piece = (segment_length, segment_inc)

            pieces = np.vstack([pieces, piece])
            string, centers, symbol_found, symbols_changed = symed_recv.digitize_online(pieces)

            ts_reconstructed_online += symed_recv.inverse_compress_online(ts_reconstructed_online[-1], piece[0],
                                                                          piece[1])[1:]

            if verbose in [1, 2]:  # pragma: no cover
                print("Symbols until second %3d:" % i, string)
            time_receiver = timer() - time_start
            times_receiver.append(time_receiver)
            if not isinstance(online_result_queue, type(None)):
                online_result_queue.put(SymED_online_result(i, string, centers, pieces))

        else:
            if not isinstance(online_result_queue, type(None)):
                online_result_queue.put(None)
            times_sender = sender_times_queue.get()
            times_combined = [sum(tup) for tup in zip(times_sender, times_receiver)]

            reconstruction_bytes = len(pieces) * 4
            compression_rate_length = round(len(string) / len(ts), 3)
            compression_rate_bytes = round(reconstruction_bytes / (len(ts) * 4), 3)

            time_start = timer()
            ts_reconstructed_offline = symed_recv.inverse_transform(string, centers, ts[0])

            times_receiver_offline = [timer() - time_start]

            ts_reconstructed_online = np.array(ts_reconstructed_online)
            ts_reconstructed_offline = np.array(ts_reconstructed_offline)
            mse_abba_online = round(sum((ts - ts_reconstructed_online) ** 2) / len(ts), 2)
            mse_abba_historical = round(sum((ts - ts_reconstructed_offline) ** 2) / len(ts), 2)

            result_online = Result(conversion=string, reconstruction=ts_reconstructed_online,
                                   compression_rate_length=compression_rate_length,
                                   compression_rate_bytes=compression_rate_bytes, mse=mse_abba_online,
                                   times_combined=times_combined,
                                   times_sender=times_sender, times_receiver=times_receiver)
            result_offline = Result(conversion=string, reconstruction=ts_reconstructed_offline,
                                    compression_rate_length=compression_rate_length,
                                    compression_rate_bytes=compression_rate_bytes, mse=mse_abba_historical,
                                    times_combined=times_combined,
                                    times_sender=times_sender, times_receiver=times_receiver_offline)
            result_queue.put((result_online, result_offline))
            break
