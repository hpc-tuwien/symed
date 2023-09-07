import math

import numpy as np
from sklearn.cluster import KMeans
from copy import deepcopy

class SymED(object):
    """
    SymED: Adaptive and Online Symbolic Representation of Data on the Edge
    SymED takes the ABBA (Aggregate Brownian bridge-based approximation of time series, see [1])
    algorithm and extends it to be online, adaptive and distributed. Some parameters and methods
    remain the same as in ABBA. Adapted ABBA methods for SymED contain an '_online' suffix.

    Parameters
    ----------
    tol - float/ list
        Tolerance used during compression and digitization. Accepts either float
        or a list of length two. If float given then same tolerance used for both
        compression and digitization. If list given then first element used for
        compression and second element for digitization.
    scl - float
        Scaling parameter in range 0 to infty. Scales the lengths of the compressed
        representation before performing clustering.
    min_k - int
        Minimum value of k, the number of clusters. If min_k is greater than the
        number of pieces being clustered then each piece will belong to its own
        cluster. Warning given.
    max_k - int
        Maximum value of k, the number of clusters.
    max_len - int
        Maximum length of any segment, prevents issue with growing tolerance for
        flat time series.
    verbose - 0, 1 or 2
        Whether to print details.
        0 - Print nothing
        1 - Print key information
        2 - Print all important information
    seed - True/False
        Determine random number generator for centroid initialization during
        sklearn KMeans algorithm. If True, then randomness is deterministic and
        ABBA produces same representation (with fixed parameters) run by run.
    norm - 1 or 2
        Which norm to use for the compression phase. Also used by digitize_inc,
        a greedy clustering approach.
    c_method - 'kmeans'
        Type of clustering algorithm used, only one supported for now
        'kmeans' - Kmeans clustering used
    alpha - float
        Weighting for the exponential moving average/standard deviation
        Used for online normalization in the online compression step
    exp_mov_avg - float
        Optional initial value for the exponential moving average
        Can be set to help online normalization adapt faster to the data, if prior knowledge is available
    exp_mov_std - float
        Optional initial value for the exponential moving standard deviation
        Can be set to help online normalization adapt faster to the data, if prior knowledge is available

    Raises
    ------
    ValueError: Invalid tol, Invalid scl, Invalid min_k, len(pieces)<min_k.
    References
    ------
    [1] S. Elsworth and S. Güttel. ABBA: Aggregate Brownian bridge-based
    approximation of time series, MIMS Eprint 2019.11
    (http://eprints.maths.manchester.ac.uk/2712/), Manchester
    Institute for Mathematical Sciences, The University of Manchester, UK, 2019.
    """

    def __init__(self, *, tol=0.1, scl=0, min_k=1, max_k=100, max_len=np.inf, verbose=1, seed=True, norm=2,
                 c_method='kmeans', alpha=0.01, exp_mov_avg=None, exp_mov_std=None):
        self.tol = tol
        self.scl = scl
        self.min_k = min_k
        self.max_k = max_k
        self.max_len = max_len
        self.verbose = verbose
        self.seed = seed
        self.norm = norm
        self.c_method = c_method
        
        self.centers = None
        self.labels = None
        self.time_series_segment = np.array([])
        self.pieces = np.empty([0, 3]) # [length, increment, error]
        self.last_inc = 0
        self.last_err = 0
        self.corr = 0.0
        self.exp_mov_avg = exp_mov_avg
        self.exp_mov_std = exp_mov_std if not isinstance(exp_mov_std, type(None)) else 1.0
        self.alpha = alpha

        self._check_parameters()

    def _check_parameters(self):
        self.compression_tol = None
        self.digitization_tol = None

        # Check tol
        if isinstance(self.tol, list) and len(self.tol) == 2:
            self.compression_tol, self.digitization_tol = self.tol
        elif isinstance(self.tol, list) and len(self.tol) == 1:
            self.compression_tol = self.tol[0]
            self.digitization_tol = self.tol[0]
        elif isinstance(self.tol, float):
            self.compression_tol = self.tol
            self.digitization_tol = self.tol
        else:
            raise ValueError('Invalid tol.')

        # Check scl (scaling parameter)
        if self.scl < 0:
            raise ValueError('Invalid scl.')

        # Check min_k and max_k
        if self.min_k > self.max_k:
            raise ValueError('Invalid limits: min_k must be less than or equal to max_k')

        # Check verbose
        if self.verbose not in [0, 1, 2]:  # pragma: no cover
            self.verbose == 1  # set to default
            print('Invalid verbose, setting to default')

        # Check norm
        if self.norm not in [1, 2]:
            raise NotImplementedError('norm = 1 or norm = 2')

        # Check ordered
        if self.c_method not in ['kmeans']:
            raise ValueError('Invalid c_method.')

        # Check alpha (normalization weight)
        if self.alpha < 0:
            raise ValueError('Invalid alpha')

    def inverse_transform(self, string, centers, start=0):
        """
        Convert ABBA symbolic representation back to numeric time series representation.
        Parameters
        ----------
        string - string
            Time series in symbolic representation using unicode characters starting
            with character 'a'.
        centers - numpy array
            Centers of clusters from clustering algorithm. Each center corresponds
            to character in string.
        start - float
            First element of original time series. Applies vertical shift in
            reconstruction. If not specified, the default is 0.
        Returns
        -------
        times_series - list
            Reconstruction of the time series.
        """

        pieces = self.inverse_digitize(string, centers)
        pieces = self.quantize(pieces)
        time_series = self.inverse_compress(start, pieces)
        return time_series

    def compress_online(self, data_point):
        """
        Approximate a time series using a continuous piecewise linear function, one piece at a time.
        Parameters
        ----------
        data_point - float
            Next online data point to compress
        Returns
        -------
        piece - numpy array
            1D Numpy array with three entries of length, increment error for the segment.
            If no linear piece was compressed yet, returns None.
        """

        self.time_series_segment = np.append(self.time_series_segment, data_point)

        # update moving avg and std
        if isinstance(self.exp_mov_avg, type(None)):
            self.exp_mov_avg = data_point
            delta = data_point
        else:
            delta = data_point - self.exp_mov_avg
            self.exp_mov_avg = self.exp_mov_avg + self.alpha * delta
        self.exp_mov_std = math.sqrt((1 - self.alpha) * (self.exp_mov_std**2 + self.alpha * delta**2))
        self.exp_mov_std = self.exp_mov_std if self.exp_mov_std > np.finfo(float).eps else 1.0

        end = len(self.time_series_segment) - 1 # end point

        # cannot compress 1 data point, wait for next one
        if end == 0:
            return None

        if self.norm == 2:
            tol = self.compression_tol ** 2
        else:
            tol = self.compression_tol
        x = np.arange(0, len(self.time_series_segment))
        epsilon = np.finfo(float).eps

        # get normalized version of ts before compression
        time_series_segment_normed = deepcopy(self.time_series_segment)
        time_series_segment_normed = np.divide(time_series_segment_normed - self.exp_mov_avg, self.exp_mov_std)
        inc = self.time_series_segment[end] - self.time_series_segment[0]
        inc_normed = time_series_segment_normed[end] - time_series_segment_normed[0]

        # error function for linear piece
        if self.norm == 2:
            err = np.linalg.norm((time_series_segment_normed[0] + (inc_normed / end) * x[0:end + 1]) - time_series_segment_normed[0:end + 1]) ** 2
        else:
            err = np.linalg.norm(
                (time_series_segment_normed[0] + (inc_normed / end) * x[0:end + 1]) - time_series_segment_normed[0:end + 1], 1)

        if self.verbose == 2:  # pragma: no cover
            print('Online compression: Error is', err)

        if (err <= tol * (end - 1) + epsilon) and end <= self.max_len:
            # epsilon added to prevent error when err ~ 0 and (end -1) = 0
            self.last_inc = inc
            self.last_err = err
            piece = None
        else:
            piece = np.array([end - 1, self.last_inc, self.last_err])
            self.pieces = np.vstack([self.pieces, piece])
            self.time_series_segment = np.array(self.time_series_segment[end-1:])
            self.last_inc = self.time_series_segment[-1] - self.time_series_segment[-2]
            self.last_err = 0.0

            if self.verbose == 2:  # pragma: no cover
                print('Online compression: Compressed segment with length', piece[0], 'and increment', piece[1])

        return piece

    def flush_compress_online(self):
        """
        Finish online compression by compressing the last segment using all remaining uncompressed data points.
        Returns
        -------
        piece - numpy array
            1D Numpy array with three entries of length, increment error for the segment.
            If no more linear piece can be compressed, returns None.
        """
        end = len(self.time_series_segment) - 1
        if end == 0:
            return None
        piece = np.array([end, self.last_inc, self.last_err])
        self.pieces = np.vstack([self.pieces, piece])
        self.time_series_segment = np.array(self.time_series_segment[-1:])
        return piece

    def inverse_compress(self, start, pieces):
        """
        Reconstruct time series from its first value `ts0` and its `pieces`.
        `pieces` must have (at least) two columns, incremenent and window width, resp.
        A window width w means that the piece ranges from s to s+w.
        In particular, a window width of 1 is allowed.
        Parameters
        ----------
        start - float
            First element of original time series. Applies vertical shift in
            reconstruction.
        pieces - numpy array
            Numpy array with three columns, each row contains increment, length,
            error for the segment. Only the first two columns are required.

        Returns
        -------
        time_series : Reconstructed time series
        """
        time_series = [start]
        # stitch linear piece onto last
        for j in range(0, len(pieces)):
            x = np.arange(0,pieces[j,0]+1)/(pieces[j,0])*pieces[j,1]
            y = time_series[-1] + x
            time_series = time_series + y[1:].tolist()
        return time_series

    def inverse_compress_online(self, start, length, inc):
        """
        Reconstruct a single linear piece of (length, inc) to a time series segment.
        Parameters
        ----------
        start - float
            First element of original time series. Applies vertical shift in
            reconstruction.
        length - float
            Length of time series segment to be reconstructed.
        inc - float
            Increment of time series segment to be reconstructed.

        Returns
        -------
        time_series : Reconstructed time series segment
        """
        time_series = [start]
        x = np.arange(0, length + 1) / length * inc
        y = time_series[-1] + x
        time_series = time_series + y[1:].tolist()
        return time_series

    def _max_cluster_var(self, pieces, labels, centers, k):
        """
        Calculate the maximum variance among all clusters after k-means, in both
        the inc and len dimension.
        Parameters
        ----------
        pieces - numpy array
            One or both columns from compression. See compression.
        labels - list
            List of ints corresponding to cluster labels from k-means.
        centers - numpy array
            centers of clusters from clustering algorithm. Each center corresponds
            to character in string.
        k - int
            Number of clusters. Corresponds to numberof rows in centers, and number
            of unique symbols in labels.
        Returns
        -------
        variance - float
            Largest variance among clusters from k-means.
        """
        d1 = [0]  # direction 1
        d2 = [0]  # direction 2
        for i in range(k):
            matrix = ((pieces[np.where(labels == i), :] - centers[i])[0]).T
            # Check not all zero
            if not np.all(np.abs(matrix[0, :]) < np.finfo(float).eps):
                # Check more than one value
                if len(matrix[0, :]) > 1:
                    d1.append(np.var(matrix[0, :]))

            # If performing 2-d clustering
            if matrix.shape[0] == 2:
                # Check not all zero
                if not np.all(np.abs(matrix[1, :]) < np.finfo(float).eps):
                    # Check more than one value
                    if len(matrix[1, :]) > 1:
                        d2.append(np.var(matrix[1, :]))
        return np.max(d1), np.max(d2)

    def _build_centers(self, pieces, labels, c1, k, col):
        """
        utility function for digitize, helps build 2d cluster centers after 1d clustering.
        Parameters
        ----------
        pieces - numpy array
            Time series in compressed format. See compression.
        labels - list
            List of ints corresponding to cluster labels from k-means.
        c1 - numpy array
            1d cluster centers
        k - int
            Number of clusters
        col - 0 or 1
            Which column was clustered during 1d clustering
        Returns
        -------
        centers - numpy array
            centers of clusters from clustering algorithm. Each centre corresponds
            to character in string.
        """
        c2 = []
        for i in range(k):
            location = np.where(labels == i)[0]
            if location.size == 0:
                c2.append(np.NaN)
            else:
                c2.append(np.mean(pieces[location, col]))
        if col == 0:
            return (np.array((c2, c1))).T
        else:
            return (np.array((c1, c2))).T

    def digitize_online(self, pieces):
        """
        Convert compressed representation to symbolic representation using clustering.
        Is called repeatedly, once for each new piece. Keeps track of cluster centers
        and symbols in self.centers and self.symbols.
        Parameters
        ----------
        pieces - numpy array
            Time series in compressed format. All linear pieces found so far.
        Returns
        -------
        self.string - string
            Time series in symbolic representation using unicode characters starting
            with character 'a'.
        new_centers - numpy array
            Centers of clusters from clustering algorithm. Each centre corresponds
            to character in string. Updated after each call to digitize_online.
        new_symbol_found - bool
            Flag for additional information. True if new symbol was found, False otherwise.
        old_symbols_changed - bool
            Flag for additional information. True if old symbols were changed, False otherwise.
            Since online centers are updated after each call, pieces may be re-assigned to
            different clusters and get a different symbol. Usually only the case at the beginning
            when there are not many pieces and the clustering is still small.
        """
        # Construct deep copy and scale data
        data = deepcopy(pieces[:, 0:2])

        if isinstance(self.centers, type(None)) or len(self.centers) < self.min_k:
            new_centers = data
            new_labels = np.arange(len(new_centers))
        else:
            if self.c_method == 'kmeans':
                if self.scl == np.inf or self.scl == 0:
                    new_labels, new_centers = self.digitize_1D_kmeans_online(data)
                else:
                    new_labels, new_centers = self.digitize_2D_kmeans_online(data)
            else:
                raise NotImplementedError("method not implemented")

        old_centers = self.centers
        old_labels = self.labels if not isinstance(self.labels, type(None)) else []
        k_new = len(new_centers)
        k_old = len(old_centers) if not isinstance(old_centers, type(None)) else 0
        new_symbol_found = False
        old_symbols_changed = False

        # check for significant changes in clustering
        if k_new > k_old:
            if self.verbose in [1, 2]:  # pragma: no cover
                print("New Symbol found")
            new_symbol_found = True
        if k_new < k_old or np.any(old_labels != new_labels[:len(old_labels)]):
            if self.verbose in [1, 2]:  # pragma: no cover
                print("Old symbol(s) changed")
            old_symbols_changed = True

        # Convert labels to string
        self.string = ''.join([chr(97 + j) for j in new_labels])

        self.centers = new_centers
        self.labels = new_labels

        return self.string, new_centers, new_symbol_found, old_symbols_changed

    def digitize_2D_kmeans_online(self, data):
        """
         Cluster data (pieces) by length and increment using streaming based 2D k-means clustering.
         Is called consecutively for each new piece added to data. Previous centers are used as initial centers.
         ----------
         data - numpy array
             Time series in compressed format. All linear pieces found so far.
         Returns
         -------
         labels - numpy array
             List of ints corresponding to cluster labels from k-means.
         centers - numpy array
             Centers of clusters from clustering algorithm. Each centre corresponds
             to character in string. Updated after each call to digitize_online.
        """
        # Initialise variables
        centers = np.zeros((0, 2))
        labels = [-1] * np.shape(data)[0]
        has_centers = not isinstance(self.centers, type(None))
        ########################################################################
        #     scl in (0, inf)
        ########################################################################
        # construct tol_s
        s = .20
        N = 1
        for i in data:
            N += i[0]
        bound = ((6 * (N - len(data))) / (N * len(data))) * ((self.digitization_tol * self.digitization_tol) / (s * s))

        # scale length to unit variance
        self.len_std = np.std(data[:, 0])
        self.len_std = self.len_std if self.len_std > np.finfo(float).eps else 1
        data[:, 0] /= self.len_std

        # scale inc to unit variance
        self.inc_std = np.std(data[:, 1])
        self.inc_std = self.inc_std if self.inc_std > np.finfo(float).eps else 1
        data[:, 1] /= self.inc_std

        data[:, 0] *= self.scl  # scale lengths accordingly

        if has_centers:
            init_centers = deepcopy(self.centers)
            init_centers[:, 0] /= self.len_std
            init_centers[:, 0] *= self.scl  # scaling
            init_centers[:, 1] /= self.inc_std
        else:
            init_centers = 'k-means++'

        # Run through values of k from min_k to max_k checking bound
        if self.digitization_tol != 0:
            error = np.inf
            k = len(init_centers) - 1 if has_centers else self.min_k - 1

            while k < self.max_k and k < len(data) and (error > bound):
                # tol=0 ensures labels and centres coincide
                k += 1
                if has_centers:
                    if k == len(self.centers) + 1:
                        init_centers = np.vstack([init_centers, data[len(init_centers), 0:2]])
                    elif k > len(self.centers) + 1:
                        if self.verbose in [1, 2]:  # pragma: no cover
                            print("Start clustering from scratch")
                        init_centers = 'k-means++'
                        has_centers = False
                if self.seed:
                    kmeans = KMeans(init=init_centers, n_clusters=k, tol=0, random_state=0, n_init=1 if has_centers else 10).fit(data)
                else:
                    kmeans = KMeans(init=init_centers, n_clusters=k, tol=0, n_init=1 if has_centers else 10).fit(data)

                centers = kmeans.cluster_centers_
                labels = kmeans.labels_
                error_1, error_2 = self._max_cluster_var(data, labels, centers, k)
                error = max([error_1, error_2])

                if self.verbose == 2:  # pragma: no cover
                    print('k:', k)
                    print('d1_error:', error_1, 'd2_error:', error_2, 'bound:', bound)

            if self.verbose == 2:  # pragma: no cover
                print('Digitization: Using', k, 'symbols')
                if has_centers and len(self.centers) != k:
                    print(f"Clusters got increased from {len(self.centers)} to {k}")

        # Zero error so cluster with largest possible k.
        else:
            unique_points = np.array(list(set(tuple((round(p[0],4), round(p[1], 4))) for p in data)))
            if len(unique_points) <= self.max_k:
                k = len(unique_points)
                init_centers = deepcopy(unique_points)
            else:
                k = self.max_k
                init_centers = deepcopy(self.centers)
            init_centers[:, 0] /= self.len_std
            init_centers[:, 0] *= self.scl  # scaling
            init_centers[:, 1] /= self.inc_std

            # tol=0 ensures labels and centres coincide
            kmeans = KMeans(init=init_centers, n_clusters=k, tol=0, n_init=1 if has_centers else 10).fit(data)
            centers = kmeans.cluster_centers_
            labels = kmeans.labels_
            error = self._max_cluster_var(data, labels, centers, k)
            if self.verbose == 2:  # pragma: no cover
                print('Digitization: Using', k, 'symbols')
                if has_centers and len(self.centers) != k:
                    print(f"Clusters got increased from {len(self.centers)} to {k}")
        # build cluster centers
        centers[:, 0] *= self.len_std
        centers[:, 1] *= self.inc_std
        centers[:, 0] /= self.scl  # reverse scaling
        return labels, centers

    def digitize_1D_kmeans_online(self, data):
        """
         Cluster data (pieces) by length and increment using streaming based 1D k-means clustering,
         either by length or increment, depending on self.scl.
         Is called consecutively for each new piece added to data. Previous centers are used as initial centers.
         ----------
         data - numpy array
             Time series in compressed format. All linear pieces found so far.
         Returns
         -------
         labels - numpy array
             List of ints corresponding to cluster labels from k-means.
         centers - numpy array
             Centers of clusters from clustering algorithm. Each centre corresponds
             to character in string. Updated after each call to digitize_online.
        """
        # Initialise variables
        centers = np.zeros((0, 2))
        labels = [-1] * np.shape(data)[0]
        has_centers = not isinstance(self.centers, type(None))
        ########################################################################
        #     scl == 0
        ########################################################################
        if self.scl == 0:
            # construct tol_s
            s = .20
            N = 1
            for i in data:
                N += i[0]
            bound = ((6 * (N - len(data))) / (N * len(data))) * ((self.digitization_tol * self.digitization_tol) / (s * s))

            # scale inc to unit variance
            self.inc_std = np.std(data[:, 1])
            self.inc_std = self.inc_std if self.inc_std > np.finfo(float).eps else 1
            data[:, 1] /= self.inc_std

            if has_centers:
                init_centers = deepcopy(self.centers)
                init_centers[:, 1] /= self.inc_std
            else:
                init_centers = 'k-means++'

            # Run through values of k from min_k to max_k checking bound
            if self.digitization_tol != 0:
                error = np.inf
                k = len(init_centers) - 1 if has_centers else self.min_k - 1

                while k < self.max_k and k < len(data) and (error > bound):
                    # tol=0 ensures labels and centres coincide
                    k += 1
                    if has_centers:
                        if k == len(self.centers) + 1:
                            init_centers = np.vstack([init_centers, data[len(init_centers), 0:2]])
                        elif k > len(self.centers) + 1:
                            if self.verbose in [1, 2]:  # pragma: no cover
                                print("Start clustering from scratch")
                            init_centers = 'k-means++'
                            has_centers = False
                    if self.seed:
                        kmeans = KMeans(init=init_centers[:,1].reshape(-1,1) if not isinstance(init_centers, str) else init_centers, n_clusters=k, tol=0, random_state=0, n_init=1 if has_centers else 10).fit(data[:,1].reshape(-1,1))
                    else:
                        kmeans = KMeans(init=init_centers[:,1].reshape(-1,1) if not isinstance(init_centers, str) else init_centers, n_clusters=k, tol=0, n_init=1 if has_centers else 10).fit(data[:,1].reshape(-1,1))

                    centers = kmeans.cluster_centers_
                    labels = kmeans.labels_
                    error_1, error_2 = self._max_cluster_var(data[:,1].reshape(-1,1), labels, centers, k)
                    error = max([error_1, error_2])

                    if self.verbose == 2:  # pragma: no cover
                        print('k:', k)
                        print('d1_error:', error_1, 'd2_error:', error_2, 'bound:', bound)

                if self.verbose == 2:  # pragma: no cover
                    print('Digitization: Using', k, 'symbols')
                    if has_centers and len(self.centers) != k:
                        print(f"Clusters got increased from {len(self.centers)} to {k}")

            # Zero error so cluster with largest possible k.
            else:
                unique_points = np.array(list(set(tuple((round(p[0],4), round(p[1], 4))) for p in data)))
                if len(unique_points) <= self.max_k:
                    k = len(unique_points)
                    init_centers = deepcopy(unique_points)
                else:
                    k = self.max_k
                    init_centers = deepcopy(self.centers)
                init_centers[:, 1] /= self.inc_std

                # tol=0 ensures labels and centres coincide
                kmeans = KMeans(init=init_centers[:,1].reshape(-1,1), n_clusters=k, tol=0, n_init=1 if has_centers else 10).fit(data[:,1].reshape(-1,1))
                centers = kmeans.cluster_centers_
                labels = kmeans.labels_
                error = self._max_cluster_var(data[:,1].reshape(-1,1), labels, centers, k)
                if self.verbose == 2:  # pragma: no cover
                    print('Digitization: Using', k, 'symbols')
                    if has_centers and len(self.centers) != k:
                        print(f"Clusters got increased from {len(self.centers)} to {k}")
            c = centers.reshape(1, -1)[0]
            c *= self.inc_std
            centers = self._build_centers(data, labels, c, k, 0)

        ########################################################################
        #     scl == inf
        ########################################################################
        elif self.scl == np.inf:
            # construct tol_s
            s = .20
            N = 1
            for i in data:
                N += i[0]
            bound = ((6 * (N - len(data))) / (N * len(data))) * (
                        (self.digitization_tol * self.digitization_tol) / (s * s))

            # scale length to unit variance
            self.len_std = np.std(data[:, 0])
            self.len_std = self.len_std if self.len_std > np.finfo(float).eps else 1
            data[:, 0] /= self.len_std

            if has_centers:
                init_centers = deepcopy(self.centers)
                init_centers[:, 0] /= self.len_std
            else:
                init_centers = 'k-means++'

            # Run through values of k from min_k to max_k checking bound
            if self.digitization_tol != 0:
                error = np.inf
                k = len(init_centers) - 1 if has_centers else self.min_k - 1

                while k < self.max_k and k < len(data) and (error > bound):
                    # tol=0 ensures labels and centres coincide
                    k += 1
                    if has_centers:
                        if k == len(self.centers) + 1:
                            init_centers = np.vstack([init_centers, data[len(init_centers), 0:2]])
                        elif k > len(self.centers) + 1:
                            if self.verbose in [1, 2]:  # pragma: no cover
                                print("Start clustering from scratch")
                            init_centers = 'k-means++'
                            has_centers = False
                    if self.seed:
                        kmeans = KMeans(init=init_centers[:,0].reshape(-1,1) if not isinstance(init_centers, str) else init_centers, n_clusters=k, tol=0, random_state=0,
                                        n_init=1 if has_centers else 10).fit(data[:, 0].reshape(-1, 1))
                    else:
                        kmeans = KMeans(init=init_centers[:,0].reshape(-1,1) if not isinstance(init_centers, str) else init_centers, n_clusters=k, tol=0,
                                        n_init=1 if has_centers else 10).fit(data[:, 0].reshape(-1, 1))

                    centers = kmeans.cluster_centers_
                    labels = kmeans.labels_
                    error_1, error_2 = self._max_cluster_var(data[:, 0].reshape(-1, 1), labels, centers, k)
                    error = max([error_1, error_2])

                    if self.verbose == 2:  # pragma: no cover
                        print('k:', k)
                        print('d1_error:', error_1, 'd2_error:', error_2, 'bound:', bound)

                if self.verbose == 2:  # pragma: no cover
                    print('Digitization: Using', k, 'symbols')
                    if has_centers and len(self.centers) != k:
                        print(f"Clusters got increased from {len(self.centers)} to {k}")

            # Zero error so cluster with largest possible k.
            else:
                unique_points = np.array(list(set(tuple((round(p[0], 4), round(p[1], 4))) for p in data)))
                if len(unique_points) <= self.max_k:
                    k = len(unique_points)
                    init_centers = deepcopy(unique_points)
                else:
                    k = self.max_k
                    init_centers = deepcopy(self.centers)
                init_centers[:, 0] /= self.len_std

                # tol=0 ensures labels and centres coincide
                kmeans = KMeans(init=init_centers[:,0].reshape(-1,1), n_clusters=k, tol=0, n_init=1 if has_centers else 10).fit(data[:,0].reshape(-1,1))
                centers = kmeans.cluster_centers_
                labels = kmeans.labels_
                error = self._max_cluster_var(data[:,0].reshape(-1,1), labels, centers, k)
                if self.verbose == 2:  # pragma: no cover
                    print('Digitization: Using', k, 'symbols')
                    if has_centers and len(self.centers) != k:
                        print(f"Clusters got increased from {len(self.centers)} to {k}")

            c = centers.reshape(1, -1)[0]
            c *= self.len_std
            centers = self._build_centers(data, labels, c, k, 1)

        return labels, centers

    def inverse_digitize(self, string, centers):
        """
        Convert symbolic representation back to compressed representation for reconstruction.
        Parameters
        ----------
        string - string
            Time series in symbolic representation using unicode characters starting
            with character 'a'.
        centers - numpy array
            centers of clusters from clustering algorithm. Each centre corresponds
            to character in string.
        Returns
        -------
        pieces - np.array
            Time series in compressed format. See compression.
        """
        pieces = np.empty([0, 2])
        for p in string:
            pc = centers[ord(p) - 97, :]
            pieces = np.vstack([pieces, pc])
        return pieces

    def inverse_digitize_online(self, symbol, centers):
        """
        Convert symbolic representation back to compressed representation for reconstruction
        in an online fashion for one symbol.
        Parameters
        ----------
        symbol - string
            Time series segment in symbolic representation using a unicode character
        centers - numpy array
            centers of clusters from clustering algorithm. Each centre corresponds
            to character in string.
        Returns
        -------
        pieces - np.array
            Time series in compressed format (linear pieces).
        """
        return deepcopy(centers[ord(symbol) - 97, :])

    def quantize(self, pieces):
        """
        Realign window lengths with integer grid.
        Parameters
        ----------
        pieces: Time series in compressed representation.
        Returns
        -------
        pieces: Time series in compressed representation with window length adjusted to integer grid.
        """
        if len(pieces) == 1:
            pieces[0, 0] = round(pieces[0, 0])
        else:
            for p in range(len(pieces) - 1):
                corr = round(pieces[p, 0]) - pieces[p, 0]
                pieces[p, 0] = round(pieces[p, 0] + corr)
                pieces[p + 1, 0] = pieces[p + 1, 0] - corr
                if pieces[p, 0] == 0:
                    pieces[p, 0] = 1
                    pieces[p + 1, 0] -= 1
            pieces[-1, 0] = round(pieces[-1, 0])
        return pieces
