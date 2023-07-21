import unittest
from tabensemb.utils import set_random_seed
import numpy as np
import torch
from src.model._transformer.clustering.common.gmm import GMM, Cluster, TwolayerGMM
from src.model._transformer.clustering.common.bmm import BMM


def get_data(n_clusters, n_features, n_each_cluster, seed):
    datas = []
    set_random_seed(seed)
    for i in range(n_clusters):
        v = np.eye(n_features) + np.random.rand(n_features, n_features)
        r = np.random.multivariate_normal(
            np.random.rand(n_features) * 20,
            v * v.T,
            n_each_cluster,
        )
        datas.append(r)

    data = np.concatenate(datas, axis=0)
    return data


def plot_gaussian(mean, var, ax):
    cov_inv = np.linalg.inv(var)
    cov_det = np.linalg.det(var)
    x = np.linspace(
        mean[0] - np.sqrt(var[0, 0]) * 3, mean[0] + np.sqrt(var[0, 0]) * 3, 100
    )
    y = np.linspace(
        mean[1] - np.sqrt(var[1, 1]) * 3, mean[1] + np.sqrt(var[1, 1]) * 3, 100
    )
    X, Y = np.meshgrid(x, y)
    coe = 1.0 / ((2 * np.pi) ** 2 * cov_det) ** 0.5
    Z = coe * np.e ** (
        -0.5
        * (
            cov_inv[0, 0] * (X - mean[0]) ** 2
            + (cov_inv[0, 1] + cov_inv[1, 0]) * (X - mean[0]) * (Y - mean[1])
            + cov_inv[1, 1] * (Y - mean[1]) ** 2
        )
    )
    ax.contour(X, Y, Z)
    return ax


class TestClustering(unittest.TestCase):
    def test_bmm(self):
        def _single_test(device):
            n_clusters = 2
            n_features = 2
            n_each_cluster = 100
            n_clustering_clusters = 10
            seed = 0
            n_iter = 5
            precision = "32"
            dt = torch.float64 if precision == "64" else torch.float32
            torch.set_default_dtype(dt)
            bmm = BMM(
                n_clusters=n_clustering_clusters,
                n_input=n_features,
                init_method="kmeans",
                adaptive_momentum=False,
                on_cpu=device == "cpu",
            ).to(device)
            data = get_data(n_clusters, n_features, n_each_cluster, seed)
            t = torch.tensor(data, device=device, dtype=dt)
            set_random_seed(seed)
            bmm.initialize(t)

            from sklearn.mixture import BayesianGaussianMixture as SkBMM

            sk_bmm = SkBMM(
                n_components=n_clustering_clusters,
                max_iter=n_iter,
                init_params="kmeans",
                random_state=0,
                n_init=1,
                # weight_concentration_prior=bmm.weight_concentration_prior_,
                # mean_precision_prior=bmm.mean_precision_prior,
                # mean_prior=bmm.mean_prior_.cpu().numpy(),
                # degrees_of_freedom_prior=bmm.degrees_of_freedom_prior_,
                # covariance_prior=bmm.covariance_prior_.cpu().numpy(),
            )
            sk_bmm._check_parameters(data)
            sk_bmm._initialize_parameters(data, np.random.RandomState(0))

            bmm.set_params(
                weights=bmm.pi,
                means=torch.tensor(sk_bmm.means_, dtype=dt, device=device),
                covariances=torch.tensor(sk_bmm.covariances_, dtype=dt, device=device),
            )
            bmm.mean_precision_ = torch.tensor(
                sk_bmm.mean_precision_, dtype=dt, device=device
            )
            bmm.weight_concentration_0 = torch.tensor(
                sk_bmm.weight_concentration_[0], dtype=dt, device=device
            )
            bmm.weight_concentration_1 = torch.tensor(
                sk_bmm.weight_concentration_[1], dtype=dt, device=device
            )
            bmm.degrees_of_freedom_ = torch.tensor(
                sk_bmm.degrees_of_freedom_, dtype=dt, device=device
            )
            bmm.weight_concentration_prior = sk_bmm.weight_concentration_prior_
            bmm.mean_precision_prior = sk_bmm.mean_precision_prior_
            bmm.mean_prior = torch.tensor(sk_bmm.mean_prior_, dtype=dt, device=device)
            bmm.degrees_of_freedom_prior = sk_bmm.degrees_of_freedom_prior_
            bmm.covariance_prior = torch.tensor(
                sk_bmm.covariance_prior_, dtype=dt, device=device
            )

            # sk_bmm.means_ = bmm.mu.cpu().numpy()
            # sk_bmm.covariances_ = bmm.var.cpu().numpy()
            # sk_bmm.mean_precision_ = bmm.mean_precision_.cpu().numpy()
            # sk_bmm.weight_concentration_ = tuple(x.cpu().numpy() for x in bmm.weight_concentration_)
            # sk_bmm.degrees_of_freedom_ = bmm.degrees_of_freedom_.cpu().numpy()
            # sk_bmm.weight_concentration_prior = bmm.weight_concentration_prior_
            # sk_bmm.mean_precision_prior = bmm.mean_precision_prior
            # sk_bmm.mean_prior = bmm.mean_prior_.cpu().numpy()
            # sk_bmm.degrees_of_freedom_prior = bmm.degrees_of_freedom_prior_
            # sk_bmm.covariance_prior = bmm.covariance_prior_.cpu().numpy()
            # sk_bmm.precisions_cholesky_ = bmm._compute_precision_cholesky(bmm.var).cpu().numpy()
            if precision == "64":
                bmm.double()
            else:
                bmm.float()
            bmm.fit(t, n_iter=n_iter)
            sk_bmm.fit(data)
            res_bmm = bmm.pi.cpu().numpy()
            res_skbmm = sk_bmm.weights_
            np.testing.assert_almost_equal(res_bmm, res_skbmm, decimal=6, verbose=True)

        print("\n-- Clustering on CPU --\n")
        _single_test("cpu")
        if torch.cuda.is_available():
            print("\n-- Clustering on CUDA --\n")
            _single_test("cuda")

    def test_gmm2l(self):
        n_input_1 = 2
        n_input_2 = 2
        n_clusters = 5
        n_clusters_per_cluster = 3
        data_per_cluster = 40
        precision = "32"
        dt = torch.float64 if precision == "64" else torch.float32
        torch.set_default_dtype(dt)
        n_features = n_input_1 + n_input_2
        gmm2l = TwolayerGMM(
            n_clusters=n_clusters,
            n_input_1=n_input_1,
            n_input_2=n_input_2,
            input_1_idx=[0, 1],
            input_2_idx=[2, 3],
            n_clusters_per_cluster=n_clusters_per_cluster,
        )
        # plt.figure()
        # ax1 = plt.subplot(121)
        # ax2 = plt.subplot(122)
        datas = []
        set_random_seed(2)
        for i in range(n_clusters):
            v1 = np.eye(n_input_1) + np.random.rand(n_input_1, n_input_1)
            r1 = np.random.multivariate_normal(
                np.random.rand(n_input_1) * 30,
                v1 * v1.T,
                data_per_cluster,
            )
            for j in range(n_clusters_per_cluster):
                v2 = np.eye(n_input_2) + np.random.rand(n_input_2, n_input_2)
                r2 = np.random.multivariate_normal(
                    np.random.rand(n_input_2) * 30,
                    v2 * v2.T,
                    data_per_cluster,
                )
                data = np.hstack([r1, r2])
                # ax1.scatter(data[:, 0], data[:, 1])
                # ax2.scatter(data[:, 2], data[:, 3])
                datas.append(data)
        x = np.concatenate(datas, axis=0)
        x_tensor = torch.tensor(x, dtype=dt)
        set_random_seed(3)
        if precision == "64":
            gmm2l.double()
        else:
            gmm2l.float()
        gmm2l.fit(x_tensor, 100)
        label = gmm2l.predict(x_tensor)
        # means = gmm2l.first_clustering.mu.cpu().numpy()
        # vars = gmm2l.first_clustering.var.cpu().numpy()
        # for i in range(gmm2l.first_clustering.n_clusters):
        #     mean = means[i, :]
        #     var = vars[i, :, :]
        #     plot_gaussian(mean, var, ax1)
        #
        # means = gmm2l.second_clustering.mu.cpu().numpy()
        # vars = gmm2l.second_clustering.var.cpu().numpy()
        # for i in range(gmm2l.n_total_clusters):
        #     mean = means[i, :]
        #     var = vars[i, :, :]
        #     plot_gaussian(mean, var, ax2)
        # ax.axis("equal")
        # plt.show()
        # plt.close()

    def test_gmm(self):
        def _single_test(device):
            n_clusters = 2
            n_features = 2
            n_each_cluster = 100
            seed = 0
            n_iter = 1
            precision = "32"
            dt = torch.float64 if precision == "64" else torch.float32
            torch.set_default_dtype(dt)
            gmm = GMM(
                n_clusters=n_clusters,
                n_input=n_features,
                init_method="kmeans",
                adaptive_momentum=False,
                on_cpu=device == "cpu",
            ).to(device)
            data = get_data(n_clusters, n_features, n_each_cluster, seed)
            t = torch.tensor(data, device=device, dtype=dt)
            set_random_seed(seed)
            gmm.initialize(t)
            from sklearn.mixture import GaussianMixture as SkGMM

            sk_gmm = SkGMM(
                n_components=n_clusters,
                covariance_type="full",
                init_params="random",
                random_state=0,
                max_iter=n_iter,
            )

            sk_gmm._initialize_parameters(data, np.random.RandomState(0))

            (
                weights_,
                means_,
                covariances_,
                precisions_cholesky_,
            ) = sk_gmm._get_parameters()
            gmm.set_params(
                weights=torch.tensor(weights_, dtype=dt, device=device),
                means=torch.tensor(means_, dtype=dt, device=device),
                covariances=torch.tensor(covariances_, dtype=dt, device=device),
            )
            if precision == "64":
                gmm.double()
            else:
                gmm.float()
            gmm.fit(t, n_iter=n_iter)
            sk_gmm.fit(data)
            res_gmm = gmm.pi.cpu().numpy()
            res_skgmm = sk_gmm.weights_
            np.testing.assert_almost_equal(res_gmm, res_skgmm, decimal=6, verbose=True)

        print("\n-- Clustering on CPU --\n")
        _single_test("cpu")
        if torch.cuda.is_available():
            print("\n-- Clustering on CUDA --\n")
            _single_test("cuda")
