import matplotlib.pyplot as plt
import numpy as np
import os

K = 2           # Number of Gaussians in the mixture model
NUM_TRIALS = 3  # Number of trials to run (can be adjusted for debugging)
UNLABELED = -1  # Cluster label for unlabeled data points (do not change)

def main():
    print('Running semi-supervised EM algorithm...')
    
    # Load dataset
    train_path = os.path.join('..', 'data', 'ds3_train.csv')
    x, z = load_gmm_dataset(train_path)
    x_tilde = None

#     if is_semi_supervised:
#         # Split into labeled and unlabeled examples
#         labeled_idxs = (z != UNLABELED).squeeze()
#         x_tilde = x[labeled_idxs, :]   # Labeled examples
#         z = z[labeled_idxs, :]         # Corresponding labels
#         x = x[~labeled_idxs, :]        # Unlabeled examples

    m, n = x.shape
    indices = np.random.randint(0, m, m).reshape((int(m/K), K))
    x_val = x[indices]
    mu = np.mean(x_val, axis = 0)
    sigma = np.array([np.cov(x_val[:, i, :].T) for i in range(K)])
    
    # initialize phi
    phi = np.ones((K,))/K
    # initialize w
    w = np.ones((m, K))/K
    # run the SSEM algorithm
    w = run_semi_supervised_em(x, x_tilde, z, w, phi, mu, sigma)


def run_semi_supervised_em(x, x_tilde, z, w, phi, mu, sigma):
    
    alpha = 20.  # Weight for the labeled examples
    eps = 1e-3   # Convergence threshold
    max_iter = 1000
    it = 0
    ll = prev_ll = None
    
    mt, nt = x_tilde.shape
    m, n = x.shape
    
    while it < max_iter and (prev_ll is None or np.abs(ll - prev_ll) >= eps):
        prev_ll = ll
        # (1) E-step: Update your estimates in w
        for i in range(m):
            sum_ = sum(gaussian(x[i], phi[l], mu[l], sigma[l]) for l in range(K))
            for j in range(K):
                w[i][j] = gaussian(x[i], phi[j], mu[j], sigma[j])/sum_

        # (2) M-step: Update the model parameters phi, mu, and sigma
        sum_zit = np.zeros(K)
        for j in range(K):
            for i in range(mt):
                if z[i] == j:
                    sum_zit[j] += alpha

        mu_tilde_num = np.zeros((K, n))
        for j in range(K):
            for i in range(mt):
                if z[i] == j:
                    mu_tilde_num[j, :] += alpha*x_tilde[i]

        mu = ((np.matmul(x.T, w) + mu_tilde_num.T)/(np.sum(w, axis = 0) + sum_zit)).T
        phi = (np.sum(w, axis = 0) + sum_zit)/(m + alpha*mt)
        sum_ = np.sum(w, axis = 0) + sum_zit
        
        for j in range(K):
            inner_sum = np.zeros_like(sigma[0])
            for i in range(m):
                inner_sum += w[i][j]*np.outer(x[i] - mu[j], x[i] - mu[j])
            for i in range(mt):
                if z[i] == j:
                    inner_sum += alpha*np.outer(x_tilde[i] - mu[j], x_tilde[i] - mu[j])
            sigma[j] = inner_sum/sum_[j]

        # (3) Compute the log-likelihood of the data to check for convergence.
        ll = 0
        for i in range(m):
            inner_sum = 0
            for j in range(K):
                inner_sum += gaussian(x[i], phi[j], mu[j], sigma[j])
            ll += np.log(inner_sum)

        if it % 10 == 0: print(ll, it)
        it += 1
    print("It: %d"%it)
    return w


def gaussian(x, phi, mu, sigma):
    expon = np.exp(-0.5*(x - mu).T.dot(np.linalg.inv(sigma)).dot(x - mu))
    expon /= (np.sqrt(np.linalg.det(sigma))*(2*np.pi)**(N/2))
    return expon*phi


def plot_gmm_preds(x, z, with_supervision, plot_id):
    pass

# TODO: extract dataset
def load_dataset(csv_path):
    return X, X_t, y_t
    
if __name__ == '__main__':
    np.random.seed(229)
    for t in range(NUM_TRIALS):
        main()