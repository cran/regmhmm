// -*- mode: C++; c-indent-level: 4; c-basic-offset: 4; indent-tabs-mode: nil; -*-
#include <RcppArmadillo.h>
#include "some_function.h"
#include "IRLS_EM.h"

using namespace arma;
using namespace Rcpp;
using namespace std;

// [[Rcpp::depends(RcppArmadillo)]]

arma::vec IRLS_EM_one(arma::mat &X, arma::vec &gamma, arma::vec &Y, arma::vec &beta, char &family)
{
    int p = X.n_cols;
    int N = Y.size();
    arma::vec rec_beta(p);
    arma::vec fderi_eta(N), fderi_mu(N);
    arma::vec eta(N), mu(N);
    arma::vec Z(N);
    arma::vec W(N);

    eta = X * beta;

    // logit
    if (family == 'D')
    {
        for (int i = 0; i < N; i++)
        {
            // mu(i) = exp(eta(i)) / (1 + exp(eta(i)));
            mu(i) = 1 / (1 + exp(-eta(i)));
            fderi_eta(i) = 1 / (mu(i) * (1 - mu(i)));
            // fderi_mu(i) = exp(eta(i)) / pow((1 + exp(eta(i))), 2);
            fderi_mu(i) = sqrt(exp(eta(i))) / (1 + exp(eta(i)));
            Z(i) = eta(i) + (Y(i) - mu(i)) * fderi_eta(i);
            // W(i) = sqrt(fderi_mu(i) * gamma(i));
            W(i) = fderi_mu(i) * sqrt(gamma(i));
        }
    }
    else if (family == 'P')
    {
        for (int i = 0; i < N; i++)
        {
            mu(i) = exp(eta(i));
            fderi_eta(i) = 1 / mu(i);
            fderi_mu(i) = exp(eta(i));
            Z(i) = eta(i) + (Y(i) - mu(i)) * fderi_eta(i);
            W(i) = sqrt(fderi_mu(i) * gamma(i));
        }
    }

    rec_beta = arma::solve((X.t() * arma::diagmat(W)) * (arma::diagmat(W) * X),
                           X.t() * arma::diagmat(W) * (arma::diagmat(W) * Z));
    return rec_beta;
}

//' Iterative Reweighted Least Squares algorithm for optimizing the parameters in the M-step of the EM algorithm.
//'
//' @title Iterative Reweighted Least Squares for the EM algorithm
//'
//' @param Y A vector of observations of size n.
//' @param X A design matrix of size n x p.
//' @param gamma A vector of size n specifying the posterior probability of the hidden states.
//' @param beta A vector of size p + 1 specifying the GLM parameters.
//' @param family The family of the response.
//' @param eps_IRLS convergence tolerance in the iteratively reweighted least squares step.
//' @param max_N the maximal number of IRLS iterations.
//'
//' @returns
//' A vector representing the estimates of beta.
//'
//' @examples
//' \dontrun{
//' # Example usage of the function
//' IRLS_EM_one_step <- IRLS_EM_one(X,
//'                                 gamma,
//'                                 Y,
//'                                 beta,
//'                                 family)
//' }
//' @export
// [[Rcpp::export]]
arma::vec IRLS_EM(arma::mat &X, arma::vec &gamma, arma::vec &Y, arma::vec &beta, char &family, double eps_IRLS, int max_N)
{
    int p = X.n_cols;
    arma::vec rec_beta(p), old_beta(p);

    old_beta = beta;

    for (int i = 0; i < max_N; i++)
    {
        rec_beta = IRLS_EM_one(X, gamma, Y, old_beta, family);
        if (min(abs(old_beta - rec_beta)) < eps_IRLS)
        {
            break;
        }

        if (i == (max_N - 1))
        {
            Rcout << "IRLS doesn't converge after " << i << " steps."
                  << "\n";
        }
        old_beta = rec_beta;
    }

    return rec_beta;
}
