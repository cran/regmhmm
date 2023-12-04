// -*- mode: C++; c-indent-level: 4; c-basic-offset: 4; indent-tabs-mode: nil; -*-
// compute gamma (posterior probability used in the EM algorithm)
#include <RcppArmadillo.h>
#include "some_function.h"
#include "forward_backward.h"
#include "compute_loglikelihood.h"
#include "compute_state.h"

using namespace arma;
using namespace Rcpp;
using namespace std;

//' @title Posterior Probability Estimation for Hidden States in Hidden Markov Models
//'
//' @description
//' Calculate the posterior probability of hidden states given parameters of a hidden Markov model.
//'
//' @param delta a vector of length S specifying the initial probabilities.
//' @param Y a vector of observations of size T.
//' @param X a design matrix of size T x p.
//' @param A a matrix of size S x S specifying the transition probabilities.
//' @param B a matrix of size S x (p + 1) specifying the GLM parameters of the emission probabilities.
//' @param family the family of the response.
//'
//' @returns
//' A matrix of size S x T that represents the posterior probability of hidden states.
//'
//' @examples
//' # Example usage of the function
//' parameters_setting <- list()
//' parameters_setting$emis_mat <- matrix(NA, nrow = 2, ncol = 4)
//' parameters_setting$emis_mat[1, 1] <- 0.1
//' parameters_setting$emis_mat[1, 2] <- 0.5
//' parameters_setting$emis_mat[1, 3] <- -0.75
//' parameters_setting$emis_mat[1, 4] <- 0.75
//' parameters_setting$emis_mat[2, 1] <- -0.1
//' parameters_setting$emis_mat[2, 2] <- -0.5
//' parameters_setting$emis_mat[2, 3] <- 0.75
//' parameters_setting$emis_mat[2, 4] <- 1
//' parameters_setting$trans_mat <- matrix(NA, nrow = 2, ncol = 2)
//' parameters_setting$trans_mat[1, 1] <- 0.65
//' parameters_setting$trans_mat[1, 2] <- 0.35
//' parameters_setting$trans_mat[2, 1] <- 0.2
//' parameters_setting$trans_mat[2, 2] <- 0.8
//' parameters_setting$init_vec <- c(0.65, 0.35)
//' simulated_data <- simulate_HMM_data(
//'   seed_num = 1,
//'   p_noise = 7,
//'   N = 100,
//'   N_persub = 10,
//'   parameters_setting = parameters_setting
//' )
//' compute_state_get <- compute_state(
//'   delta = parameters_setting$init_vec,
//'   Y = simulated_data$y_mat[1, ],
//'   A = parameters_setting$trans_mat,
//'   B = parameters_setting$emis_mat,
//'   X = simulated_data$X_array[, 1:4, 1],
//'  family = "P")
//'
//' @export
// [[Rcpp::export]]
arma::mat compute_state(arma::vec &delta,
                        arma::vec &Y,
                        arma::mat &A,
                        arma::mat &B,
                        arma::mat &X,
                        char &family)
{

    int T = Y.size();
    int S = delta.size();
    arma::mat gamma(S, T);
    NumericMatrix log_alpha(S, T), log_beta(S, T);
    List get_forward_backward;
    get_forward_backward = forward_backward(delta, Y, A, B, X, family);
    log_alpha = as<NumericMatrix>(get_forward_backward["log_alpha"]);
    log_beta = as<NumericMatrix>(get_forward_backward["log_beta"]);
    double ll = compute_loglikelihood(delta, Y, A, B, X, family);

    for (int s = 0; s < S; s++)
    {
        for (int t = 0; t < T; t++)
        {
            gamma(s, t) = exp(log_alpha(s, t) + log_beta(s, t) - ll);
        }
    }
    return gamma;
}
