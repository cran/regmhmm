// -*- mode: C++; c-indent-level: 4; c-basic-offset: 4; indent-tabs-mode: nil; -*-
#include <RcppArmadillo.h>
#include "some_function.h"
#include "forward.h"
#include "backward.h"
#include "forward_backward.h"

using namespace arma;
using namespace Rcpp;
using namespace std;

//' @title Probability Calculation in Hidden Markov Models using Forward-Backward Algorithm
//'
//' @description
//' Calculate the probability given parameters of a hidden Markov model using a combination of the forward and backward algorithms.
//'
//' @param delta a vector of length S specifying the initial probabilities.
//' @param Y a vector of observations of size T.
//' @param X a design matrix of size T x p.
//' @param A a matrix of size S x S specifying the transition probabilities.
//' @param B a matrix of size S x (p + 1) specifying the GLM parameters of the emission probabilities.
//' @param family the family of the response.
//'
//' @returns
//' A list object with the following slots:
//'
//' \item{log_alpha}{a matrix of size S x T that is the forward probabilities in log scale.}
//'
//' \item{log_beta}{a matrix of size S x T that is the backward probabilities in log scale.}
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
//' forward_backward_C <- forward_backward(
//'   delta = parameters_setting$init_vec,
//'   Y = simulated_data$y_mat[1, ],
//'   A = parameters_setting$trans_mat,
//'   B = parameters_setting$emis_mat,
//'   X = simulated_data$X_array[, 1:4, 1],
//'   family = "P"
//' )
//'
//' @export
// [[Rcpp::export]]
List forward_backward(arma::vec &delta,
                      arma::vec &Y,
                      arma::mat &A,
                      arma::mat &B,
                      arma::mat &X,
                      char &family)
{
  arma::mat log_alpha = forward(delta, Y, A, B, X, family);
  arma::mat log_beta = backward(delta, Y, A, B, X, family);
  return List::create(Named("log_alpha") = log_alpha, _["log_beta"] = log_beta);
}
