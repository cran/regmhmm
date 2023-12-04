// -*- mode: C++; c-indent-level: 4; c-basic-offset: 4; indent-tabs-mode: nil; -*-
#include <RcppArmadillo.h>
#include "some_function.h"
#include "backward.h"

using namespace arma;
using namespace Rcpp;
using namespace std;

//' @title Probability Calculation using the Backward Algorithm for Hidden Markov Models
//'
//' @description
//' Calculate the probability given parameters of a hidden Markov model utilizing the backward algorithm.
//'
//' @param delta a vector of length S specifying the initial probabilities.
//' @param Y a vector of observations of size T.
//' @param X a design matrix of size T x p.
//' @param A a matrix of size S x S specifying the transition probabilities.
//' @param B a matrix of size S x (p + 1) specifying the GLM parameters of the emission probabilities.
//' @param family the family of the response.
//' @returns
//'
//' A matrix of size S x T that is the backward probabilities in log scale.
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
//' backward_C <- backward(
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
arma::mat backward(arma::vec &delta, // List forward(vec & delta,
                   arma::vec &Y,
                   arma::mat &A,
                   arma::mat &B,
                   arma::mat &X,
                   char &family)
{
  int T = Y.size();
  int S = delta.size();
  int p = X.n_cols;
  arma::mat log_beta(S, T);
  arma::mat working_P(S, S);
  arma::vec ones_v = ones(S);

  arma::mat log_phi(S, T);
  arma::vec log_v(S);
  arma::vec x(p);
  arma::vec log_w(T);
  double log_u = 0;

  // Log-Sum-Exp
  mat log_A(S, S);
  double c;
  double working_sum;
  mat working_log_A_P(S, S);

  // log sum exp for v because of small phi
  for (int j = 0; j < S; j++)
  {
    for (int k = 0; k < S; k++)
    {
      log_A(j, k) = log(A(j, k));
    }
  }

  // when t=T
  x = X.row(T - 1).t();
  working_P = diag_P_mat_covariate(Y(T - 1), A, B, x, family);
  for (int s = 0; s < S; s++)
  {
    log_v(s) = log(0);
  }
  log_w(T - 1) = log(as_scalar(ones_v.t() * ones_v));
  log_phi.col(T - 1) = log(ones_v) - log_w(T - 1);
  log_beta.col(T - 1) = log_w(T - 1) + log_phi.col(T - 1);

  // when t=T to 2
  for (int t = T - 2; t >= 0; t--)
  {
    x = X.row(t + 1).t();
    working_P = diag_P_mat_covariate(Y(t + 1), A, B, x, family);

    for (int j = 0; j < S; j++)
    {
      for (int k = 0; k < S; k++)
      {
        working_log_A_P(j, k) = log_A(j, k) + working_P(k, k);
      }
    }

    for (int j = 0; j < S; j++)
    {
      c = max(log_phi.col(t + 1) + working_log_A_P.row(j).t());
      log_v(j) = c;
      working_sum = 0;
      for (int k = 0; k < S; k++)
      {
        working_sum += exp(log_phi(k, t + 1) + working_log_A_P(j, k) - c);
      }
      log_v(j) += log(working_sum);
    }

    c = max(log_v);
    log_u = c;
    working_sum = 0;
    for (int j = 0; j < S; j++)
    {
      working_sum += exp(log_v(j) - c);
    }
    log_u += log(working_sum);
    log_phi.col(t) = log_v - log_u;
    log_w(t) = log_w(t + 1) + log_u;
    log_beta.col(t) = log_w(t) + log_phi.col(t);
  }

  return log_beta;
}
