// -*- mode: C++; c-indent-level: 4; c-basic-offset: 4; indent-tabs-mode: nil; -*-
#include <RcppArmadillo.h>
#include "compute_joint_state.h"
#include "compute_state.h"
#include "IRLS_EM.h"
#include "HMM_one_step.h"

using namespace arma;
using namespace Rcpp;
using namespace std;

// [[Rcpp::depends(RcppArmadillo)]]
//' @title Single EM Iteration for Fitting Hidden Markov Models (HMM)
//'
//' @description
//' Execute a single iteration of the Expectation-Maximization (EM) algorithm tailored for fitting Hidden Markov Models (HMMs).
//'
//' @param delta a vector of length S specifying the initial probabilities.
//' @param A a matrix of size S x S specifying the transition probabilities.
//' @param B a matrix of size S x (p + 1) specifying the GLM parameters of the emission probabilities.
//' @param Y_mat a matrix of observations of size N x T.
//' @param X_cube a design array of size T x p x N.
//' @param family the family of the response.
//' @param eps_IRLS convergence tolerance in the iteratively reweighted least squares step.
//' @param max_N_IRLS the maximal number of IRLS iterations.
//'
//' @returns
//' A list object with the following slots:
//'
//' \item{delta_hat}{the estimate of delta.}
//'
//' \item{A_hat}{the estimate of A.}
//'
//' \item{B_hat}{the estimate of B.}
//'
//' \item{log_likelihood}{the log-likelihood of the model.}
//'
//' @examples
//' # Example usage of the function
//' seed_num <- 1
//' p_noise <- 2
//' N <- 100
//' N_persub <- 10
//' parameters_setting <- list(
//'   init_vec = c(0.5, 0.5),
//'   trans_mat = matrix(c(0.7, 0.3, 0.2, 0.8), nrow = 2, byrow = TRUE),
//'   emis_mat = matrix(c(1, 0.5, 0.5, 2), nrow = 2, byrow = TRUE)
//' )
//' simulated_data <- simulate_HMM_data(seed_num, p_noise, N, N_persub, parameters_setting)
//' init_start = c(0.5, 0.5)
//' trans_start = matrix(c(0.5, 0.5, 0.5, 0.5), nrow = 2)
//' emis_start = matrix(rep(1, 8), nrow = 2)
//' HMM_fit_raw_one_step <- HMM_one_step(delta=as.matrix(init_start),
//'                Y_mat=simulated_data$y_mat,
//'                A=trans_start,
//'                B=emis_start,
//'                X_cube=simulated_data$X_array,
//'                family="P")
//'
//' @export
// [[Rcpp::export]]
List HMM_one_step(arma::vec &delta,
                  arma::mat &Y_mat, // N*T
                  arma::mat &A,
                  arma::mat &B,
                  arma::cube &X_cube, // T*p*N
                  char &family,
                  double eps_IRLS = 1e-4,
                  int max_N_IRLS = 300)
{
  // some marcos
  int T = Y_mat.n_cols;
  int S = delta.size();
  int N = Y_mat.n_rows;
  int p = X_cube.n_cols;

  // working Y and X
  arma::vec Y;
  arma::mat X;

  // flatten Y and X
  arma::vec Y_long(N * T);
  arma::mat X_long(N * T, p);

  // long gamma
  arma::mat gamma_mat_wide(S, N * T);
  arma::vec working_gamma(N * T);

  // recivers
  arma::mat gamma_slice(S, T);
  List get_forward_backward;
  arma::cube xi(S, S, T, arma::fill::zeros);

  // A, B, delta
  arma::mat A_hat(S, S);
  arma::mat B_hat(S, p);
  arma::vec delta_hat(S);

  // objects to compute A, B, delta
  arma::mat numerator_xi_mat(S, S), denomiator_xi_mat(S, S);
  arma::mat numerator_gamma_mat(S, 1), denomiator_gamma_mat(S, 1);
  arma::vec init_beta(p);

  // small things
  arma::vec rec_beta(p);
  double ll = 0;

  // flatten X_cube and Y_mat
  for (int i = 0; i < N; i++)
  {
    for (int t = 0; t < T; t++)
    {
      for (int j = 0; j < p; j++)
      {
        X_long(i * T + t, j) = X_cube(t, j, i);
      }
      Y_long(i * T + t) = Y_mat(i, t);
    }
  }

  // initialize
  for (int s = 0; s < S; s++)
  {
    delta_hat(s) = 0;
    for (int u = 0; u < S; u++)
    {
      A_hat(s, u) = 0;
      numerator_xi_mat(s, u) = 0;
      denomiator_xi_mat(s, u) = 0;
    }
    for (int i = 0; i < p; i++)
    {
      B_hat(s, i) = 0;
    }
    numerator_gamma_mat(s, 0) = 0;
    denomiator_gamma_mat(s, 0) = 0;
    for (int t = 0; t < T; t++)
    {
      gamma_slice(s, t) = 0;
    }
  }

  // E step
  for (int i = 0; i < N; i++)
  {
    Y = Y_mat.row(i).t();
    X = X_cube.slice(i);
    xi = compute_joint_state(delta, Y, A, B, X, family);
    gamma_slice = compute_state(delta, Y, A, B, X, family);
    for (int s = 0; s < S; s++)
    {
      for (int t = 0; t < T; t++)
      {
        gamma_mat_wide(s, i * T + t) = gamma_slice(s, t);
        if (t == (T - 1))
          continue; // don't want xi to add the finally iteration
        for (int u = 0; u < S; u++)
        {
          numerator_xi_mat(s, u) += xi(s, u, t);
          for (int k = 0; k < S; k++)
          {
            denomiator_xi_mat(s, u) += xi(s, k, t);
          }
        }
      }
    }
    delta_hat += gamma_slice.col(0);
  }

  // M step
  // for A
  for (int s = 0; s < S; s++)
  {
    for (int u = 0; u < S; u++)
    {
      A_hat(s, u) = numerator_xi_mat(s, u) / denomiator_xi_mat(s, u);
    }
  }

  // for delta
  delta_hat = delta_hat / N;

  // for B
  for (int s = 0; s < S; s++)
  {
    working_gamma = gamma_mat_wide.row(s).t();
    init_beta = B.row(s).t();
    rec_beta = IRLS_EM(X_long, working_gamma, Y_long, init_beta, family, eps_IRLS, max_N_IRLS);
    for (int a = 0; a < p; a++)
    {
      B_hat(s, a) = rec_beta(a);
    }
  }

  // for log-likelihood
  for (int i = 0; i < N; i++)
  {
    Y = Y_mat.row(i).t();
    X = X_cube.slice(i);
    ll += compute_loglikelihood(delta_hat, Y, A_hat, B_hat, X, family);
  }

  return List::create(Named("delta_hat") = delta_hat, _["B_hat"] = B_hat, _["A_hat"] = A_hat, _["log_likelihood"] = ll);
}
