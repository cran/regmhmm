// -*- mode: C++; c-indent-level: 4; c-basic-offset: 4; indent-tabs-mode: nil; -*-
#include <RcppArmadillo.h>
#include "HMM_one_step.h"

using namespace arma;
using namespace Rcpp;
using namespace std;

// [[Rcpp::depends(RcppArmadillo)]]
//' @title Fit Hidden Markov Model (HMM)
//'
//' @description
//' Employ this function to fit a Hidden Markov Model (HMM) to the provided data. It iteratively estimates model parameters using the EM algorithm.
//'
//' @param delta a vector of length S specifying the initial probabilities.
//' @param A a matrix of size S x S specifying the transition probabilities.
//' @param B a matrix of size S x (p + 1) specifying the GLM parameters of the emission probabilities.
//' @param Y_mat a matrix of observations of size N x T.
//' @param X_cube a design array of size T x p x N.
//' @param family the family of the response.
//' @param eps convergence tolerance in the EM algorithm for fitting HMM.
//' @param eps_IRLS convergence tolerance in the iteratively reweighted least squares step.
//' @param N_iter the maximal number of the EM algorithm for fitting HMM.
//' @param max_N_IRLS the maximal number of IRLS iterations.
//' @param trace logical indicating if detailed output should be produced during the fitting process.
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
//' HMM_fit_raw <- HMM_C_raw(delta=as.matrix(init_start),
//'                Y_mat=simulated_data$y_mat,
//'                A=trans_start,
//'                B=emis_start,
//'                X_cube=simulated_data$X_array,
//'                family="P",
//'                eps=1e-4,
//'                trace = 0
//' )
//'
//' @export
// [[Rcpp::export]]
List HMM_C_raw(arma::vec &delta,
               arma::mat &Y_mat, // T*N
               arma::mat &A,
               arma::mat &B,
               arma::cube &X_cube, // T*p*N
               char &family,
               double eps = 1e-5,
               double eps_IRLS = 1e-4,
               int N_iter = 1000,
               int max_N_IRLS = 300,
               int trace = 0)
{
      List get_EM_one;
      // run once for initialization
      arma::mat A_hat;
      arma::mat B_hat;
      arma::vec delta_hat;
      double ll, ll_old;

      get_EM_one = HMM_one_step(delta, Y_mat, A, B, X_cube, family, eps_IRLS, max_N_IRLS);
      A_hat = as<arma::mat>(get_EM_one["A_hat"]);
      B_hat = as<arma::mat>(get_EM_one["B_hat"]);
      delta_hat = as<arma::vec>(get_EM_one["delta_hat"]);

      // initialize the list with 1 run
      if (trace == 1)
      {
            Rcout << "Initial run:\n"
                  << ".\n";
            Rcout << "A_hat is:\n"
                  << A_hat << ".\n";
            Rcout << "B_hat is:\n"
                  << B_hat << ".\n";
      }

      for (int i = 1; i < N_iter; i++)
      {
            ll_old = get_EM_one["log_likelihood"];
            get_EM_one = HMM_one_step(delta_hat, Y_mat, A_hat, B_hat, X_cube, family, eps_IRLS, max_N_IRLS);

            A_hat = as<arma::mat>(get_EM_one["A_hat"]);
            B_hat = as<arma::mat>(get_EM_one["B_hat"]);
            delta_hat = as<arma::vec>(get_EM_one["delta_hat"]);

            ll = get_EM_one["log_likelihood"];

            if (trace == 1)
            {
                  Rcout << "Currently iterating: " << i << ".\n";
                  Rcout << "A_hat is:\n"
                        << A_hat << ".\n";
                  Rcout << "B_hat is:\n"
                        << B_hat << ".\n";
            }
            if (min(abs(ll_old - ll)) < eps)
            {
                  if (trace == 1)
                  {
                        Rcout << "For EM, it takes " << i << " steps to converge."
                              << "\n";
                  }
                  break;
            }
      }
      ll = get_EM_one["log_likelihood"];

      return List::create(Named("delta_hat") = delta_hat, _["B_hat"] = B_hat, _["A_hat"] = A_hat, _["log_likelihood"] = ll);
}
