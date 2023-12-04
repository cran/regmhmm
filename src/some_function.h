#ifndef SOME_FUNCTION_H
#define SOME_FUNCTION_H

#include <RcppArmadillo.h>

using namespace arma;
using namespace Rcpp;
using namespace std;

double inner_product(NumericVector x, NumericVector y);

void assign_vec_from_matrix(NumericVector &v, NumericVector m_1row);

double inv_logit(double &x);

arma::mat diag_P_mat(double &x, arma::mat &A, arma::mat &B, char &family);

arma::mat diag_P_mat_covariate(double &O,
                               arma::mat &A,
                               arma::mat &B, // B matrix would serve as a matrix of betas, with p+1 columns as p coefficient and 1 variance
                               arma::vec &X,
                               char &family);

arma::mat diag_P_mat_covariate_add_random(double &O,
                                          arma::mat &A,
                                          arma::mat &B, // B matrix would serve as a matrix of betas, with p+1 columns as p coefficient and 1 variance
                                          arma::vec &x,
                                          arma::vec &z,
                                          arma::vec &b, // random effect, b_l, when there is only random intercept, it is just a scalar
                                          char &family);

void assign_mat_from_cube(NumericMatrix &mat, arma::cube cube, int s);

NumericMatrix armaMat_2_rcppMat(arma::mat &arma_mat);

void cube_2_mat(arma::mat &mat, arma::cube cube);

#endif
