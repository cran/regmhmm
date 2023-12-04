// -*- mode: C++; c-indent-level: 4; c-basic-offset: 4; indent-tabs-mode: nil; -*-
#include <RcppArmadillo.h>

using namespace arma;
using namespace Rcpp;
using namespace std;

double inner_product(NumericVector x, NumericVector y)
{
  double rec = 0;

  for (int i = 0; i < x.size(); i++)
  {
    rec += x[i] * y[i];
  }
  return (rec);
}

void assign_vec_from_matrix(arma::vec &v, arma::vec m_1row)
{
  for (int i = 0; i < v.size(); i++)
  {
    v(i) = m_1row(i);
  }
}

double inv_logit(double &x)
{
  double y;
  y = exp(x) / (1 + exp(x));
  return y;
}

// create diag P
arma::mat diag_P_mat(double &x, arma::mat &A, arma::mat &B, char &family)
{
  int S = A.n_cols;
  arma::mat P(S, S);

  for (int s = 0; s < S; s++)
  {
    if (family == 'D')
    {
      P(s, s) = B(s, x - 1);
    }
    else if (family == 'G')
    {
      P(s, s) = R::dnorm(x, B(s, 0), B(s, 1), false);
    }
    else if (family == 'P')
    {
      P(s, s) = R::dpois(x, B(s, 0), false);
    }
  }
  return P;
}

// create diag P when there are covariates
arma::mat diag_P_mat_covariate(double &O,
                               arma::mat &A,
                               arma::mat &B, // B matrix would serve as a matrix of betas, with p+1 columns as p coefficient and 1 variance
                               arma::vec &x,
                               char &family)
{
  int S = A.n_cols;
  double mu_working, eta_working = 0;
  int p = x.size(); // # covariates
  arma::mat P(S, S);
  arma::vec beta_working(p);

  for (int s = 0; s < S; s++)
  {
    // get beta from B, work for every familybution
    for (int i = 0; i < p; i++)
    {
      beta_working(i) = B(s, i);
    }

    if (family == 'D')
    {
      eta_working = as_scalar(x.t() * beta_working);
      mu_working = inv_logit(eta_working);
      P(s, s) = R::dbinom(O, 1, mu_working, true);
    }
    else if (family == 'G')
    {
      eta_working = as_scalar(x.t() * beta_working);
      mu_working = eta_working;
      P(s, s) = R::dnorm(O, mu_working, B(s, p), true);
    }
    else if (family == 'P')
    {
      eta_working = as_scalar(x.t() * beta_working);
      mu_working = exp(eta_working);
      P(s, s) = R::dpois(O, mu_working, true);
    }
  }
  return P;
}

// an extension with random effect
arma::mat diag_P_mat_covariate_add_random(double &O,
                                          arma::mat &A,
                                          arma::mat &B, // B matrix would serve as a matrix of betas, with p+1 columns as p coefficient and 1 variance
                                          arma::vec &x,
                                          arma::vec &z,
                                          arma::vec &b, // random effect, b_l, when there is only random intercept, it is just a scalar
                                          char &family)
{
  int S = A.n_cols;
  double mu_working, eta_working = 0;
  int p = x.size(); // # covariates
  arma::mat P(S, S);
  arma::vec beta_working(p);

  for (int s = 0; s < S; s++)
  {
    // get beta from B, work for every familybution
    for (int i = 0; i < p; i++)
    {
      beta_working(i) = B(s, i);
    }

    if (family == 'D')
    {
      eta_working = as_scalar(x.t() * beta_working + z.t() * b);
      mu_working = inv_logit(eta_working);
      // P(s, s) = R::dbinom(O, 1, mu_working, false);
      P(s, s) = R::dbinom(O, 1, mu_working, true);
    }
    else if (family == 'G')
    {
      eta_working = as_scalar(x.t() * beta_working + z.t() * b);
      mu_working = eta_working;
      // P(s, s) = R::dnorm(O, mu_working, B(s, p), false);
      P(s, s) = R::dnorm(O, 1, mu_working, true);
    }
    else if (family == 'P')
    {
      eta_working = as_scalar(x.t() * beta_working + z.t() * b);
      mu_working = exp(eta_working);
      // P(s, s) = R::dpois(O, mu_working, false);
      P(s, s) = R::dpois(O, mu_working, true);
    }
  }
  return P;
}

// for EM with covariates

// old
void assign_mat_from_cube(NumericMatrix &mat, arma::cube cube, int s)
{
  for (int i = 0; i < mat.nrow(); i++)
  {
    for (int j = 0; j < mat.ncol(); j++)
    {
      mat(i, j) = cube(s, i, j);
    }
  }
}

NumericMatrix armaMat_2_rcppMat(arma::mat &arma_mat)
{
  NumericMatrix Rcpp_mat(arma_mat.n_rows, arma_mat.n_cols);
  for (int i = 0; i < Rcpp_mat.nrow(); i++)
  {
    for (int j = 0; j < Rcpp_mat.ncol(); j++)
    {
      Rcpp_mat(i, j) = arma_mat(i, j);
    }
  }
  return (Rcpp_mat);
}
// end of old

// in R, it is row*col*slice
// in cube for arma, it is col*row*slice
// => cube(col, row, slice)=array(row, col, slice)
// but after slice, it becomes correct again
void cube_2_mat(arma::mat &mat,  // N*T, p
                arma::cube cube) // T*p*N
{
  int T = cube.n_rows;
  int N = cube.n_slices;
  int p = cube.n_cols;
  Rcout << "T is: " << T << "\n";
  Rcout << "N is: " << N << "\n";
  Rcout << "p is: " << p << "\n";

  for (int i = 0; i < N; i++)
  {
    for (int j = 0; j < p; j++)
    {
      for (int t = 0; t < T; t++)
      {
        Rcout << "i is: " << i << "\n";
        Rcout << "j is: " << j << "\n";
        Rcout << "t is: " << t << "\n";
        mat(i * T + t, j) = cube(t, j, i);
      }
    }
  }
}
