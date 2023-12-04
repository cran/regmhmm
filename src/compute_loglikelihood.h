#ifndef COMPUTE_LOGLIKELIHOOD_H
#define COMPUTE_LOGLIKELIHOOD_H

#include <RcppArmadillo.h>
#include "some_function.h"

using namespace arma;
using namespace Rcpp;
using namespace std;

double compute_loglikelihood(arma::vec &delta, // List forward(vec & delta,
                             arma::vec &Y,
                             arma::mat &A,
                             arma::mat &B,
                             arma::mat &X,
                             char &family);

#endif
