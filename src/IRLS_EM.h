#ifndef IRLS_EM_H // To make sure you don't declare the function more than once by including the header multiple times.
#define IRLS_EM_H

#include <RcppArmadillo.h>
#include "some_function.h"

using namespace arma;
using namespace Rcpp;
using namespace std;

arma::vec IRLS_EM(arma::mat &X, arma::vec &gamma, arma::vec &y, arma::vec &beta, char &family, double eps_IRLS, int max_N);

#endif
