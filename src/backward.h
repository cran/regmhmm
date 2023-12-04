#ifndef BACKWARD_H
#define BACKWARD_H

#include <RcppArmadillo.h>
#include "some_function.h"

using namespace arma;
using namespace Rcpp;
using namespace std;

arma::mat backward(arma::vec &delta, // List forward(vec & pi,
                   arma::vec &Y,
                   arma::mat &A,
                   arma::mat &B,
                   arma::mat &X,
                   char &family);

#endif
