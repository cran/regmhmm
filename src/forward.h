#ifndef FORWARD_H
#define FORWARD_H

#include <RcppArmadillo.h>
#include "some_function.h"

using namespace arma;
using namespace Rcpp;
using namespace std;

arma::mat forward(arma::vec &delta, // List forward(vec & delta,
                  arma::vec &Y,
                  arma::mat &A,
                  arma::mat &B,
                  arma::mat &X,
                  char &family);

#endif
