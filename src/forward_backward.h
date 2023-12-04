#ifndef FORWARD_BACKWARD_H
#define FORWARD_BACKWARD_H

#include <RcppArmadillo.h>
#include "some_function.h"
#include "forward.h"
#include "backward.h"

using namespace arma;
using namespace Rcpp;
using namespace std;

List forward_backward(arma::vec &delta,
                      arma::vec &Y,
                      arma::mat &A,
                      arma::mat &B,
                      arma::mat &X,
                      char &family);

#endif
