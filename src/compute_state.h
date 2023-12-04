#ifndef compute_state_H // To make sure you don't declare the function more than once by including the header multiple times.
#define compute_state_H

#include <RcppArmadillo.h>
#include "some_function.h"
#include "forward_backward.h"
#include "compute_loglikelihood.h"

using namespace arma;
using namespace Rcpp;
using namespace std;

arma::mat compute_state(arma::vec &delta,
                        arma::vec &Y,
                        arma::mat &A,
                        arma::mat &B,
                        arma::mat &X,
                        char &family);

#endif
