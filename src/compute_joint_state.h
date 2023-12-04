#ifndef compute_joint_state_H
#define compute_joint_state_H

#include <RcppArmadillo.h>
#include "some_function.h"
#include "forward_backward.h"
#include "compute_loglikelihood.h"

using namespace arma;
using namespace Rcpp;
using namespace std;

arma::cube compute_joint_state(arma::vec &delta,
                               arma::vec &Y,
                               arma::mat &A,
                               arma::mat &B,
                               arma::mat &X,
                               char &family);

#endif
