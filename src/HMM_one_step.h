#ifndef HMM_ONE_STEP_H
#define HMM_ONE_STEP_H

#include <RcppArmadillo.h>
#include "compute_joint_state.h"
#include "compute_state.h"
#include "IRLS_EM.h"

using namespace arma;
using namespace Rcpp;
using namespace std;

List HMM_one_step(arma::vec &delta,
                  arma::mat &Y_mat, // N*T
                  arma::mat &A,
                  arma::mat &B,
                  arma::cube &X_cube, // T*p*N
                  char &family,
                  double eps_IRLS,
                  int max_N_IRLS);
#endif
