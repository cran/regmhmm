% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/RcppExports.R
\name{backward}
\alias{backward}
\title{Probability Calculation using the Backward Algorithm for Hidden Markov Models}
\usage{
backward(delta, Y, A, B, X, family)
}
\arguments{
\item{delta}{a vector of length S specifying the initial probabilities.}

\item{Y}{a vector of observations of size T.}

\item{A}{a matrix of size S x S specifying the transition probabilities.}

\item{B}{a matrix of size S x (p + 1) specifying the GLM parameters of the emission probabilities.}

\item{X}{a design matrix of size T x p.}

\item{family}{the family of the response.}
}
\value{
A matrix of size S x T that is the backward probabilities in log scale.
}
\description{
Calculate the probability given parameters of a hidden Markov model utilizing the backward algorithm.
}
\examples{
# Example usage of the function
parameters_setting <- list()
parameters_setting$emis_mat <- matrix(NA, nrow = 2, ncol = 4)
parameters_setting$emis_mat[1, 1] <- 0.1
parameters_setting$emis_mat[1, 2] <- 0.5
parameters_setting$emis_mat[1, 3] <- -0.75
parameters_setting$emis_mat[1, 4] <- 0.75
parameters_setting$emis_mat[2, 1] <- -0.1
parameters_setting$emis_mat[2, 2] <- -0.5
parameters_setting$emis_mat[2, 3] <- 0.75
parameters_setting$emis_mat[2, 4] <- 1
parameters_setting$trans_mat <- matrix(NA, nrow = 2, ncol = 2)
parameters_setting$trans_mat[1, 1] <- 0.65
parameters_setting$trans_mat[1, 2] <- 0.35
parameters_setting$trans_mat[2, 1] <- 0.2
parameters_setting$trans_mat[2, 2] <- 0.8
parameters_setting$init_vec <- c(0.65, 0.35)
simulated_data <- simulate_HMM_data(
  seed_num = 1,
  p_noise = 7,
  N = 100,
  N_persub = 10,
  parameters_setting = parameters_setting
)
backward_C <- backward(
  delta = parameters_setting$init_vec,
  Y = simulated_data$y_mat[1, ],
  A = parameters_setting$trans_mat,
  B = parameters_setting$emis_mat,
  X = simulated_data$X_array[, 1:4, 1],
  family = "P"
)

}
