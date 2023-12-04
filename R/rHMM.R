#' @importFrom glmnetUtils cva.glmnet
#'
#' @title Fit Regularized Hidden Markov Models (rHMM) with Modified CCD
#' @description
#' Utilize the modified Cyclic Coordinate Descent (CCD) algorithm to effectively fit a regularized Hidden Markov Model (rHMM).
#' 
#' @param delta a vector of length S specifying the initial probabilities.
#' @param A a matrix of size S x S specifying the transition probabilities.
#' @param B a matrix of size S x (p + 1) specifying the GLM parameters of the emission probabilities.
#' @param Y_mat a matrix of observations of size N x T.
#' @param X_cube a design array of size T x p x N.
#' @param family the family of the response.
#' @param eps convergence tolerance.
#' @param N_iter the maximal number of the EM algorithm for fitting HMM.
#' @param trace logical indicating if detailed output should be produced during the fitting process.
#' @param omega_cva a vector of omega values for the modified cyclical coordinate descent algorithm used for cross-validation.
#'
#' @returns
#' A list object with the following slots:
#' 
#' \item{delta_hat}{the estimate of delta.}
#'
#' \item{A_hat}{the estimate of A.}
#'
#' \item{B_hat}{the estimate of B.}
#' 
#' \item{log_likelihood}{the log-likelihood of the model.}
#' 
#' \item{lambda}{lambda from CV.}
#' 
#' \item{omega}{omega from CV.}
#' 
#' @examples
#' \donttest{
#' # Example usage of the function
#' seed_num <- 1
#' p_noise <- 2
#' N <- 100
#' N_persub <- 50
#' parameters_setting <- list(
#'   init_vec = c(0.5, 0.5),
#'   trans_mat = matrix(c(0.7, 0.3, 0.2, 0.8), nrow = 2, byrow = TRUE),
#'   emis_mat = matrix(c(1, 0.5, 0.5, 2), nrow = 2, byrow = TRUE)
#' )
#' simulated_data <- simulate_HMM_data(seed_num, p_noise, N, N_persub, parameters_setting)
#' 
#' init_start = c(0.5, 0.5)
#' trans_start = matrix(c(0.5, 0.5, 0.5, 0.5), nrow = 2)
#' emis_start = matrix(rep(1, 8), nrow = 2)
#' 
#' rHMM_one_step <- rHMM(delta=as.matrix(init_start),
#'                                Y_mat=simulated_data$y_mat,
#'                                A=trans_start,
#'                                B=emis_start,
#'                                X_cube=simulated_data$X_array,
#'                                family="P",
#'                                omega_cva=sqrt(sqrt(seq(0, 1, len = 5))),
#'                                N_iter=10,
#'                                trace = 0)
#'}
#'
#' @export
rHMM <- function(
    delta, Y_mat, A, B, X_cube, family,
    omega_cva = sqrt(sqrt(seq(0, 1, len = 5))),
    N_iter = 1000, eps = 1e-07, trace = 0) {
  S <- length(delta)
  p <- dim(B)[2]
  # run once for initialization A, B, delta
  A_hat <- matrix(NA, nrow = S, ncol = S)
  B_hat <- matrix(NA, nrow = S, ncol = p)
  delta_hat <- matrix(NA, nrow = S, ncol = 1)
  ll_old <- ll_new <- 0
  best_lambda_vec <- rep(NA, 0)
  best_alpha_vec <- rep(NA, 0)
  
  # initialize the list with 1 run
  get_EM_one <- rHMM_one_step(
    delta = delta, Y_mat = Y_mat, A = A, B = B,
    X_cube = X_cube, family = family,
    omega_cva = omega_cva, trace = trace
  )
  
  A_hat <- get_EM_one[["A_hat"]]
  B_hat <- get_EM_one[["B_hat"]]
  delta_hat <- get_EM_one[["delta_hat"]]
  ll_old <- get_EM_one[["log_likelihood"]]
  
  for (i in 1:N_iter)
  {
    get_EM_one <- rHMM_one_step(
      delta = delta_hat, Y_mat = Y_mat, A = A_hat, B = B_hat, X_cube = X_cube,
      omega_cva = omega_cva, family = family, trace = trace
    )
    
    best_lambda_vec <- get_EM_one[["lambda"]]
    best_alpha_vec <- get_EM_one[["omega"]]
    
    A_hat <- get_EM_one[["A_hat"]]
    B_hat <- get_EM_one[["B_hat"]]
    delta_hat <- get_EM_one[["delta_hat"]]
    ll_new <- get_EM_one[["log_likelihood"]]
    
    if (trace == 1) {
      message(ll_new, "\n")
    }
    
    if ((abs(ll_old - ll_new) / abs(ll_new)) <
        eps) {
      if (trace == 1) {
        message(
          "EM: Converged for objective function", " it takes ", i,
          " steps to converge. Break\n"
        )
      }
      break
    }
    ll_old <- ll_new
    
    if (i >= N_iter) {
      if (trace == 1) {
        message("EM: Reach the maximum steps \n")
      }
      break
    }
    B_hat_new <- B_hat
  }
  rec_return <- vector(mode = "list", length = 4)
  rec_return[[1]] <- delta_hat
  rec_return[[2]] <- A_hat
  rec_return[[3]] <- B_hat
  rec_return[[4]] <- ll_new
  rec_return[[5]] <- best_lambda_vec
  rec_return[[6]] <- best_alpha_vec
  
  names(rec_return) <- c("delta_hat", "A_hat", "B_hat", "log_likelihood", "lambda", "omega")
  
  return(rec_return)
}
