#' @title Print Outputs from a Hidden Markov Model (HMM)
#'
#' @description
#' Display detailed summary outputs and relevant information derived from a Hidden Markov Model (HMM) object. This includes state-specific parameters, transition probabilities, log-likelihood, and other essential metrics, providing an overview of the fitted model.
#'
#' @param x an object used to select a method.
#' @param ... further arguments passed to or from other methods.
#'
#' @returns
#' Return a invisible copy of "HMM" object
#' 
#' @method print HMM
#' @examples
#' \donttest{
#' # Example usage of the function
#' seed_num <- 1
#' p_noise <- 2
#' N <- 100
#' N_persub <- 10
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
#' HMM_fit <- HMM(delta=as.matrix(init_start),
#'                Y_mat=simulated_data$y_mat,
#'                A=trans_start,
#'                B=emis_start,
#'                X_cube=simulated_data$X_array,
#'                family="P",
#'                eps=1e-4,
#'                trace = 0
#' )
#' print(HMM_fit)
#' }
#' 
#' @export
print.HMM <- function(x, ...) {
  UseMethod("print")
  trans_mat <- HMM$A_hat
  emiss_mat <- HMM$B_hat
  init_vec <- as.numeric(HMM$delta_hat)
  log_likelihood <- HMM$log_likelihood

  num_state <- length(HMM$delta_hat)
  row.names(emiss_mat) <- row.names(trans_mat) <-
    colnames(trans_mat) <- names(init_vec) <- paste0("State ", 1:num_state)

  cat("Initial Vector:\n")
  print(init_vec)
  cat("\nTransition Matrix:\n")
  print(trans_mat)
  cat("\nEmission Matrix:\n")
  print(emiss_mat)
  cat("\nLog-likelihood:\n")
  print(log_likelihood)
  
  invisible(x)
}
