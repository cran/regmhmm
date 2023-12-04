#' @title Single Iteration of EM Algorithm for Fitting Regularized Hidden Markov Model (rHMM)
#' 
#' @description
#' Execute a single iteration of the Expectation-Maximization (EM) algorithm designed for fitting a regularized Hidden Markov Model (rHMM).
#' 
#' @param delta a vector of length S specifying the initial probabilities.
#' @param A a matrix of size S x S specifying the transition probabilities.
#' @param B a matrix of size S x (p + 1) specifying the GLM parameters of the emission probabilities.
#' @param Y_mat a matrix of observations of size N x T.
#' @param X_cube a design array of size T x p x N.
#' @param family the family of the response.
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
#' rHMM_one_step <- rHMM_one_step(delta=as.matrix(init_start),
#'                                Y_mat=simulated_data$y_mat,
#'                                A=trans_start,
#'                                B=emis_start,
#'                                X_cube=simulated_data$X_array,
#'                                family="P",
#'                                omega_cva=sqrt(sqrt(seq(0, 1, len = 5))),
#'                                trace = 0)
#' }
#'
#' @export
rHMM_one_step <- function(
    delta, Y_mat, A, B, X_cube,
    family, omega_cva = sqrt(sqrt(seq(0, 1, len = 5))),
    trace = 0) {
  # some marcos
  N_persub <- dim(Y_mat)[2]
  S <- length(delta)
  N <- dim(Y_mat)[1]
  p <- dim(X_cube)[2]
  
  ## working Y and X
  Y <- matrix(NA, nrow = N_persub, ncol = 1)
  X <- matrix(NA, nrow = N, ncol = p)
  
  ## flatten Y and X
  Y_long <- matrix(NA, nrow = N * N_persub, ncol = 1)
  X_long <- matrix(NA, nrow = N * N_persub, ncol = p)
  
  ## long gamma
  gamma_mat_wide <- matrix(NA, nrow = S, ncol = N * N_persub)
  working_gamma <- matrix(NA, nrow = N * N_persub, ncol = 1)
  
  # recivers
  gamma_slice <- matrix(NA, nrow = S, ncol = N_persub)
  get_forward_backward <- list()
  xi <- array(NA, dim = c(S, S, N_persub))
  
  # A, B, delta
  A_hat <- matrix(NA, nrow = S, ncol = S)
  B_hat <- matrix(NA, nrow = S, ncol = p)
  delta_hat <- matrix(NA, nrow = S, ncol = 1)
  
  # objects to compute A, B, delta
  numerator_xi_mat <- matrix(NA, nrow = S, ncol = S)
  denomiator_xi_mat <- matrix(NA, nrow = S, ncol = S)
  numerator_gamma_mat <- matrix(NA, nrow = S, ncol = 1)
  denomiator_gamma_mat <- matrix(NA, nrow = S, ncol = 1)
  init_beta <- matrix(NA, nrow = p, ncol = 1)
  
  # get best alpha and lambda
  best_alpha_vec <- rep(NA, S)
  best_lambda_vec <- rep(NA, S)
  
  # small things
  rec_beta <- matrix(NA, nrow = p, ncol = 1)
  ll <- 0
  
  # glmnet family
  glmnet_family <- NULL
  if (family == "P") {
    glmnet_family <- "poisson"
  } else if (family == "D") {
    glmnet_family <- "binomial"
  }
  
  # flatten X_cube and Y_mat
  for (i in 1:N)
  {
    for (t in 1:N_persub)
    {
      for (j in 1:p)
      {
        X_long[(i - 1) * N_persub + t, j] <- X_cube[t, j, i]
      }
      Y_long[(i - 1) * N_persub + t] <- Y_mat[i, t]
    }
  }
  
  # initialize
  for (s in 1:S)
  {
    delta_hat[s] <- 0
    for (u in 1:S)
    {
      A_hat[s, u] <- 0
      numerator_xi_mat[s, u] <- 0
      denomiator_xi_mat[s, u] <- 0
    }
    for (i in 1:p)
    {
      B_hat[s, i] <- 0
    }
    numerator_gamma_mat[s, 0] <- 0
    denomiator_gamma_mat[s, 0] <- 0
    for (t in 1:N_persub)
    {
      gamma_slice[s, t] <- 0
    }
  }
  
  # E step
  for (i in 1:N)
  {
    Y <- t(t(Y_mat[1, ])) # column factor
    X <- X_cube[, , i]
    xi <- compute_joint_state(delta, Y, A, B, X, family)
    gamma_slice <- compute_state(delta, Y, A, B, X, family)
    for (s in 1:S)
    {
      for (t in 1:N_persub)
      {
        gamma_mat_wide[s, (i - 1) * N_persub + t] <- gamma_slice[
          s,
          t
        ]
        if (t == N_persub) {
          next
        } # don't want xi to add the finally iteration
        for (u in 1:S)
        {
          numerator_xi_mat[s, u] <- numerator_xi_mat[s, u] + xi[
            s,
            u, t
          ]
          for (k in 1:S)
          {
            denomiator_xi_mat[s, u] <- denomiator_xi_mat[s, u] +
              xi[s, k, t]
          }
        }
      }
    }
    delta_hat <- delta_hat + gamma_slice[, 1]
  }
  
  # M step for A
  for (s in 1:S)
  {
    for (u in 1:S)
    {
      A_hat[s, u] <- numerator_xi_mat[s, u] / denomiator_xi_mat[
        s,
        u
      ]
    }
  }
  
  # for delta
  delta_hat <- delta_hat / N
  
  # for B
  for (s in 1:S)
  {
    working_gamma <- t(t(gamma_mat_wide[s, ]))
    init_beta <- t(t(B[s, ]))

    # deal with extemely small weights
    if (sum(is.nan(working_gamma)) !=
        0) {
      working_gamma[is.nan(working_gamma), ] <- 0
    }
    if (sum(working_gamma == 0) !=
        0) {
      zero_ind <- working_gamma == 0
      working_gamma[zero_ind, ] <- 1e-300
    }
    if(any(is.nan(working_gamma[,1])) || any(is.na(working_gamma[,1]))){
      nan_gamma_ind <- which(is.nan(working_gamma[,1]))
      working_gamma[nan_gamma_ind,1] <- 1e-300

      #print(na_gamma_ind)
      na_gamma_ind <- which(is.na(working_gamma[,1]))
      working_gamma[na_gamma_ind,1] <- 1e-300
    }
    
    # if (any(working_gamma[, 1] == 0) || any(is.na(working_gamma[, 
    #                                                             1]))) {
    #   small_gamma_ind <- which(working_gamma[, 1] < 1e-07)
    #   na_gamma_ind <- which(is.na(working_gamma[, 1]))
    #   working_gamma[small_gamma_ind, 1] <- 1e-06
    # }
    
    # if(any(working_gamma[,1]==0) || any(is.na(working_gamma[,1]))){
    #   small_gamma_ind <- which(working_gamma[,1]<1e-7)
    #   working_gamma[small_gamma_ind,1] <- 0.000001
    #   
    #   nan_gamma_ind <- which(is.nan(working_gamma[,1]))
    #   working_gamma[nan_gamma_ind,1] <- 0.000000001
    #   
    #   #print(na_gamma_ind)
    #   na_gamma_ind <- which(is.na(working_gamma[,1]))
    #   working_gamma[na_gamma_ind,1] <- 0.000000001
    # }
    
    cva_glmnet_get <- glmnetUtils::cva.glmnet(
      x = X_long, y = Y_long, alpha = omega_cva, family = glmnet_family,
      weights = working_gamma, intercept = TRUE, standardize = TRUE
    )
    
    cvm_1se_for_all_alpha <- rep(NA, length = length(cva_glmnet_get$modlist))
    for (i in seq(1,length(cva_glmnet_get$modlist)))
    {
      cvm_1se_for_all_alpha[i] <- cva_glmnet_get$modlist[[i]]$cvm[
        cva_glmnet_get$modlist[[i]]$index[2]]
    }
    best_alpha <- cva_glmnet_get$alpha[which.min(cvm_1se_for_all_alpha)]
    
    best_alpha_vec[s] <- best_alpha
    
    if (trace == 1) {
      message("Best alpha is: ", best_alpha, "\n")
    }
    
    # make sure things are correct so that rerun using cv.glmnet
    # with the best best_alpha
    rec_cv_glmnet <- glmnet::cv.glmnet(
      x = X_long, y = Y_long, family = glmnet_family, weights = working_gamma,
      alpha = best_alpha, intercept = TRUE, standardize = TRUE
    )
    
    rec_beta <- rec_cv_glmnet$glmnet.fit$beta[, rec_cv_glmnet$index[2]]
    
    best_lambda <- rec_cv_glmnet$lambda[rec_cv_glmnet$index[2]]
    
    best_lambda_vec[s] <- best_lambda
    
    for (a in 1:p)
    {
      B_hat[s, a] <- rec_beta[a]
    }
  }
  
  # for log-likelihood
  for (i in 1:N)
  {
    Y <- t(t(Y_mat[i, ]))
    X <- X_cube[, , i]
    if (is.nan(compute_loglikelihood(delta_hat, Y, A_hat, B_hat, X, family))) {
      if (trace == 1) {
        message("NaN for subject: ", i, ", skip.\n")
      }
      next
    } else {
      ll <- ll + compute_loglikelihood(delta_hat, Y, A_hat, B_hat, X, family)
    }
  }
  
  rec_return <- vector(mode = "list", length = 4)
  rec_return[[1]] <- delta_hat
  rec_return[[2]] <- A_hat
  rec_return[[3]] <- B_hat
  rec_return[[4]] <- ll
  rec_return[[5]] <- best_lambda_vec
  rec_return[[6]] <- best_alpha_vec
  
  names(rec_return) <- c("delta_hat", "A_hat", "B_hat", "log_likelihood", "lambda", "omega")
  
  return(rec_return)
}