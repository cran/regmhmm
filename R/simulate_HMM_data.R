#' @importFrom MASS mvrnorm
#' @importFrom stats rpois
#' 
#' @title Simulate Hidden Markov Model (HMM) Data
#'
#' @description
#' Generate synthetic HMM data for testing and validation purposes. This function creates a simulated dataset with specified parameters, including initial probabilities, transition probabilities, emission matrix, and noise covariates.
#'
#' @param seed_num Seed for reproducibility.
#' @param p_noise Number of noise covariates.
#' @param N Number of subjects.
#' @param N_persub Number of time points per subject.
#' @param parameters_setting A list containing the parameters for the HMM.
#' @return A list containing the design matrix (X_array) and response variable matrix (y_mat).
#'
#' @examples
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
#' @export
simulate_HMM_data <- function(seed_num,
                              p_noise,
                              N,
                              N_persub,
                              parameters_setting) {
  set.seed(seed_num)
  # Simulation
  
  ## Notation:
  ### S = # states
  ### p = # covariates including
  
  ## macro
  S <- dim(parameters_setting$emis_mat)[1]
  p <- dim(parameters_setting$emis_mat)[2] - 1 # first one is always intercept
  
  ### initial probability
  delta_true <- parameters_setting$init_vec
  
  ### transitional probability
  #### (S x S: row -> column i.e. [1,2] => From state 1 to state 2.
  #### That means row sums should be 1)
  trans_mat_true <- parameters_setting$trans_mat
  
  ### covariates: emission matrix
  #### (S x p: covariates are different across states)
  emiss_mat_true <- parameters_setting$emis_mat
  
  ### true hidden state
  #### (hid_state_true: N x T(N_persub). Same dim as y_mat)
  hid_state_true <- matrix(NA, nrow = N, ncol = N_persub)
  for (i in 1:N) {
    # initial states, following delta_true
    hid_state_true[i, 1] <- sample(1:S, 1, prob = delta_true)
    # other states, following initial state and transition matrix
    for (j in 2:N_persub) {
      previous_state <- hid_state_true[i, j - 1]
      # the new state should based on the previous state
      hid_state_true[i, j] <- sample(1:S, 1,
                                     prob = trans_mat_true[previous_state, ])
    }
  }
  
  ### design matrix
  #### for each subject
  gen_data_each_subject <- function(sub_i) {
    X <- matrix(NA, nrow = N_persub, ncol = p_noise + p + 1) #+1 for intercept
    X[, 1] <- 1
    pho <- matrix(NA, nrow = p_noise + p, ncol = p_noise + p)
    pho[] <- 0.5
    diag(pho) <- 1
    
    D_diag <- sqrt(diag(1, nrow = p_noise + p, ncol = p_noise + p))
    sigma_noise <- D_diag %*% pho %*% D_diag
    X[, 2:(p_noise + p + 1)] <- mvrnorm(N_persub,
                                              mu = rep(0, p_noise + p), Sigma = sigma_noise)
    
    # Z <- matrix(NA, nrow=N_persub, ncol=1)
    # Z[,1] <- 1
    
    ### observation
    #### half are State 1 or State 2
    ##### eta
    eta_true <- rep(NA, N_persub)
    beta_s1 <- as.matrix(emiss_mat_true[1, ])
    beta_s2 <- as.matrix(emiss_mat_true[2, ])
    
    for (j in 1:N_persub) {
      if (hid_state_true[sub_i, j] == 1) {
        # eta_true[j] <- X[j,]%*%beta_s1 + rnorm(1, sd=sqrt(0.1))
        # +1 is for the intercept
        eta_true[j] <- X[j, 1:(p + 1)] %*% beta_s1
      } else if (hid_state_true[sub_i, j] == 2) {
        # eta_true[j] <- X[j,]%*%beta_s2 + rnorm(1, sd=sqrt(0.1))
        eta_true[j] <- X[j, 1:(p + 1)] %*% beta_s2
      }
    }
    
    ##### mu
    mu_true <- exp(eta_true)
    
    ##### y
    ###### adding some random noise for mu?
    y <- rpois(N_persub, lambda = mu_true)
    
    return_list <- list()
    return_list[["X"]] <- X
    return_list[["y"]] <- y
    # return_list[["Z"]] <- Z
    
    return(return_list)
  }
  
  ### combine all subject's data
  #### X is a N x T(N_persub) x p cube
  #### y_mat is a N x T(N_persub) matrix
  X_array <- array(NA, dim = c(N_persub, p + p_noise + 1, N)) #+1 for intercept
  y_mat <- matrix(NA, nrow = N, ncol = N_persub)
  # Z_array <- array(NA, dim=c(N_persub, 1, N))
  for (i in 1:N) {
    get_data_iter <- gen_data_each_subject(i)
    X_array[, , i] <- get_data_iter$X
    # Z_array[,,i] <- get_data_iter$Z
    y_mat[i, ] <- get_data_iter$y
  }
  
  
  # sim_dat_list <- vector(mode="list", length = 3)
  sim_dat_list <- vector(mode = "list", length = 2)
  # need the one with noise
  # sim_dat_list[[1]] <- X_array
  sim_dat_list[[1]] <- X_array
  # sim_dat_list[[2]] <- Z_array
  sim_dat_list[[2]] <- y_mat
  
  names(sim_dat_list) <- c(
    "X_array",
    # "Z_array",
    "y_mat"
  )
  
  return(sim_dat_list)
}

