import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import warnings

################################################################
# Optimization statistical engine function.
################################################################
verbose = True

def information_mat(A, W):
    return A.T @ np.linalg.inv(W) @ A
    #return A.T @ W @ A

def D_opt(A, W):
    #return np.linalg.det(np.linalg.inv(information_mat(A, W)))
    return np.linalg.det(information_mat(A, W))
 
 
def A_opt(A, W):
    inf_mat = information_mat(A, W)
    try:
        inv_inf_mat = np.linalg.inv(inf_mat)
    except np.linalg.LinAlgError:
        return -np.inf
    else:
        return -np.sum(np.diag(inv_inf_mat))
        

def generate_seed_design(A_use, W_use, n_pairs):
    # Find a non-random connected seed design.
    # If it has taken too many iterations to find a random seed design,
    # make a radial design and add the number of edges needed from highest
    # weights available.
    reference_row = np.where(np.sum(np.abs(A_use), axis=1) == 1)[0]
    # If there is more than one reference ligand, use the first one.
    if len(reference_row) > 1:
        reference_row = reference_row[0]
    reference_col = np.where(A_use[:, np.where(A_use[reference_row, :] == 1)[1]] != 0)[0]
    # Get the rows of A that are nonzero in the column of reference_col.
    radial_rows = reference_col[reference_col != reference_row]

    # Get extra rows to add if needed:
    if len(radial_rows) < n_pairs:  # Find highest weight edges to add.
        n_nodes = A_use.shape[1]
        edges_needed = n_pairs - n_nodes + 1

        # Find the highest weight rows of W, excluding radial_rows
        W_considered = W_use[np.ix_(~np.isin(np.arange(W_use.shape[0]), np.concatenate(([reference_row], radial_rows))), ~np.isin(np.arange(W_use.shape[1]), np.concatenate(([reference_row], radial_rows))))]

        # Get the row numbers from W_use
        all_cols = np.arange(W_use.shape[1])
        exclude_cols = np.concatenate(([reference_row], radial_rows))
        remaining_cols = all_cols[~np.isin(all_cols, exclude_cols)]

        # Find the indexes of highest values of W_considered.
        top_edges_ix = np.argsort(W_considered, axis=None)[-edges_needed:]

        # Convert index to column or row.
        top_cols = top_edges_ix % W_considered.shape[1]
        top_cols[top_cols == 0] = W_considered.shape[1]

        # top_cols is the index of entries in remaining_cols.
        top_W_cols = remaining_cols[top_cols]

        current_rows = np.sort(np.concatenate(([reference_row], radial_rows, top_W_cols)))
        print("Generated seed design from radial map and high weight edges.")
    else:  # Output star graph (radial).
        current_rows = np.sort(np.append(reference_row, radial_rows))
        
        if verbose:
            print(f'The rows of the deterministic seed design are: {current_rows}')
        
        print("Generated seed design from radial map.")

    return current_rows



def random(A, W, starting_rows):
    #n_pairs = len(starting_rows[np.logical_not(np.isin(starting_rows, np.where(np.sum(np.abs(A_use), axis=1) == 1)))])
    n_pairs = len(starting_rows) - 1
    rand_iter = 1
    print('Searching for seed design ...')
    max_iterations = 1
    while rand_iter < max_iterations:
        reference_row = np.where(np.sum(np.abs(A_use), axis=1) == 1.)[0][0]
        # Select random rows from possible rows without duplicaton
        random_rows = np.random.choice(np.setdiff1d(np.arange(A_use.shape[0]), reference_row), size=n_pairs, replace=False)
        current_rows = np.append(random_rows, reference_row)
        
        if verbose:
            print(f'Current rows: {current_rows}')

        #if np.where(np.sum(np.abs(A_use), axis=1) == 1) is not None:
            #print(A_use)
            #n_pairs = len(starting_rows[np.logical_not(np.isin(starting_rows, np.where(np.sum(np.abs(A_use), axis=1) == 1)))])
            #print(n_pairs)
            
        A = A_use[current_rows, :]
        W = W_use[current_rows, :][:, current_rows]
        
        if verbose:
            print(f'The weight matrix, W: \n {W}')
        
        # How should I avoid this warning: RuntimeWarning: divide by zero encountered in log
        crit_i = np.log(np.linalg.det(information_mat(A, W)))
        
        if verbose:
            print(f'crit_i: {crit_i}')
        
        
        if np.isfinite(crit_i) and not np.allclose(crit_i, -np.inf):
            rand_iter += 1
            try:
                np.linalg.inv(information_mat(A, W))
            except np.linalg.LinAlgError:
                pass
            else:
                print(f"Invertible seed design found. Iteration = {rand_iter}. Criterion = {crit_i:.4f}")
                break

    if rand_iter >= max_iterations:
        current_rows = generate_seed_design(A_use, W_use, n_pairs)
        A = A_use[current_rows, :]
        W = W_use[current_rows, :][:, current_rows]
        crit_i = np.log(np.linalg.det(information_mat(A, W)))
        
    return {
            'A': A_use[current_rows, :],
            'W': W_use[np.ix_(current_rows, current_rows)],
            'criterion': crit_i,
            'rows': current_rows
            }


class CriticalFunc:
    def __init__(self, A, W, starting_rows=None):
        self.A = np.array(A)
        self.W = np.array(W)
        self.starting_rows = np.array(starting_rows) if starting_rows is not None else None


    def calculate(self, criterion='D'):
        if criterion == 'D':
            result = D_opt(self.A, self.W)
        elif criterion == 'A':
            result = A_opt(self.A, self.W)
        elif criterion == 'random':
            if self.starting_rows is None:
                raise ValueError("Starting rows not provided for 'random' criterion.")
            result = random(self.A, self.W, self.starting_rows)
        else:
            raise ValueError("Invalid criterion. Supported values are 'D', 'A', and 'random'.")
        return result


def return_numeric(A, W, criterion, starting_rows=None):
    calculator = CriticalFunc(A, W, starting_rows)
    result = calculator.calculate(criterion = criterion)
    
    if criterion == 'random':
        crit_i = result['criterion']
    elif criterion in ['A', 'D']:
        crit_i = result
    return crit_i
        

def fedorov_exchange(A_use, W_use, criterion, starting_rows, maxiter=1e4):
    # Define the functions which compute the optimality criteria
    '''
    if criterion == 'D':  # D-optimal (minimizes determinant of the information matrix)
        crit_val = D_opt(np.array(A_use), np.array(W_use))
    elif criterion == 'A':
        crit_val = A_opt(np.array(A_use), np.array(W_use))
    elif criterion == 'random':
        result = random(A_use, W_use, starting_rows)
        crit_i = result['criterion']
    else:
        raise ValueError("Invalid criterion. Supported values are 'D', 'A', and 'random'.")
    #print(crit_val)
      # Return the optimal design, weight matrix, criteria, and rows
    '''
    # Initialize the design matrix
    current_rows = starting_rows
    print("CURRENT ROWS")
    print(current_rows)
    # Keep track of iterations
    current_iter = 1

    while True:
        # Compute the criterion for the current design
        #current_D = crit_func(A_use[current_rows], W_use[current_rows][:, current_rows])
        A = A_use[current_rows, :]
        W = W_use[current_rows, :][:, current_rows]
        
        # Calculate the current critical values.
        current_D = return_numeric(A, W, criterion)

        # Prevent an error that occurs for designs w/ n choose 2 edges.
        if len(current_rows) == A_use.shape[0]:
            print("All possible designs have been sampled.")
            break

        print(f"Iteration {current_iter} (criterion = {current_D:.4f})")

        # Find all possible couples of current rows and potential rows
        candidate_rows = np.setdiff1d(np.arange(A_use.shape[0]), current_rows)
        couples = np.array(np.meshgrid(current_rows, candidate_rows)).T.reshape(-1, 2)
        couples = np.column_stack((couples, np.zeros(couples.shape[0])))

        # Function for making A matrices
        def gen_A(val):
            rows = np.append(current_rows[current_rows != couples[val, 0]], couples[val, 1])
            return A_use[rows, :]

        # Function for making W matrices
        def gen_W(val):
            rows = np.append(current_rows[current_rows != couples[val, 0]], couples[val, 1])
            return W_use[rows, :][:, rows]

        # Currently for loop form
        vals = np.arange(couples.shape[0]).astype(int)
        c = couples.astype(float)
        for val in vals:
            c[val, 2] = return_numeric(gen_A(val), gen_W(val), criterion)

        # Calculate difference
        diff = c[:, 2] - current_D

        # If no exchanges improve the design, stop. Otherwise, make the best swap and continue the algorithm
        if np.all(diff <= 0):
            break
        elif current_iter == maxiter:
            print("The maximum number of iterations occurred.")
            break
        else:
            max_diff_idx = np.argmax(couples[:, 3])
            current_rows = np.append(current_rows[current_rows != c[max_diff_idx, 0]], c[max_diff_idx, 1])
            current_iter += 1

    # Return the optimal design, weight matrix, criterion, and rows
    return {
        'A': A_use[current_rows],
        'W': W_use[np.ix_(current_rows, current_rows)],
        'criterion': current_D,
        'rows': current_rows
        }



######  Testing modules  ##################################################################

def read_latex_mat(latex_matrix: str):
    #Extract the matrix content from the LaTeX string
    matrix_content = latex_matrix.strip('\begin{bmatrix}').strip('\end{bmatrix}').strip()
    # Split the matrix rows and elements
    rows = matrix_content.split('\\')
    matrix_elements = [row.split('&') for row in rows]
    # Convert elements to integers
    matrix = np.array([[float(element) for element in row] for row in matrix_elements])
    return matrix

# Read the LaTeX graph file
#with open('graph.tex', 'r') as file:
    #latex_code = file.read()
latex_code = '\begin{bmatrix}1&1&1&0\\-1&0&0&0\\0&-1&0&1\\0&0&-1&-1\\\end{bmatrix}'
W_latex = '\begin{bmatrix}0.7&0&0&0&0\\0&0.5&0&0&0\\0&0&0.3&0&0\\0&0&0&0.8&0\\0&0&0&0&2.0\\\end{bmatrix}'


criterion = 'D'

# Read latex code into numpy matrix.
incidence_mat = read_latex_mat(latex_code).T
newrow = [1, 0, 0, 0]
A_use = np.vstack([incidence_mat, newrow])
print(f'A_use.shape = {A_use.shape}')

W_use = read_latex_mat(W_latex)
#print(W_use)

C = A_use.T @ np.linalg.inv(W_use) @ A_use
print("C \n")
print(C)

k_numb = 3
print("TEST")
print(list(range(1, k_numb+1)).append(A_use.shape[0]))


#result = fedorov_exchange(A_use, W_use, 'A', list(range(1, k_numb+1)).append(A_use.shape[0]))
result = fedorov_exchange(A_use, W_use, 'A', [1, 2, 3, 4])
print(result)

current_rows = [1, 2, 3, 4]

A = A_use[current_rows, :]
W = W_use[current_rows, :][:, current_rows]
C = A.T @ np.linalg.inv(W) @ A
print("C \n")
print(C)


Latest traceback issue:
'''
Traceback (most recent call last):
  File "/Users/mpitman/work/himap/HiMap_dev/himap/optimal_design_rewrite.py", line 292, in <module>
    result = fedorov_exchange(A_use, W_use, 'A', [1, 2, 3, 4])
  File "/Users/mpitman/work/himap/HiMap_dev/himap/optimal_design_rewrite.py", line 226, in fedorov_exchange
    c[val, 2] = return_numeric(gen_A(val), gen_W(val), criterion)
  File "/Users/mpitman/work/himap/HiMap_dev/himap/optimal_design_rewrite.py", line 214, in gen_A
    rows = np.append(current_rows[current_rows != couples[val, 0]], couples[val, 1])
TypeError: only integer scalar arrays can be converted to a scalar index
'''


'''
#starting_rows = result['rows']

current_rows = np.array([0, 1, 3, 4])
remaining_rows = current_rows[current_rows != 4]
print(remaining_rows)

A = A_use[current_rows, :]
W = W_use[current_rows, :][:, current_rows]

# Calculate the current critical values.
current_D = return_numeric(A, W, criterion)

# Find all possible couples of current rows and potential rows
candidate_rows = np.setdiff1d(np.arange(A_use.shape[0]), current_rows)
print(f'np.arange(A_use.shape[0])  {np.arange(A_use.shape[0])}')
print(f' candidate_rows {candidate_rows}')

couples = np.array(np.meshgrid(current_rows, candidate_rows)).T.reshape(-1, 2).astype(int)
print(f'couples: \n {couples}')

couples = np.column_stack((couples, np.zeros(couples.shape[0]))).astype(int)
print(f'couples: \n {couples}')

vals = np.arange(couples.shape[0])
print(f'vals {vals}')

# Function for making A matrices
def gen_A(val):
    rows = np.append(current_rows[current_rows != couples[val, 0]], couples[val, 1])
    return A_use[rows, :]

# Function for making W matrices
def gen_W(val):
    rows = np.append(current_rows[current_rows != couples[val, 0]], couples[val, 1])
    return W_use[rows, :][:, rows]
    


c = couples.astype(float)
for val in vals:
    c[val, 2] = return_numeric(gen_A(val), gen_W(val), criterion)
print(c)
'''








'''
  # Controls the optimization iterations, below.
  # Initialize the design matrix
  current.rows <- starting.rows
  # Keep track of iterations
  current.iter <- 1
  # Run the exchange algorithm
  repeat{
    # Compute the criterion for the current design
    current.D <- crit.func( A.use[ current.rows,], W.use[ current.rows, current.rows])
    
    # Prevent an error that occurs for designs w/ n choose 2 edges.
    if( length( current.rows) == dim(A.use)[1]) {
      print("All possible designs have been sampled.")
      break
    }
    print(paste0( 'Iteration ', current.iter, ' (criterion = ', signif( current.D, 4), ')'))
        
    # Find all possible couples of current rows and potential rows
    candidate.rows <- setdiff( 1:nrow( A.use), current.rows)
    couples <- expand.grid( current = current.rows, candidate = candidate.rows, D = NA)
    
    # Func for making A matrices
    gen_A <- function(val){
      current.design <- A.use[ c(current.rows[-which(current.rows == couples$current[val])], couples$candidate[val]),]
      return(current.design)
    }

    # Func for making W matrices
    gen_W <- function(val){
      current.weights <- W.use[ c(current.rows[-which(current.rows == couples$current[val])], couples$candidate[val]), c(current.rows[-which(current.rows == couples$current[val])], couples$candidate[val])]
      return(current.weights)
    }
    
    # Execute vectorized form
    vals <- 1:nrow( couples); vals
    
    # This parrallel option is parrallel for both mapply and lapply:
    couples$D <- mcmapply(crit.func, mclapply(vals, gen_A, mc.cores = 2), mclapply(vals, gen_W, mc.cores = 2), mc.cores = 1)

    # Calculate difference
    couples$D.diff <- couples$D - current.D
  
    # If no exchanges improve the design, stop.  Otherwise, make the best swap and continue the algorithm
    if( sum( couples$D.diff > 0) == 0){
      break
    } else if( current.iter == maxiter) {
      print("The maximum number of iterations occured.")
      break
    }else{
      current.rows <- c( current.rows[ -which( current.rows == couples$current[ which.max( couples$D.diff)[1]])], couples$candidate[ which.max( couples$D.diff)[1]])
      current.iter <- current.iter + 1
    }
  }
  # Return the optimal design, weight matrix, criteria, and rows
  return(
    list(
      A = A.use[ current.rows,],
      W = W.use[ current.rows,current.rows],
      criterion = current.D,
      rows = current.rows
    )
  )
}
################################################################
# End optimization statistical engine.
################################################################

# Functions to check if optimization can numerically be performed.
# If not, (ex: scores are all 1.0), return deterministic seed design.
test_singular_A <- function(design.mat, weight.mat) {
  result <- tryCatch(
    {
      sum( diag( solve( t( design.mat) %*% diag( weight.mat) %*% design.mat)))
    },
    error = function(e) {
      # NULL means that the matrices are exactly singular
      NULL
    }
  )
  #print(paste0('The result: ', result))
}

test_singular_D <- function(design.mat, weight.mat) {
  result <- tryCatch(
    {
      det( solve( t( design.mat) %*% diag(weight.mat) %*% design.mat))
    },
    error = function(e) {
      # NULL means that the matrices are exactly singular
      NULL
    }
  )
  #print(paste0('The result: ', result))

}


################################################################
# Import scores from python
################################################################

run_optimization <- function(ref_lig, dataframe, r_optim_types, num_edges, random_seed){
    # ref_lig = reference ligand to use, currently must be manually selected
    # dataframe  = r dataframe generate from similarity scores
    # optim_type1 and optim_type2 = optimization type such as 'A' and 'D'.
    #                               currently two types are required.
    # random_seed = if set with an int makes the optimization nondeterministic.
    #               Default value is 'NULL' which mean no random seed selected.
    # Import atom pair scores
    optim_type1 <- r_optim_types[1]
    optim_type2 <- r_optim_types[2]
    k_numb <- num_edges
    # Convert default str from python to readable in R.
    if (random_seed == 'NULL'){
      random_seed <- NULL
    }
    set.seed(random_seed)
    print(paste("Preparing", optim_type1, "optimization"))
    print(paste("Preparing", optim_type2, "optimization"))

    sim.scores <- dataframe %>%
      as.matrix() %>%
      reshape2::melt( as.is = TRUE) %>%
      as_tibble() %>%
      setNames( c( 'LIGAND_1', 'LIGAND_2', 'SIM')) %>%
      filter( LIGAND_1 != LIGAND_2) %>%
      # Normalize weights.
      mutate( WEIGHT = (SIM^2 - min(SIM)^2)/(max(SIM)^2 - min(SIM)^2))
      #print("WEIGHT")
      #print(sim.scores$WEIGHT)
    
    # If all scores are equal, normalization produces NaNs. Replace NaN with 1.0.
    sim.scores <- mutate(sim.scores, WEIGHT = ifelse(is.nan(WEIGHT), 1.0, WEIGHT))
    #print("WEIGHT")
    #print(sim.scores$WEIGHT)
    
    ## Find the optimal designs
    # Define the list of ligands
    ligand.list <- sim.scores$LIGAND_1 %>% unique() %>% sort()
    reference.ligand <- c(ref_lig)
 
    # Generate full design
    A.full <- matrix( 0, ncol = length( ligand.list), nrow = choose( length( ligand.list), 2))
    for( i in 1:nrow( A.full)){
      A.full[i, t( combn( 1:length( ligand.list), 2))[i,]] <- c( 1, -1)
    }
    #print('A.full dims is:')
    #print(dim(A.full))
    print( paste("Number of ligands:", length( ligand.list)))
    print(paste('The number of edges is', k_numb))
    ###############

    # Define the A and W matrices for a full design with weights
    # Output array with a 1 at column of reference ligands. Rows: number of ref ligands
    a <- rbind(diag( ncol( A.full))[ which( ligand.list %in% reference.ligand),])

    A.use <- rbind( A.full, a)
    #print('A.use dims is:')
    #print(dim(A.use))

    # Generate array of 2's for length of reference ligands
    ref_weights <- rep(2, length(reference.ligand))
    W.use <- diag( c(
      apply( A.full, 1, function(x){
        sim.scores %>%
          filter(
            LIGAND_1 == ligand.list[ which( x == 1)] &
              LIGAND_2 == ligand.list[ which( x == -1)]
          ) %>%
          pull( WEIGHT)
      }), ref_weights
    ))
    #print('W.use dims in:')
    #print(dim(W.use))
    
    # Print functions
    #print('The ref_weights are')
    #print(ref_weights)
    #print('The weight matrix is')
    #print(W.use)
    #print('The A.use')
    #print(A.use)
    
    # Generate seed design to optimize.
    starting.rows <- Fedorov.exchange( A.use, W.use, c( 1:k_numb, nrow(A.use)), 'random')$rows

    # Optimize.
    a.opt.new <- Fedorov.exchange( A.use, W.use, starting.rows, optim_type1)$A %>%
      {.[which( rowSums( .) == 0),]} %>%
      apply( 1, function(x){
        tibble(
          LIGAND_1 = ligand.list[ which( x == 1)],
          LIGAND_2 = ligand.list[ which( x == -1)]
        )
      }) %>% {do.call( "rbind", .)}
     
   # pprv <- profvis({
    d.opt.new <- Fedorov.exchange( A.use, W.use, starting.rows, optim_type2)$A %>%
    {.[which( rowSums( .) == 0),]} %>%
      apply( 1, function(x){
        tibble(
          LIGAND_1 = ligand.list[ which( x == 1)],
          LIGAND_2 = ligand.list[ which( x == -1)]
        )
      }) %>% {do.call( "rbind", .)}
      #prof_output = 'rprof.out'
    #})
    #print(pprv)
    
    #now2 <- Sys.time()
    #pprv_ofile = paste0("timing", format(now2, "_%Y%m%d_%H%M%S"), ".html")
    #save(pprv, file= pprv_ofile)
    #htmlwidgets::saveWidget(pprv, pprv_ofile)
    

    ###################################################################
    ## Combine all designs and weights together
    string_A = paste(optim_type1, "-optimal")
    string_D = paste(optim_type2, "-optimal")
        
    analysis.dat <- bind_rows(
      a.opt.new %>% mutate( DESIGN = string_A),
      d.opt.new %>% mutate( DESIGN = string_D)
    ) %>%
      left_join( sim.scores %>% select( LIGAND_1, LIGAND_2, 'SIM_WEIGHT' = WEIGHT))
      
    ###################################################################
    ## Visualize designs

    # Compute the optimality criteria for each design.
    crit.dat <- analysis.dat %>%
      tidyr::crossing( as_tibble( t( setNames( rep( 0, length( ligand.list)), ligand.list)))) %>%
      rowwise() %>%
      do( mutate_if( as_tibble(.), grepl( .$LIGAND_1, names(.)), function(x){-1})) %>%
      do( mutate_if( as_tibble(.), grepl( .$LIGAND_2, names(.)), function(x){1})) %>%
      ungroup() %>%
      group_by( DESIGN) %>%
      do({
        design.mat <- {.} %>%
          select_at( ligand.list) %>%
          as.matrix() %>%
          rbind( diag( length( ligand.list))[ which( ligand.list == reference.ligand),])
        
        null.weight <- c( rep( 1, nrow(.)), 2)
        sim.weight <- c( .$SIM_WEIGHT, 2)
        
        # Removed solve from null.inner because det(inverse A) = 1/det(A)
        null.inner =  t( design.mat) %*% diag( null.weight) %*% design.mat
        weighted.inner = t( design.mat) %*% diag( sim.weight) %*% design.mat
        
        # This is the hat matrix values, in development.
        # H diags
        #print("H diags")
        #print(diag(((design.mat %*% solve(t(design.mat) %*% diag( sim.weight) %*% design.mat) %*% t(design.mat) %*% diag( sim.weight)))))
        
        # Output critical data dataframe
        data.frame(
          A = test_singular_A(design.mat, null.weight),
          D = (test_singular_D(design.mat, null.weight))^(1/length( ligand.list)),
          A.ap = test_singular_A(design.mat, sim.weight),
          D.ap = (test_singular_D(design.mat, sim.weight))^(1/length( ligand.list))
          #A = sum( diag( solve( t( design.mat) %*% diag( null.weight) %*% design.mat))),
          #D = det( solve( t( design.mat) %*% diag( null.weight) %*% design.mat))^(1/length( ligand.list)),
          #A.ap = sum( diag( solve( t( design.mat) %*% diag( sim.weight) %*% design.mat))),
          #D.ap = det( solve( t( design.mat) %*% diag( sim.weight) %*% design.mat))^(1/length( ligand.list))
          #H =  ((design.mat %*% solve(t(design.mat) %*% diag( sim.weight) %*% design.mat) %*% t(design.mat) %*% diag( sim.weight))),
          #Diag.Hi =  diag(((design.mat %*% solve(t(design.mat) %*% diag( sim.weight) %*% design.mat) %*% t(design.mat) %*% diag( sim.weight)))),
          #Diag.nullHi =  diag(((design.mat %*% solve(t(design.mat) %*% design.mat) %*% t(design.mat))))
          )
    
      }) %>%
      ungroup()
    
    # For in development.
    #print( "crit.dat, leverages")
    #print( crit.dat[, c("Diag.Hi", "Diag.nullHi")])
    
    print( "Critical Data:")
    print( crit.dat)
  
    # Plot the designs.
    # Collect vertices or node info for designs.
    vertex.dat <- analysis.dat %>%
      group_by( DESIGN) %>%
      do({
        net <- graph_from_data_frame(., directed = FALSE)
        bind_cols(
          as.data.frame( vertex.attributes( net)),
          as.data.frame( layout_with_gem( net))
        ) %>% setNames( c( 'LIGAND', 'X', 'Y'))
      }) %>% ungroup() %>%
      mutate( REF = ifelse( LIGAND %in% reference.ligand, 'Reference Ligand', 'Other')) %>%
      mutate( REF = factor( REF, levels = c( 'Reference Ligand', 'Other')))
    
    print("Vertex Data:")
    vertex.dat %>% as_tibble() %>% print(n=60)
    
    # Collect edge data for the designs.
    edge.dat <- analysis.dat %>%
      left_join( vertex.dat %>% select( 'LIGAND_1' = LIGAND, DESIGN, 'X1' = X, 'Y1' = Y)) %>%
      left_join( vertex.dat %>% select( 'LIGAND_2' = LIGAND, DESIGN, 'X2' = X, 'Y2' = Y))

    # Change name based on metric type. Might need to deprecate.
    vertex.dat <- vertex.dat %>%
      mutate( DESIGN = factor( DESIGN, levels = c( string_A,  string_D)))

    edge.dat <- edge.dat %>%
      mutate( DESIGN = factor( DESIGN, levels = c( string_A,  string_D)))
    
   # For in development.
    #leverage.dat <- crit.dat %>%
      #mutate( DESIGN = factor( DESIGN, levels = c( "A-optimal (lomap)",  "D-optimal (lomap)")))
        
    # Plotting of the weights colored.
    bottom.plot <- ggplot( vertex.dat, aes( x = X, y = Y)) +
      geom_segment( size = 1, data = edge.dat, aes( x = X1, y = Y1, xend = X2, yend = Y2, color = SIM_WEIGHT)) +
      geom_point( size = 10, pch = 21, aes( fill = REF)) +
      geom_text( size = 2, aes( x = X, y = Y, label = LIGAND)) +
      facet_wrap( ~ DESIGN, scales = "free", ncol = 2) +
      scale_color_gradientn( colors = c( "red1", "orange", "gold", "green", "springgreen"), limits = c( 0, 1)) +
      theme_bw( base_size = 20) + labs( fill = NULL, color = NULL) +
      theme( panel.border = element_blank(), panel.grid.major = element_blank(), panel.grid.minor = element_blank(), strip.background = element_blank()) +
      theme( axis.title = element_blank(), axis.text = element_blank(), axis.ticks = element_blank()) +
      theme( plot.title = element_text( hjust = 0.5), legend.position = 'top', legend.box = 'vertical', legend.key.width = unit(4, 'line'), legend.title=element_text(vjust = 0.9)) +
      labs( color = glue('Weights:'))
    now <- Sys.time()
    ggsave(paste0("Rgraph_colored", format(now, "_%Y%m%d_%H%M%S"), ".pdf"))
    
    # Write csv file
    csv_ofile = paste0("edge_data", format(now, "_%Y%m%d_%H%M%S"), ".csv")
    write.csv(edge.dat,csv_ofile, row.names = FALSE)
    
    return(edge.dat)
    }
'''
