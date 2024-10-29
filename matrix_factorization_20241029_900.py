import numpy as np

# Function for matrix factorization using SGD with biases
def matrix_factorization_with_bias(R, K, steps, alpha, beta):
    num_users, num_items = R.shape
    
    # Initialize latent factors and biases
    U = np.random.normal(scale=1./K, size=(num_users, K))  # User matrix
    V = np.random.normal(scale=1./K, size=(num_items, K))  # Item matrix
    b_u = np.zeros(num_users)  # User biases
    b_i = np.zeros(num_items)  # Item biases
    mu = np.mean(R[R > 0])  # Global average rating, ignoring zero ratings

    for step in range(steps):
        loss = 0
        for i in range(num_users):
            for j in range(num_items):
                if R[i, j] > 0:  # Only update for non-zero ratings
                    # Calculate prediction with biases
                    prediction = mu + b_u[i] + b_i[j] + np.dot(U[i, :], V[j, :].T)
                    error = R[i, j] - prediction
                    loss += error**2

                    # Regularization terms
                    loss += (beta/2) * (np.linalg.norm(U[i, :])**2 + np.linalg.norm(V[j, :])**2 + b_u[i]**2 + b_i[j]**2)

                    # Update biases
                    b_u[i] += alpha * (error - beta * b_u[i])
                    b_i[j] += alpha * (error - beta * b_i[j])

                    # Update latent factors U and V
                    U[i, :] += alpha * (error * V[j, :] - beta * U[i, :])
                    V[j, :] += alpha * (error * U[i, :] - beta * V[j, :])

        # Early stopping if loss is sufficiently low
        if loss < 0.001:
            break 

    return U, V, b_u, b_i, mu, loss

# Grid search over different values of K, alpha, and beta
def grid_search(R, K_values, alpha_values, beta_values, steps=5001):
    results = {}
    best_loss = float('inf')
    best_params = {}

    for K in K_values:
        for alpha in alpha_values:
            for beta in beta_values:
                print(f"\nRunning matrix factorization with K = {K}, alpha = {alpha}, beta = {beta}")
                U, V, b_u, b_i, mu, final_loss = matrix_factorization_with_bias(R, K, steps, alpha, beta)
                results[(K, alpha, beta)] = final_loss
                print(f"Final loss for K={K}, alpha={alpha}, beta={beta}: {final_loss:.4f}")
                
                # Track best parameters based on loss
                if final_loss < best_loss:
                    best_loss = final_loss
                    best_params = {'K': K, 'alpha': alpha, 'beta': beta}

    print(f"\nBest configuration: K={best_params['K']}, alpha={best_params['alpha']}, beta={best_params['beta']} with loss = {best_loss:.4f}")
    return best_params, results


# Prediction function to print predictions for zeros only
def print_zero_predictions(R, U, V, b_u, b_i, mu):
    num_users, num_items = R.shape
    predicted_ratings = mu + b_u[:, np.newaxis] + b_i[np.newaxis, :] + np.dot(U, V.T)
    
    print("Predicted Ratings for Zero Entries in R:")
    for i in range(num_users):
        for j in range(num_items):
            if R[i, j] == 0:
                print(f"User {i}, Item {j}: Predicted Rating = {predicted_ratings[i, j]:.2f}")



# Example usage with a small ratings matrix
R = np.array([
    [3, 1, 0, 3, 1],
    [1, 2, 4, 1, 0],
    [3, 1, 1, 3, 1],
    [0, 3, 5, 4, 4]
])



"""
# Run grid search

# Values of K, alpha, and beta to search over
K_values = [2, 5, 10, 20]
alpha_values = [0.001, 0.005, 0.01, 0.02]
beta_values = [0.01, 0.05, 0.1, 0.2]

best_params, results = grid_search(R, K_values, alpha_values, beta_values)

# Best configuration
print("\nOptimal configuration:")
print(f"K = {best_params['K']}, Alpha = {best_params['alpha']}, Beta = {best_params['beta']}")
print("Loss values for each combination:")
for params, loss in results.items():
    print(f"K={params[0]}, alpha={params[1]}, beta={params[2]}: Loss = {loss:.4f}")
"""

Alpha = 0.02
Beta = 0.01
K=5
steps = 5000+1
U, V, b_u, b_i, mu, loss = matrix_factorization_with_bias(R, K,steps, Alpha, Beta)

# print error
print(f"loss= {loss}")

# Print predicted ratings only for zero entries
print_zero_predictions(R, U, V, b_u, b_i, mu)    