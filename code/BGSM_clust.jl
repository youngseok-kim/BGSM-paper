# a function for BGSM_clust (Section 4)

function BGSM_clust(      y; 
                          k = size(y,1),                        # a number of clusters
                          v0 = 1e-1, v1 = 1e4,                  # tuning parameters
                          a = 0, b = 2 * prod(size(y)),         # inverse gamma hyperparameters
                          convtol = 1e-10,                      # tolerances
                          iter = 1000,                          # a number of maximum allowed iteration
                          verbose = true)                       # verbose option
    
    # n is the length of vector y
    n = length(y);
    
    # initialze
    theta = copy(y); mu = copy(y); sigmasq = 1;
    
    # loop start
    for i = 1:iter
        
        # calculate squared distance
        deltasq = reshape(sum((repmat(theta,n,1) - kron(mu,ones(n))).^2,2),n,n);
        
        # update Q
        Q = exp.(-deltasq/(v0*sigmasq)); Q = Q./ sum(Q,2);
        R = Diagonal(1./sum(Q,1)[:]) * Q'; L = speye(n) - Q * R;
        
        # update theta, mu, sigma^2
        theta = (speye(n) + L/v0)\y;
        mu = R * theta;
        sigmasq = (sum((y-theta).^2) + sum((theta - Q * mu).^2)/v0 + b)/(2*n+a+1);
    end
    
    R = Array{Int}(reshape(sum((repmat(mu,n,1) - kron(mu,ones(n))).^2,2),n,n) .< 1e-8);
    A = eig(R);
    ind = find(A[1] .> 0.5);
    B = abs.(A[2][:,ind] .* sqrt.(A[1][ind]'));
    B[B .< 0.5] = 0;
    theta_tilde = B*inv(B'*B)B'*y;
    
    return Dict([
                (:theta, theta), (:mu, mu), (:theta_tilde, theta_tilde), (:R, R), (:B,B)
                ])
end