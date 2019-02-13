# a function for BGSM_cartesian_biclust (Section 5.2.)

# y       : a matrix of data
# init    : initialization of y

function BGSM_cartesian_biclust(y;
                                ind1 = 1:size(y,1),
                                ind2 = 1:size(y,2),
                                init = y,
                                v0 = 1e-1,
                                convtol = 1e-14,  
                                iter = 100,
                                verbose = true)
                            
    
    # get size
    n1,n2 = size(y); n = n1 * n2;
    
    
    # initialize
    theta = copy(y);
    k1 = length(ind1); k2 = length(ind2);
    mu1 = copy(y[ind1,:]); mu2 = copy(y[:,ind2]);
    q1 = zeros(n1,k1); q2 = zeros(n2,k2);
    sigmasq1 = 1; sigmasq2 = 1;
    
    # loop start
    for i = 1:iter
        
        # save previous iteration
        q1_old = copy(q1);
        q2_old = copy(q2);
        
        # E-step: update g1
        d1 = reshape(sum((repmat(theta,k1,1) - kron(mu1,ones(n1))).^2,2),n1,k1);
        q1 = exp.(-d1/(2*n2*v0)); q1 = q1 ./ sum(q1,2);
        
        # E-step: update g2
        d2 = reshape(sum((repmat(theta',k2,1) - kron(mu2',ones(n2))).^2,2),n2,k2);
        q2 = exp.(-d2/(2*n1*v0)); q2 = q2 ./ sum(q2,2);
        
        # M-step: update theta
        L1 = (speye(n1) - (q1 ./ sum(q1,1)) * q1')/v0;
        L2 = (speye(n2) - (q2 ./ sum(q2,1)) * q2')/v0;
        theta = ((speye(n1) + L1)\y)/(speye(n2) + L2);
        mu1 = (q1 ./ sum(q1,1))' * theta;
        mu2 = theta * (q2 ./ sum(q2,1));
        
        if verbose & (rem(i,5) == 0)
            @printf "%3d-th iteration done: error = %0.2e\n" i norm(q1 - q1_old) + norm(q2 - q2_old)
        end
        
        if verbose & (norm(q1 - q1_old) + norm(q2 - q2_old) < convtol)
            @printf "the algorithm converged at %3d-th iteration: error = %0.2e\n" i norm(q1 - q1_old) + norm(q2 - q2_old)
            break;
        end
        
    end
    
    return Dict([
                (:theta, theta), (:q1, q1), (:q2, q2), (:mu1, mu1), (:mu2, mu2)
                ])
end

# a function for BGSM_cartesian (Section 5.2.)

# y       : a matrix of data
# init    : initialization of y

function BGSM_kronecker_biclust(y;
                                ind1 = 1:size(y,1), ind2 = 1:size(y,2),
                                init = y,
                                v0 = 1e-1,
                                c = 1,
                                convtol = 1e-14,  
                                iter = 100,
                                verbose = true)
                            
    
    # get size
    n1, n2 = size(y); n = n1 * n2;
    k1 = length(ind1); k2 = length(ind2);
    p1 = n1 * k1; p2 = n2 * k2;
    
    # initialize
    theta = copy(y);
    mu = y[ind1,ind2];
    q1 = zeros(n1,k1); q2 = zeros(n2,k2);
    for i = 1:k1
        q1[ind1[i],i] = 1;
    end
    for i = 1:k2
        q2[ind2[i],i] = 1;
    end
    sigmasq = 1;
    
    # loop start
    for i = 1:iter
        
        # save previous iteration
        q1_old = copy(q1);
        q2_old = copy(q2);
        
        # E-step: update g1, g2
        temp = (repeat(theta, outer = [k1, k2]) - repeat(mu, inner = [n1, n2])).^2;
        q1 = reshape(exp.(-sum(q2[:]' .* temp,2)/(2*n2*v0*c)),n1,k1); q1 = q1./sum(q1,2);
        q2 = reshape(exp.(-sum(q1[:] .* temp,1)[:]/(2*n1*v0)),n2,k2); q2 = q2./sum(q2,2);
        
        # M-step: update theta
        L1 = (speye(n1) - (q1 ./ sum(q1,1)) * q1')/v0;
        L2 = (speye(n2) - (q2 ./ sum(q2,1)) * q2')/v0;
        theta = ((speye(n1) + L1)\y)/(speye(n2) + L2);
        
        # M-step: update mu
        mu = (q1 ./ sum(q1,1))' * theta * (q2 ./ sum(q2,1));
        
        if verbose & (rem(i,5) == 0)
            @printf "%2d-th iteration done: error = %0.2e\n" i norm(q1 - q1_old) + norm(q2 - q2_old)
        end
        
        if  verbose & (norm(q1 - q1_old) + norm(q2 - q2_old) < convtol)
            @printf "the algorithm converged at %2d-th iteration: error = %0.2e\n" i norm(q1 - q1_old) + norm(q2 - q2_old)
            break;
        end
        
    end
    
    return Dict([
                (:theta, theta), (:q1, q1), (:q2, q2), (:mu, mu), (:final, q1*mu*q2')
                ])
end