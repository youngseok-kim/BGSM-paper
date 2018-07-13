# a function for BGSM_general (Section 2,3)

# y : a vector of data
# D : graph incidence matrix
# X : a design matrix
# w : a support vector
# nu : a prior precision for alpha
# v0, v1 : tuning parameters
# a, b : inverse gamma hyperparameters
# convtol : convergence tolerance
# orthotol : tolerance to check whether w is an eigenvector of X'X
# iter : a number of maximum allowed iteration

function BGSM_general(    y, D;                                 # a 
                          X = zeros(0,0), w = zeros(0),         # a design matrix and a support vector
                          nu = 0,                               # a prior precision for alpha
                          v0 = 1e-1, v1 = 1e4,                  # tuning parameters
                          a = 0, b = 1,                         # inverse gamma hyperparmeters
                          A = 1, B = 1,                         # beta bernoulli hyperparmeters
                          convtol = 1e-10, orthotol = 1e-10,    # tolerances
                          iter = 1000,                          # a number of maximum allowed iteration
                          verbose = true)                       # verbose option
    
    # n is the length of vector y
    n = length(y);
    m,p = size(D);
    if verbose == true
        println("A data \"y\" has input of length \"n\" = $n");
        println("A graph incidence matrix \"D\" has input of size (\"m\" = $m,\"p\" = $p)");
    end

    if size(X) == (0,0)
        if verbose == true
            println("A design matrix \"X\" has no input: use speye(n) as a default, \"n\" = $n");
        end
        X = speye(n);
    elseif size(X,1) != n
        error("Error: A data \"y\" has length \"n\" = $n thus a design matrix \"X\" must have $n rows.");
    end
    
    if length(w) == 0
        if verbose == true
            println("A support vector \"w\" has no input: use ones(p) as a default, \"p\" = $p");
        end
        w = ones(p);
    elseif length(w) != p
        error("Error: A graph incidence matrix \"D\" has \"p\" = $p columns thus a support vector \"w\" must have $p rows.");
    end
    
    if nu == 0
        # orthogonality check
        temp = X'*X*w;
        alpha = w'*X'*y / norm(w)^2;
        if norm(temp/norm(temp) - w/norm(w)) < orthotol && verbose == true
            println("\"w\" seems an eigenvector of \"X'*X\". We use a simpler algorithm.");
        end
    elseif nu == Inf
        alpha = 0;
    else
        Error("\"ν\" values other than 0 or ∞ are not being implemented yet.")
    end

    # initialization
    theta = (X'*X + 1e-4 * speye(p))\(X'*(y - alpha));
    delta = D * theta;
    if X == speye(n)
        sigmasq = sum((y - alpha).^2)/n;
    else
        sigmasq = 1;
    end
    eta = 1/2;
    q = ones(m)/2; q_old = copy(q);
    if verbose == true
        println("We initialize at \"α\" = mean(y), \"θ\" = y-α, \"σ^2\" = var(y) and \"η\" = 1/2");
    end

    # loop start
    for i = 1:iter
        
        # E-step : update q and tau
        q = 1./( 1 + (1-eta)/eta * sqrt(v0/v1) * exp.(delta.^2/2 * (1/v0 - 1/v1)/sigmasq) );
        tau = q/v0 + (1-q)/v1;
        
        # M-step : update theta, delta, sigma^2
        theta = (X'*X + D'*Diagonal(tau)*D) \ (X'*(y - alpha));
        theta = theta - (w'*theta) * w/norm(w)^2; # projection
        delta = D * theta;
        sigmasq = (sum((y - alpha - X*theta).^2) + sum((delta .* sqrt.(tau)).^2) + b)/(n+m+a+1);

        # convergence criterion
        err = norm(q - q_old)/length(q);
        
        # convergence check
        if err < convtol
            if verbose == true
                @printf "iteration: %d, error: %0.2e\n" i err;
                println("the algorithm converges at $i-th iteration");
            end
            break;
        end
        
        # printout
        if rem(i,5) == 0 && verbose == true
            @printf "iteration: %d, error: %0.2e\n" i err;
        end
        
        # update eta
        eta = (sum(q) + A - 1)/(m + A + B - 2);
        
        # save for convergence check
        q_old = copy(q);
    end
    return Dict([
                 (:alpha,alpha), (:theta,theta), (:delta, delta), (:q, q),
                 (:sigmasq,sigmasq), (:eta,eta), 
                 (:y, y), (:D, D), (:X,X), (:w, w), (:nu,nu),
                 (:n, n), (:p, p), (:m, m),
                 (:v0, v0), (:v1, v1)
               ])
end