function model_selection(out; thresh = 0.5)
    # thresholding step
    ind = find(out[:q] .>= thresh);
    
    # get original graph
    A_orig = out[:D]'*out[:D]; A_orig = (abs.(A_orig) - A_orig)/2;
    
    # get a thresholded graph
    D = out[:D][ind,:]; L = D'*D;
    g = Graph((abs.(L) - L)/2);
    
    # edge contraction / node merger
    c = connected_components(g);
    s = length(c);
    
    # get reduced X, w
    if out[:X] == speye(out[:n]) || out[:w] == ones(out[:n])
        X = zeros(out[:n],s);
        for i = 1:s
            X[c[i],i] = 1;
        end
        w = ones(s);
    elseif out[:w] == [1;zeros(out[:p]-1)]
        S = collect(Iterators.flatten(c[2:end]));
        X = zeros(out[:n],s); X[:,2:end] = out[:X][:,S];
        w = sparse(zeros(s)); w[1] = 1;
    else
        error("For this case of \"x\" and \"w\", model selection is not supported yet.")
    end
    
    # get reduced P
    P = speye(s) - w*w'/norm(w)^2;
    
    # get a reduced model graph-algebraic objects
    A = zeros(s,s);
    for i = 1:s
        for j = i+1:s
            A[i,j] = sum(A_orig[c[i],c[j]]);
        end
    end
    A = (A .> 0); A = A + A';
    L = Diagonal(sum(A,2)[:]) - A; L = L/out[:v1];
    H = X'*X + P*L*P + out[:nu]*w*w'/norm(w)^2;
    
    # t_reduced : a vector of length s, which equals reduced alpha w + reduced theta
    t_reduced = zeros(s);
    if out[:nu] == Inf # just to avoid numerical issue - convention does not work here
        H = H[2:end,2:end]; L = L[2:end,2:end];
        eig_L = sort(eig(L)[1]);
        eig_H = sort(eig(H)[1]);
        t_reduced[2:end] = H\(X[:,2:end]'*out[:y]);
    else
        eig_L = sort(real(eig(L)[1]))[2:end];
        eig_H = sort(real(eig(H)[1]))[2:end];
        t_reduced = H\(X'*out[:y]);    
    end
    
    # t_full : a vector of length p, which equals full alpha w + full theta
    t_full = zeros(out[:p]);
    for i = 1:s
        t_full[c[i]] = t_reduced[i];
    end
    
    # Xt : X * t_reduced
    Xt = X*t_reduced;

    
    # model selection scores
    # first_term = -1/2 * log det_w(L), negative log-sum of nonzero eigenvalues of L
    first_term = -sum(log.(eig_L))/2;
    # second_term = 1/2 * log det_w(H), log-sum of nonzero eigenvalues of H
    second_term = sum(log.(eig_H))/2;
    # third_term = n/2 * log l2 loss 
    third_term = log(sum((out[:y]-X*t_reduced).^2) + out[:b]) * out[:n]/2;
    
    return Dict([(:t_reduced, t_reduced), (:Xt, Xt), (:t_full, t_full),
                 (:each, (first_term, second_term, third_term)), (:beta, t_full),
                 (:X, X), (:H, H), (:A, A), (:w, w), (:s, s), (:c, c), (:L, L),
                 (:score, first_term + second_term + third_term)
                ])
end