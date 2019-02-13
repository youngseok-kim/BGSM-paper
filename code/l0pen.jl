function eff_resistance(g, D = incidence_matrix(g, oriented = true))
    L_pinv = pinv(full(laplacian_matrix(g)));
    e = size(D,2); out = zeros(e);
    for i = 1:e
        out[i] = D[:,i]'* L_pinv * D[:,i];
    end
    return out
end
function weight_matrix(g, w = eff_resistance(g))
    n = length(vertices(g))
    W = sparse(zeros(n,n))
    j = 1
    for i in edges(g)
        W[i.src,i.dst] = W[i.dst,i.src] = w[j];
        j += 1;
    end
    return W
end
function alpha_expansion(y,t,c,lambda,g, w = eff_resistance(g), D = incidence_matrix(g,oriented = true))
    # ind1 : (i,j) with t_i = t_j
    # ind2 : (i,j) with t_i != t_j
    # ind3 : boolean whether t_k != c
    # ind4 : k with t_k != c
    temp = (D' * t .== 0);
    ind1 = find(temp); ind2 = find(.!temp); ind3 = (t .!= c); ind4 = find(ind3)
    n = length(t); m = length(ind2)
    
    # make augmentation G of g
    G = DiGraph(n+m+2);
    
    # delete edge (i,j) if t_i = t_j
    w[ind2] = 0;
    
    # construct capacity matrix for max-flow algorithm
    C = sparse(zeros(n+m+2,n+m+2));
    
    # add edge from source (n+m+1) to node i, from node i to sink (n+m+2)
    # update edge weight
    for i = 1:n
        add_edge!(G, n+m+1, i)
        C[n+m+1,i] = (y[i] - c)^2/2;
        
        add_edge!(G, i, n+m+2)
        C[1:n,n+m+2] = Inf;
        C[ind4,n+m+2] = (y[ind4] - t[ind4]).^2/2;
    end
    
    # add edges (j,k) if t_j = t_k
    for i in ind1
        j,k = find(D[:,i] .!= 0);
        add_edge!(G, j, k);
        add_edge!(G, k, j);
    end
    # assign lambda * w[j,k] only if t_j != c and t_k != c
    C[ind4,ind4] = lambda * weight_matrix(g,w)[ind4,ind4];
    
    # l : from 1 to m
    # we introduce m nodes where m is the number of edges (i,j) with t_i != t_j
    # add three edges and weights
    l = 1;
    for i in ind2
        j,k = find(D[:,i] .!= 0);
        add_edge!(G, j, n+l);
        add_edge!(G, k, n+l);
        add_edge!(G, n+l, n+m+2);
        C[j,n+l] = lambda * w[i] * ind3[j];
        C[n+l,j] = lambda * w[i] * ind3[j];
        C[k,n+l] = lambda * w[i] * ind3[k];
        C[n+l,k] = lambda * w[i] * ind3[k];
        C[n+l,n+m+2] = lambda * w[i];
    end
    out = maximum_flow(G,n+m+1,n+m+2,C,algorithm=BoykovKolmogorovAlgorithm());
    source = find(out[3][1:n] .== out[3][n+m+1]);
    sink = find(out[3][1:n] .== out[3][n+m+2]);
    t[sink] = c;
    return t
end
function local_minimizer(y, g; lambda = 0.5, delta= 0.01, tau = 1e-10, R = eff_resistance(g),
                        D = incidence_matrix(g,oriented = true), fix = [])
    obj_func(y,t) = sum((y-t).^2)/2 + lambda * sum(D'*t .!= 0);
    n = length(y);
    t = mean(y) * ones(n);
    Z_min = round(Int,minimum(y)/delta)-1
    Z_max = round(Int,maximum(y)/delta)+1
    i = 0;
    for i = 1:10
        t_new = zeros(n);
        t_prev = t + 0.0;
        for j = Z_min:Z_max
            c = j * delta;
            t_new = alpha_expansion(y,t,c,lambda,g,R,D) + 0.0;
            if obj_func(y,t_new) <= obj_func(y,t) - tau
                t = t_new + 0.0;
            end
        end
        if t_prev == t
            break;
        end
    end
    return t
end