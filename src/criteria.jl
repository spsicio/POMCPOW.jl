struct MaxUCBe
    c::Float64
    e::Float64
    threshold::Int
end

function select_best(crit::MaxUCBe, h_node::POWTreeObsNode, rng)
    tree = h_node.tree
    h = h_node.node
    best_criterion_val = -Inf
    local best_node::Int
    istied = false
    local tied::Vector{Int}
    ltn = log(tree.total_n[h])
    for node in tree.tried[h]
        n = tree.n[node]
        if isinf(tree.v[node])
            criterion_value = tree.v[node]
        elseif n == 0
            criterion_value = Inf
        else
            delta_ent = 0
            sub_max_delta_ent = 0
            for (_, hao) in tree.generated[node]
                delta_ent += tree.total_n[hao] * tree.sr_beliefs[hao].ent
                if tree.max_delta_ent[hao] > sub_max_delta_ent
                    sub_max_delta_ent = tree.max_delta_ent[hao]
                end
                if tree.max_delta_ent[hao] > tree.max_delta_ent[h] # BackPropagate
                    tree.max_delta_ent[h] = tree.max_delta_ent[hao]
                end
            end
            delta_ent = (h == 1 ? 0 : tree.sr_beliefs[h].ent) - delta_ent / n
            if n > crit.threshold && delta_ent > tree.max_delta_ent[h] # BackPropagate
                tree.max_delta_ent[h] = delta_ent
            end
            criterion_value = n == 1 ? Inf :
                    tree.v[node] + crit.c * sqrt(ltn/n) +
                    crit.e * (delta_ent + sub_max_delta_ent) / sqrt(log(n))
        end
        if criterion_value > best_criterion_val
            best_criterion_val = criterion_value
            best_node = node
            istied = false
        elseif criterion_value == best_criterion_val
            if istied
                push!(tied, node)
            else
                istied = true
                tied = [best_node, node]
            end
        end
    end
    if istied
        return rand(rng, tied)
    else
        return best_node
    end
end

struct MaxUCB
    c::Float64
end

function select_best(crit::MaxUCB, h_node::POWTreeObsNode, rng)
    tree = h_node.tree
    h = h_node.node
    best_criterion_val = -Inf
    local best_node::Int
    istied = false
    local tied::Vector{Int}
    ltn = log(tree.total_n[h])
    for node in tree.tried[h]
        n = tree.n[node]
        if isinf(tree.v[node])
            criterion_value = tree.v[node]
        elseif n == 0
            criterion_value = Inf
        else
            criterion_value = tree.v[node] + crit.c*sqrt(ltn/n)
        end
        if criterion_value > best_criterion_val
            best_criterion_val = criterion_value
            best_node = node
            istied = false
        elseif criterion_value == best_criterion_val
            if istied
                push!(tied, node)
            else
                istied = true
                tied = [best_node, node]
            end
        end
    end
    if istied
        return rand(rng, tied)
    else
        return best_node
    end
end

struct MaxQ end

function select_best(crit::MaxQ, h_node::POWTreeObsNode, rng)
    tree = h_node.tree
    h = h_node.node
    best_node = first(tree.tried[h])
    best_v = tree.v[best_node]
    @assert !isnan(best_v)
    for node in tree.tried[h][2:end]
        if tree.v[node] >= best_v
            best_v = tree.v[node]
            best_node = node
        end
    end
    return best_node
end

struct MaxTries end

function select_best(crit::MaxTries, h_node::POWTreeObsNode, rng)
    tree = h_node.tree
    h = h_node.node
    best_node = first(tree.tried[h])
    best_n = tree.n[best_node]
    @assert !isnan(best_n)
    for node in tree.tried[h][2:end]
        if tree.n[node] >= best_n
            best_n = tree.n[node]
            best_node = node
        end
    end
    return best_node
end
