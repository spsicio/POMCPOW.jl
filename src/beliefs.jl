const GINI = false

ent_from_part_e(ent_partial, tot) = log(tot) - ent_partial / tot
ent_from_part_g(ent_partial, tot) = 1.0 - ent_partial / (tot * tot)
ent_part_e(w) = isapprox(w, 0.0) ? 0.0 : w * log(w)
ent_part_g(w) = w * w

ent_functions = Dict(false => (ent_from_part_e, ent_part_e),
                     true  => (ent_from_part_g, ent_part_g))
ent_from_part, ent_part = ent_functions[GINI]

mutable struct POWNodeBelief{S,A,O,P}
    model::P
    a::A # may be needed in push_weighted! and since a is constant for a node, we store it
    o::O
    dist::CategoricalVector{Tuple{S,Float64}}

    ws::Dict{S, Float64} # weights of the states in the distribution
    ent::Float64 # entropy of the distribution
    ent_partial::Float64 # intermediate variables for entropy computation

    POWNodeBelief{S,A,O,P}(m,a,o,d, ws, e, ep) where {S,A,O,P} = new(m,a,o,d, ws, e, ep)
    function POWNodeBelief{S, A, O, P}(m::P, s, a, sp, o, r) where {S, A, O, P}
        w = obs_weight(m, s, a, sp, o)
        cv = CategoricalVector{Tuple{S,Float64}}((convert(S, sp), convert(Float64, r)), w)
        ws = Dict{S, Float64}(sp => w)
        new(m, a, o, cv, ws, 0.0, ent_part(w))
    end
end

function POWNodeBelief(model::POMDP{S,A,O}, s, a, sp, o, r) where {S,A,O}
    POWNodeBelief{S,A,O,typeof(model)}(model, s, a, sp, o, r)
end

rand(rng::AbstractRNG, b::POWNodeBelief) = rand(rng, b.dist)
state_mean(b::POWNodeBelief) = first_mean(b.dist)
POMDPs.currentobs(b::POWNodeBelief) = b.o
POMDPs.history(b::POWNodeBelief) = tuple((a=b.a, o=b.o))

struct POWNodeFilter end

belief_type(::Type{POWNodeFilter}, ::Type{P}) where {P<:POMDP} = POWNodeBelief{statetype(P), actiontype(P), obstype(P), P}

init_node_sr_belief(::POWNodeFilter, p::POMDP, s, a, sp, o, r) = POWNodeBelief(p, s, a, sp, o, r)

function push_weighted!(b::POWNodeBelief, ::POWNodeFilter, s, sp, r)
    w = obs_weight(b.model, s, b.a, sp, b.o)
    insert!(b.dist, (sp, convert(Float64, r)), w)
    cur = get(b.ws, sp, 0.0); nxt = cur + w
    b.ent_partial += ent_part(nxt) - ent_part(cur);
    b.ent = ent_from_part(b.ent_partial, last(b.dist.cdf))
    b.ws[sp] = nxt
end

struct StateBelief{SRB<:POWNodeBelief}
    sr_belief::SRB
end

rand(rng::AbstractRNG, b::StateBelief) = first(rand(rng, b.sr_belief))
mean(b::StateBelief) = state_mean(b.sr_belief)
POMDPs.currentobs(b::StateBelief) = currentobs(b.sr_belief)
POMDPs.history(b::StateBelief) = history(b.sr_belief)
