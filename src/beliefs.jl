ent_from_part_e(ent_partial, tot) = log(tot) - ent_partial / tot
ent_from_part_g(ent_partial, tot) = 1.0 - ent_partial / (tot * tot)
ent_part_e(w) = isapprox(w, 0.0) ? 0.0 : w * log(w)
ent_part_g(w) = w * w

abstract type AbstractPOWNodeBelief{S, A, O, P} end

struct POWNodeBelief{S, A, O, P} <: AbstractPOWNodeBelief{S, A, O, P}
    model::P
    a::A # may be needed in push_weighted! and since a is constant for a node, we store it
    o::O
    dist::CategoricalVector{Tuple{S,Float64}}

    POWNodeBelief{S,A,O,P}(m,a,o,d) where {S,A,O,P} = new(m,a,o,d)
    function POWNodeBelief{S, A, O, P}(m::P, s, a, sp, o, r) where {S, A, O, P}
        w = obs_weight(m, s, a, sp, o)
        cv = CategoricalVector{Tuple{S,Float64}}((convert(S, sp), convert(Float64, r)), w)
        new(m, a, o, cv)
    end
end

mutable struct POWeNodeBelief{S, A, O, P} <: AbstractPOWNodeBelief{S, A, O, P}
    model::P
    a::A # may be needed in push_weighted! and since a is constant for a node, we store it
    o::O
    dist::CategoricalVector{Tuple{S,Float64}}

    ws::Dict{S, Float64} # weights of the states in the distribution
    ent::Float64 # entropy of the distribution
    ent_partial::Float64 # intermediate variables for entropy computation

    POWeNodeBelief{S,A,O,P}(m,a,o,d, ws, e, ep) where {S,A,O,P} = new(m,a,o,d, ws, e, ep)
    function POWeNodeBelief{S, A, O, P}(m::P, s, a, sp, o, r, ent_part) where {S, A, O, P}
        w = obs_weight(m, s, a, sp, o)
        cv = CategoricalVector{Tuple{S,Float64}}((convert(S, sp), convert(Float64, r)), w)
        ws = Dict{S, Float64}(sp => w)
        new(m, a, o, cv, ws, 0.0, ent_part(w))
    end
end

function POWNodeBelief(model::POMDP{S,A,O}, s, a, sp, o, r) where {S, A, O}
    POWNodeBelief{S, A, O, typeof(model)}(model, s, a, sp, o, r)
end

function POWeNodeBelief(model::POMDP{S,A,O}, s, a, sp, o, r, ent_part) where {S, A, O}
    POWeNodeBelief{S, A, O, typeof(model)}(model, s, a, sp, o, r, ent_part)
end

rand(rng::AbstractRNG, b::AbstractPOWNodeBelief) = rand(rng, b.dist)
state_mean(b::AbstractPOWNodeBelief) = first_mean(b.dist)
POMDPs.currentobs(b::AbstractPOWNodeBelief) = b.o
POMDPs.history(b::AbstractPOWNodeBelief) = tuple((a=b.a, o=b.o))

abstract type AbstractPOWNodeFilter end
struct POWNodeFilter end
struct POWeNodeFilter end
struct POWgNodeFilter end

belief_type(::Type{POWNodeFilter}, ::Type{P}) where {P<:POMDP} = POWNodeBelief{statetype(P), actiontype(P), obstype(P), P}
belief_type(::Type{POWeNodeFilter}, ::Type{P}) where {P<:POMDP} = POWeNodeBelief{statetype(P), actiontype(P), obstype(P), P}
belief_type(::Type{POWgNodeFilter}, ::Type{P}) where {P<:POMDP} = POWeNodeBelief{statetype(P), actiontype(P), obstype(P), P}

init_node_sr_belief(::POWNodeFilter, p::POMDP, s, a, sp, o, r) = POWNodeBelief(p, s, a, sp, o, r)
init_node_sr_belief(::POWeNodeFilter, p::POMDP, s, a, sp, o, r) = POWeNodeBelief(p, s, a, sp, o, r, ent_part_e)
init_node_sr_belief(::POWgNodeFilter, p::POMDP, s, a, sp, o, r) = POWeNodeBelief(p, s, a, sp, o, r, ent_part_g)

function push_weighted!(b::POWNodeBelief, ::POWNodeFilter, s, sp, r)
    w = obs_weight(b.model, s, b.a, sp, b.o)
    insert!(b.dist, (sp, convert(Float64, r)), w)
end

function push_weighted!(b::POWeNodeBelief, ::POWeNodeFilter, s, sp, r)
    w = obs_weight(b.model, s, b.a, sp, b.o)
    insert!(b.dist, (sp, convert(Float64, r)), w)
    cur = get(b.ws, sp, 0.0); nxt = cur + w
    b.ent_partial += ent_part_e(nxt) - ent_part_e(cur);
    b.ent = ent_from_part_e(b.ent_partial, last(b.dist.cdf))
    b.ws[sp] = nxt
end

function push_weighted!(b::POWeNodeBelief, ::POWgNodeFilter, s, sp, r)
    w = obs_weight(b.model, s, b.a, sp, b.o)
    insert!(b.dist, (sp, convert(Float64, r)), w)
    cur = get(b.ws, sp, 0.0); nxt = cur + w
    b.ent_partial += ent_part_g(nxt) - ent_part_g(cur);
    b.ent = ent_from_part_g(b.ent_partial, last(b.dist.cdf))
    b.ws[sp] = nxt
end

struct StateBelief{SRB<:AbstractPOWNodeBelief}
    sr_belief::SRB
end

rand(rng::AbstractRNG, b::StateBelief) = first(rand(rng, b.sr_belief))
mean(b::StateBelief) = state_mean(b.sr_belief)
POMDPs.currentobs(b::StateBelief) = currentobs(b.sr_belief)
POMDPs.history(b::StateBelief) = history(b.sr_belief)
