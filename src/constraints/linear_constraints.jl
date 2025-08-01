export EqualityConstraint
export GlobalEqualityConstraint
export BoundsConstraint
export GlobalBoundsConstraint
export AllEqualConstraint
export TimeStepsAllEqualConstraint
export L1SlackConstraint
export TotalConstraint
export DurationConstraint
export SymmetryConstraint
export SymmetricControlConstraint


### 
### EqualityConstraint
###

"""
    struct EqualityConstraint

Represents a linear equality constraint.

# Fields
- `ts::AbstractArray{Int}`: the time steps at which the constraint is applied
- `js::AbstractArray{Int}`: the components of the trajectory at which the constraint is applied
- `vals::Vector{R}`: the values of the constraint
- `vardim::Int`: the dimension of a single time step of the trajectory
- `label::String`: a label for the constraint

"""
struct EqualityConstraint <: AbstractLinearConstraint
    indices::Vector{Int}
    values::Vector{Float64}
    label::String
end

"""
    EqualityConstraint(
        name::Symbol,
        ts::Vector{Int},
        val::Vector{Float64},
        traj::NamedTrajectory;
        label="equality constraint on trajectory variable [name]"
    )

Constructs equality constraint for trajectory variable in NamedTrajectory
"""
function EqualityConstraint(
    name::Symbol,
    ts::AbstractVector{Int},
    val::Vector{Float64},
    traj::NamedTrajectory;
    label="equality constraint on trajectory variable $name"
)
    @assert length(val) == traj.dims[name]
    indices = vcat([slice(t, traj.components[name], traj.dim) for t ∈ ts]...)
    values = repeat(val, length(ts))
    return EqualityConstraint(indices, values, label)
end

function EqualityConstraint(
    name::Symbol,
    ts::AbstractVector{Int},
    val::Float64,
    traj::NamedTrajectory;
    label="equality constraint on trajectory variable $name"
)
    @assert val >= 0
    return EqualityConstraint(name, ts, fill(val, traj.dims[name]), traj; label=label)
end

"""
    GlobalEqualityConstraint(
        name::Symbol,
        val::Vector{Float64},
        traj::NamedTrajectory;
        label="equality constraint on global variable [name]"
    )::EqualityConstraint

Constructs equality constraint for global variable in NamedTrajectory
"""
function GlobalEqualityConstraint(
    name::Symbol,
    val::Vector{Float64},
    traj::NamedTrajectory;
    label="equality constraint on global variable $name"
)
    @assert name ∈ traj.global_names
    @assert length(val) == traj.global_dims[name]

    indices = traj.dim * traj.T .+ traj.global_components[name]
    return EqualityConstraint(indices, val, label)
end

struct AllEqualConstraint <: AbstractLinearConstraint
    indices::Vector{Int}
    bar_index::Int
    label::String
end

function TimeStepsAllEqualConstraint(
    traj::NamedTrajectory;
    label="timesteps all equal constraint"
)
    @assert traj.timestep isa Symbol
    indices = [index(k, traj.components[traj.timestep][1], traj.dim) for k ∈ 1:traj.T-1]
    bar_index = index(traj.T, traj.components[traj.timestep][1], traj.dim)
    return AllEqualConstraint(indices, bar_index, label)
end


### 
### BoundsConstraint
###

# TODO: Refactor using `get_bounds_from_dims` from NamedTrajectories.jl

struct BoundsConstraint <: AbstractLinearConstraint
    indices::Vector{Int}
    bounds::Vector{Tuple{Float64, Float64}}
    label::String
end

function BoundsConstraint(
    name::Symbol,
    ts::AbstractVector{Int},
    bounds::Tuple{Vector{Float64}, Vector{Float64}},
    traj::NamedTrajectory;
    subcomponents=1:traj.dims[name],
    label="bounds constraint on trajectory variable $name"
)
    @assert length(bounds[1]) == length(bounds[2]) == traj.dims[name]
    @assert all(bounds[1] .<= bounds[2])

    indices = vcat([
        slice(t, traj.components[name][subcomponents], traj.dim)
            for t ∈ ts
    ]...)

    bounds = repeat(collect(zip(bounds...)), length(ts))

    return BoundsConstraint(indices, bounds, label)
end

function BoundsConstraint(
    name::Symbol,
    ts::AbstractVector{Int},
    bound::Vector{Float64},
    traj::NamedTrajectory;
    kwargs...
)
    @assert length(bound) == traj.dims[name]
    @assert all(bound .>= 0) "bound must be non-negative when only one bound is provided"

    bounds = (-bound, bound)

    return BoundsConstraint(name, ts, bounds, traj; kwargs...)
end

function BoundsConstraint(
    name::Symbol,
    ts::AbstractVector{Int},
    bound::Float64,
    traj::NamedTrajectory;
    label="bounds constraint on trajectory variable $name"
)
    @assert bound >= 0 "bound must be non-negative when only one bound is provided"

    bounds = (-fill(bound, traj.dims[name]), fill(bound, traj.dims[name]))

    return BoundsConstraint(name, ts, bounds, traj; label=label)
end

function GlobalBoundsConstraint(
    name::Symbol,
    bounds::Tuple{Vector{Float64}, Vector{Float64}},
    traj::NamedTrajectory;
    label="bounds constraint on global variable $name"
)
    @assert name ∈ traj.global_names
    @assert length(bounds[1]) == length(bounds[2]) == traj.global_dims[name]
    @assert all(bounds[1] .<= bounds[2])

    indices = traj.dim * traj.T .+ traj.global_components[name]

    bounds = collect(zip(bounds...))

    return BoundsConstraint(indices, bounds, label)
end

function GlobalBoundsConstraint(
    name::Symbol,
    bound::AbstractVector{Float64},
    traj::NamedTrajectory;
    label="bounds constraint on global variable $name"
)
    @assert name ∈ traj.global_names
    @assert length(bound) == traj.global_dims[name]
    @assert all(bound .>= 0) "bound must be non-negative when only one bound is provided"

    bounds = (-bound, bound)

    return BoundsConstraint(name, bounds, label)
end

function GlobalBoundsConstraint(
    name::Symbol,
    bound::Float64,
    traj::NamedTrajectory;
    label="bounds constraint on global variable $name"
)
    @assert name ∈ traj.global_names
    @assert bound >= 0 "bound must be non-negative when only one bound is provided"

    bounds = (-fill(bound, traj.global_dims[name]), fill(bound, traj.global_dims[name]))

    return BoundsConstraint(name, bounds, label)
end


### 
### L1SlackConstraint
###

# TODO: Doesn't work with parametric trajectory
# struct L1SlackConstraint <: AbstractLinearConstraint
#     x_indices::Vector{Int}
#     s1_indices::Vector{Int}
#     s2_indices::Vector{Int}
#     label::String
# end

# function L1SlackConstraint(
#     name::Symbol,
#     traj::NamedTrajectory;
#     indices=1:traj.dims[name],
#     ts=(name ∈ keys(traj.initial) ? 2 : 1):(name ∈ keys(traj.final) ? traj.T-1 : traj.T),
#     label="L1 slack constraint on $name[$(indices)]"
# )
#     @assert all(i ∈ 1:traj.dims[name] for i ∈ indices)

#     s1_name = Symbol("s1_$name")
#     s2_name = Symbol("s2_$name")

#     add_component!(traj, s1_name, rand(length(indices), traj.T))
#     add_component!(traj, s2_name, rand(length(indices), traj.T))

#     x_indices = stack(slice(t, traj.components[name][indices], traj.dim) for t ∈ ts)
#     s1_indices = stack(slice(t, traj.components[s1_name], traj.dim) for t ∈ ts)
#     s2_indices = stack(slice(t, traj.components[s2_name], traj.dim) for t ∈ ts)

#     return L1SlackConstraint(
#         x_indices,
#         s1_indices,
#         s2_indices,
#         label
#     )
# end

struct TotalConstraint <: AbstractLinearConstraint
    indices::Vector{Int}
    value::Float64
    label::String
end


function DurationConstraint(
    traj::NamedTrajectory,
    value::Float64;
    label="duration constraint of $value"
)
    @assert traj.timestep isa Symbol
    indices = [index(k, traj.components[traj.timestep][1], traj.dim) for k ∈ 1:traj.T]
    return TotalConstraint(indices, value ,label)
end


struct SymmetryConstraint <: AbstractLinearConstraint
    even_index_pairs::Vector{Tuple{Int64,Int64}}
    odd_index_pairs::Vector{Tuple{Int64,Int64}}
    label::String 
end

function SymmetricControlConstraint(
    traj::NamedTrajectory,
    name::Symbol,
    idx::Vector{Int64};
    even = true,
    label = "symmetry constraint on $name"
)
    even_pairs = Vector{Tuple{Int64,Int64}}()
    odd_pairs = Vector{Tuple{Int64,Int64}}()

    component_indicies = [slice(t, traj.components[name], traj.dim)[idx] for t ∈ 1:traj.T]
    if(even)
        even_pairs = vcat(even_pairs,reduce(vcat,[collect(zip(component_indicies[[idx,traj.T - idx+1]]...)) for idx in 1:traj.T ÷ 2]))
    else 
        odd_pairs = vcat(odd_pairs,reduce(vcat,[collect(zip(component_indicies[[idx,traj.T - idx+1]]...)) for idx in 1:traj.T ÷ 2]))
    end 

    if traj.timestep isa Symbol
        time_indices = [index(k, traj.components[traj.timestep][1], traj.dim) for k ∈ 1:traj.T]
        even_pairs = vcat(even_pairs,[(time_indices[idx],time_indices[traj.T + 1 - idx]) for idx ∈ 1:traj.T÷2]) 
    end 

    return SymmetryConstraint(
        even_pairs,
        odd_pairs,
        label
    )

end

# =========================================================================== #

@testitem "testing symmetry constraint" begin

    include("../../test/test_utils.jl")

    G, traj = bilinear_dynamics_and_trajectory()

    integrators = [
        BilinearIntegrator(G, traj, :x, :u),
        DerivativeIntegrator(traj, :u, :du),
        DerivativeIntegrator(traj, :du, :ddu)
    ]

    J = TerminalObjective(x -> norm(x - traj.goal.x)^2, :x, traj)
    J += QuadraticRegularizer(:u, traj, 1.0) 
    J += QuadraticRegularizer(:du, traj, 1.0)
    J += MinimumTimeObjective(traj)

    prob = DirectTrajOptProblem(traj, J, integrators;)

    sym_constraint = SymmetricControlConstraint(
        prob.trajectory, 
        :u,
        [1];
        even = true
    );
    push!(prob.constraints, sym_constraint);
    
    solve!(prob; max_iter=10)
end

@testitem "testing duration constraint" begin

    include("../../test/test_utils.jl")

    G, traj = bilinear_dynamics_and_trajectory()

    integrators = [
        BilinearIntegrator(G, traj, :x, :u),
        DerivativeIntegrator(traj, :u, :du),
        DerivativeIntegrator(traj, :du, :ddu)
    ]

    J = TerminalObjective(x -> norm(x - traj.goal.x)^2, :x, traj)
    J += QuadraticRegularizer(:u, traj, 1.0) 
    J += QuadraticRegularizer(:du, traj, 1.0)
    J += MinimumTimeObjective(traj)

    prob = DirectTrajOptProblem(traj, J, integrators;)

    dur_constraint = DurationConstraint(
        prob.trajectory,
        10.0;
    )
    push!(prob.constraints, dur_constraint);

    solve!(prob; max_iter=10)
end
