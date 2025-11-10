export BoundsConstraint
export GlobalBoundsConstraint

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

    indices = traj.dim * traj.N .+ traj.global_components[name]

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

    return GlobalBoundsConstraint(name, bounds, traj; label=label)
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

    return GlobalBoundsConstraint(name, bounds, traj; label=label)
end
