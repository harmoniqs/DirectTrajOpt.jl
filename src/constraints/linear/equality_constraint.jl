export EqualityConstraint
export GlobalEqualityConstraint

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

    indices = traj.dim * traj.N .+ traj.global_components[name]
    return EqualityConstraint(indices, val, label)
end
