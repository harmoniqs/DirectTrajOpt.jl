export QuadraticRegularizer
# ----------------------------------------------------------------------------- #
# Quadratic Regularizer
# ----------------------------------------------------------------------------- #

"""
    QuadraticRegularizer(
        name::Symbol,
        traj::NamedTrajectory,
        R::Union{Real, AbstractVector{<:Real}};
        baseline::AbstractMatrix{<:Real}=zeros(traj.dims[name], traj.T),
        times::AbstractVector{Int}=1:traj.T
    )

Create a quadratic regularization objective for a trajectory component.

Minimizes the weighted squared deviation from a baseline trajectory, integrated over time:

```math
J = \\sum_{k \\in \\text{times}} \\frac{1}{2} (v_k - v_\\text{baseline})^T R (v_k - v_\\text{baseline}) \\Delta t
```

where `v_k` is the trajectory component at knot point `k`.

# Arguments
- `name::Symbol`: Name of the trajectory component to regularize
- `traj::NamedTrajectory`: The trajectory containing the component
- `R`: Regularization weight(s). Can be:
  - Scalar: same weight for all components
  - Vector: individual weights for each component dimension
- `baseline::AbstractMatrix`: Target values (default: zeros). Size: (component_dim × N)
- `times::AbstractVector{Int}`: Time indices to include in regularization (default: all)

# Returns
- `Objective`: Regularization objective with gradient and Hessian

# Examples
```julia
# Regularize control with uniform weight
obj = QuadraticRegularizer(:u, traj, 1e-2)

# Regularize with different weights per component
obj = QuadraticRegularizer(:u, traj, [1e-2, 1e-3])

# Regularize around a reference trajectory
obj = QuadraticRegularizer(:x, traj, 1.0, baseline=x_ref)

# Only regularize middle time steps
obj = QuadraticRegularizer(:u, traj, 1e-2, times=2:traj.T-1)
```
"""
function QuadraticRegularizer(
    name::Symbol,
    traj::NamedTrajectory,
    R::AbstractVector{<:Real};
    baseline::AbstractMatrix{<:Real}=zeros(traj.dims[name], traj.T),
    times::AbstractVector{Int}=1:traj.T,
)
    @assert length(R) == traj.dims[name] "R must have the same length as the dimension of the trajectory component"

    @views function L(Z⃗::AbstractVector{<:Real})
        J = 0.0
        for t ∈ times
            if traj.timestep isa Symbol
                Δt = Z⃗[slice(t, traj.components[traj.timestep], traj.dim)]
            else
                Δt = traj.timestep
            end

            vₖ = Z⃗[slice(t, traj.components[name], traj.dim)]
            Δv = vₖ - baseline[:, t]

            rₖ = Δt .* Δv
            J += 0.5 * rₖ' * (R .* rₖ)
        end
        return J
    end

    @views function ∇L(Z⃗::AbstractVector)
        ∇ = zeros(traj.dim * traj.T + traj.global_dim)
        Threads.@threads for t ∈ times
            vₖ_slice = slice(t, traj.components[name], traj.dim)
            Δv = Z⃗[vₖ_slice] .- baseline[:, t]

            if traj.timestep isa Symbol
                Δt_slice = slice(t, traj.components[traj.timestep], traj.dim)
                Δt = Z⃗[Δt_slice]
                ∇[Δt_slice] .= Δv' * (R .* (Δt .* Δv))
            else
                Δt = traj.timestep
            end

            ∇[vₖ_slice] .= R .* (Δt.^2 .* Δv)
        end
        return ∇
    end

    function ∂²L_structure()
        structure = []
        # Hessian structure (eq. 17)
        for t ∈ times
            vₖ_slice = slice(t, traj.components[name], traj.dim)
            vₖ_vₖ_inds = collect(zip(vₖ_slice, vₖ_slice))
            append!(structure, vₖ_vₖ_inds)

            if traj.timestep isa Symbol
                Δt_slice = slice(t, traj.components[traj.timestep], traj.dim)
                # ∂²_vₖ_Δt
                vₖ_Δt_inds = [(i, j) for i ∈ vₖ_slice for j ∈ Δt_slice]
                append!(structure, vₖ_Δt_inds)
                # ∂²_Δt_vₖ
                Δt_vₖ_inds = [(i, j) for i ∈ Δt_slice for j ∈ vₖ_slice]
                append!(structure, Δt_vₖ_inds)
                # ∂²_Δt_Δt
                Δt_Δt_inds = collect(zip(Δt_slice, Δt_slice))
                append!(structure, Δt_Δt_inds)
            end
        end
        return structure
    end

    function ∂²L(Z⃗::AbstractVector) 
        values = []
        # Match Hessian structure indices
        for t ∈ times
            if traj.timestep isa Symbol
                Δt = Z⃗[slice(t, traj.components[traj.timestep], traj.dim)]
                append!(values, R .* Δt.^2)
                # ∂²_vₖ_Δt, ∂²_Δt_vₖ
                vₖ = Z⃗[slice(t, traj.components[name], traj.dim)]
                Δv = vₖ .- baseline[:, t]
                append!(values, 2 * (R .* (Δt .* Δv)))
                append!(values, 2 * (R .* (Δt .* Δv)))
                # ∂²_Δt_Δt
                append!(values, Δv' * (R .* Δv))
            else
                Δt = traj.timestep
                append!(values, R .* Δt.^2)
            end
        end
        return values
    end

    return Objective(L, ∇L, ∂²L, ∂²L_structure)
end

function QuadraticRegularizer(
    name::Symbol,
    traj::NamedTrajectory,
    R::Real;
    kwargs...
)
    return QuadraticRegularizer(name, traj, R * ones(traj.dims[name]); kwargs...)
end
