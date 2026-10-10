export MinimumTimeObjective

using TrajectoryIndexingUtils

"""
    MinimumTimeObjective <: AbstractObjective

Objective that minimizes total trajectory duration.

Computes:
```math
J = D \\sum_{k=1}^{N-1} \\Delta t_k
```

# Fields
- `D::Float64`: Scaling factor for minimum time objective

# Constructor
```julia
MinimumTimeObjective(traj::NamedTrajectory; D::Float64=1.0)
MinimumTimeObjective(traj::NamedTrajectory, D::Real)
```
"""
struct MinimumTimeObjective <: AbstractObjective
    D::Float64
end

function MinimumTimeObjective(traj::NamedTrajectory; D::Float64 = 1.0)
    @assert traj.timestep isa Symbol || traj.warp !== nothing "MinimumTimeObjective requires a variable timestep or a warp"
    return MinimumTimeObjective(D)
end

# Convenience constructor with D as positional argument
function MinimumTimeObjective(traj::NamedTrajectory, D::Real)
    return MinimumTimeObjective(traj; D = Float64(D))
end

function Base.show(io::IO, obj::MinimumTimeObjective)
    print(io, "MinimumTimeObjective (D = $(round(obj.D, sigdigits=4)))")
end

# Implement AbstractObjective interface

function objective_value(obj::MinimumTimeObjective, traj::NamedTrajectory)
    warp = traj.warp
    if warp !== nothing
        # Phase 1b (DTO#149): under a warp the duration IS the warp parameter —
        # D·T exactly (not a float-y row sum over the derived rows).
        return obj.D * duration(warp)
    end
    duration_val = 0.0
    for k = 1:(traj.N-1)
        duration_val += traj[k].timestep
    end
    return obj.D * duration_val
end

function gradient!(∇::AbstractVector, obj::MinimumTimeObjective, traj::NamedTrajectory)
    fill!(∇, 0.0)

    warp = traj.warp
    if warp !== nothing
        # Phase 1b: the SINGLE chain term — ∂(D·T)/∂θⱼ = D·∂duration/∂θⱼ at the
        # warp-parameter columns (exactly D for GlobalScale).
        warp_cols = WarpPlumbing.warp_param_indices(traj)
        ∂dur = ForwardDiff.gradient(θ -> duration(with_params(warp, θ)), warp_params(warp))
        ∇[warp_cols] .= obj.D .* ∂dur
        return nothing
    end

    @assert traj.timestep isa Symbol "MinimumTimeObjective requires variable timestep"

    for k = 1:(traj.N-1)
        zₖ = traj[k]
        Δt_comps = zₖ.components[traj.timestep]
        Δt_indices = slice(k, Δt_comps, traj.dim)
        ∇[Δt_indices] .= obj.D
    end

    return nothing
end

function hessian_structure(obj::MinimumTimeObjective, traj::NamedTrajectory)
    # Linear objective has no Hessian
    Z_dim = WarpPlumbing.packed_length(traj)
    return spzeros(Z_dim, Z_dim)
end

function get_full_hessian(obj::MinimumTimeObjective, traj::NamedTrajectory)
    Z_dim = WarpPlumbing.packed_length(traj)
    return spzeros(Z_dim, Z_dim)
end

# ============================================================================ #
# Phase 1b (DTO#149): D·T exactly under a warp
# ============================================================================ #

@testitem "MinimumTimeObjective under a warp: D·T exactly, single T gradient term" begin
    include("../../test/test_utils.jl")
    using DirectTrajOpt.Objectives
    using DirectTrajOpt.WarpPlumbing
    using NamedTrajectories
    using FiniteDiff
    using Test

    T0 = 2.5
    traj = warped_derivative_trajectory(N = 6, T = T0; seed = 5)
    obj = MinimumTimeObjective(traj; D = 3.0)

    # the objective IS D·T under a warp — exactly, not via a float-y row sum
    @test objective_value(obj, traj) === 3.0 * T0

    # the gradient carries the single ∂(D·T)/∂T = D term — no per-knot scatter
    ∇ = zeros(length(traj))
    gradient!(∇, obj, traj)
    T_col = WarpPlumbing.warp_param_indices(traj)[1]
    @test ∇[T_col] === 3.0
    @test findall(!iszero, ∇) == [T_col]

    # FD parity over the packed vector
    ∇_fd = FiniteDiff.finite_difference_gradient(collect(vec(traj))) do Z⃗
        t2 = copy(traj)
        unpack!(t2, Z⃗)
        return objective_value(obj, t2)
    end
    @test ∇ ≈ ∇_fd atol = 1e-8

    # linear in T: zero Hessian, packed size
    @test iszero(get_full_hessian(obj, traj))
    @test size(get_full_hessian(obj, traj)) == (length(traj), length(traj))
    @test iszero(DirectTrajOpt.Objectives.hessian_structure(obj, traj))
end
