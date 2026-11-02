export AbstractBilinearIntegrator
export BilinearIntegrator

using ExponentialAction
using TrajectoryIndexingUtils
using ..Integrators

# -------------------------------------------------------------------------------- #
# Abstract Bilinear Integrator
# -------------------------------------------------------------------------------- #

abstract type AbstractBilinearIntegrator <: AbstractIntegrator end

# -------------------------------------------------------------------------------- #
# Bilinear Integrator
# -------------------------------------------------------------------------------- #

"""
    BilinearIntegrator <: AbstractBilinearIntegrator

Integrator for control-linear dynamics of the form ẋ = G(u)x.

This integrator uses matrix exponential methods to compute accurate state transitions for
bilinear systems where the system matrix depends linearly on the control input.

# Fields
- `G::Function`: Function mapping control u to system matrix G(u)
- `x_name::Symbol`: Primary state variable name (the first of `x_names`)
- `x_names::Vector{Symbol}`: All state variable names this integrator stacks (single-element for the single-state form)
- `u_name::Symbol`: Control variable name
- `x_dim::Int`: Dimension of the (stacked) state variable
- `var_dim::Int`: Combined dimension of all variables this integrator depends on (2*x_dim + u_dim + 1)
- `dim::Int`: Total constraint dimension (x_dim * (N-1))
- `∂fs::Vector{SparseMatrixCSC{Float64, Int}}`: Pre-allocated compact Jacobian storage (x_dim × var_dim per timestep)
- `μ∂²fs::Vector{SparseMatrixCSC{Float64, Int}}`: Pre-allocated compact Hessian storage (var_dim × var_dim per timestep)

# Constructors
```julia
BilinearIntegrator(G::Function, x::Symbol, u::Symbol, traj::NamedTrajectory)
BilinearIntegrator(G::Function, xs::AbstractVector{Symbol}, u::Symbol, traj::NamedTrajectory)
```

# Arguments
- `G`: Function taking control u and returning the (stacked) state matrix (x_dim × x_dim)
- `x`: State variable name
- `xs`: State variable names; the integrated state is the concatenation of these components
- `u`: Control variable name
- `traj`: Trajectory structure used to determine dimensions and pre-allocate storage

# Dynamics
Computes the constraint: x_{k+1} - exp(Δt * G(u_k)) * x_k = 0
Dependencies: xₖ, uₖ, Δtₖ, xₖ₊₁

# Example
```julia
# Linear dynamics: ẋ = (A + Σᵢ uᵢ Bᵢ) x
A = [-0.1 1.0; -1.0 -0.1]
B = [0.0 0.0; 0.0 1.0]
G = u -> A + u[1] * B

integrator = BilinearIntegrator(G, :x, :u, traj)
```
"""
struct BilinearIntegrator{F} <: AbstractBilinearIntegrator
    f::F
    x_name::Symbol
    x_names::Vector{Symbol}
    u_name::Symbol
    x_dim::Int
    var_dim::Int
    dim::Int

    function BilinearIntegrator(
        G::Function,
        xs::AbstractVector{Symbol},
        u::Symbol,
        traj::NamedTrajectory,
    )
        # Phase 1b scope amendment (DTO#149): BilinearIntegrator is on the demotion
        # arc (open-core split — default dispatch flips to the Hermitian exponential
        # family). It got NO warp plumbing; a warp trajectory would produce silently
        # wrong sensitivities. Refuse instead.
        traj.warp !== nothing && throw(
            ArgumentError(
                "BilinearIntegrator does not support time-warp trajectories: it is on the " *
                "demotion arc (open-core split — default dispatch flips to the Hermitian " *
                "exponential family) and carries no warp column by scope (DirectTrajOpt#149 " *
                "amendment). Derived-Δt dynamics live with the surviving integrator family " *
                "(HermitianExponentialIntegrator, Piccolo.jl#321).",
            ),
        )
        length(xs) > 0 || throw(ArgumentError("xs must contain at least one state name"))
        x_names = collect(Symbol, xs)
        x_dim = sum(traj.dims[x] for x in x_names)
        u_dim = traj.dims[u]
        N = traj.N

        # Variables: [xₖ, uₖ, Δtₖ, xₖ₊₁]
        var_dim = x_dim + u_dim + 1 + x_dim  # = 2*x_dim + u_dim + 1

        # Total constraint dimension
        dim = x_dim * (N - 1)

        # Define f function: constraint is f(xₖ₊₁, xₖ, uₖ, Δtₖ) = 0
        f = (xₖ₊₁, xₖ, uₖ, Δtₖ) -> xₖ₊₁ - expv(Δtₖ, G(uₖ), xₖ)

        return new{typeof(f)}(f, x_names[1], x_names, u, x_dim, var_dim, dim)
    end
end

function BilinearIntegrator(G::Function, x::Symbol, u::Symbol, traj::NamedTrajectory)
    return BilinearIntegrator(G, [x], u, traj)
end

function _stacked_components(traj::NamedTrajectory, x_names::AbstractVector{Symbol})
    return reduce(vcat, (traj.components[x] for x in x_names))
end

function Base.show(io::IO, B::BilinearIntegrator)
    names = join((":$n" for n in B.x_names), ", ")
    print(
        io,
        "BilinearIntegrator: [$names] = exp(Δt G(:$(B.u_name))) [$names]  (dim = $(B.x_dim))",
    )
end

# -------------------------------------------------------------------------------- #
# Methods
# -------------------------------------------------------------------------------- #

@views function evaluate!(δ::AbstractVector, B::BilinearIntegrator, traj::NamedTrajectory)
    x_comps = _stacked_components(traj, B.x_names)
    for k = 1:(traj.N-1)
        xₖ = traj[k].data[x_comps]
        xₖ₊₁ = traj[k+1].data[x_comps]
        uₖ = traj[k][B.u_name]
        Δtₖ = traj[k].timestep
        δ[slice(k, B.x_dim)] = B.f(xₖ₊₁, xₖ, uₖ, Δtₖ)
    end
    return nothing
end

# Jacobian methods

@views function eval_jacobian(B::AbstractBilinearIntegrator, traj::NamedTrajectory)
    ∂B = spzeros(B.dim, traj.dim * traj.N + traj.global_dim)
    x_comps = _stacked_components(traj, B.x_names)
    for k = 1:(traj.N-1)
        ForwardDiff.jacobian!(
            ∂B[slice(k, B.x_dim), slice(k, 1:2traj.dim, traj.dim)],
            zz -> begin
                zₖ₊₁ = zz[(traj.dim+1):end]
                zₖ = zz[1:traj.dim]

                xₖ₊₁ = zₖ₊₁[x_comps]
                xₖ = zₖ[x_comps]
                uₖ = zₖ[traj.components[B.u_name]]
                Δtₖ = zₖ[traj.components[traj.timestep]][1]

                return B.f(xₖ₊₁, xₖ, uₖ, Δtₖ)
            end,
            [traj[k].data; traj[k+1].data],
        )
    end
    return ∂B
end

# Hessian methods

function eval_hessian_of_lagrangian(
    B::AbstractBilinearIntegrator,
    traj::NamedTrajectory,
    μ::AbstractVector,
)
    μ∂²B = spzeros(traj.dim * traj.N + traj.global_dim, traj.dim * traj.N + traj.global_dim)
    x_comps = _stacked_components(traj, B.x_names)

    for k = 1:(traj.N-1)
        μₖ = μ[slice(k, B.x_dim)]

        μ∂²Bₖ = ForwardDiff.hessian(
            zz -> begin
                zₖ = zz[1:traj.dim]
                zₖ₊₁ = zz[(traj.dim+1):end]
                xₖ = zₖ[x_comps]
                uₖ = zₖ[traj.components[B.u_name]]
                Δtₖ = zₖ[traj.components[traj.timestep]][1]
                xₖ₊₁ = zₖ₊₁[x_comps]
                return μₖ'B.f(xₖ₊₁, xₖ, uₖ, Δtₖ)
            end,
            [traj[k].data; traj[k+1].data],
        )

        μ∂²B[slice(k, 1:2traj.dim, traj.dim), slice(k, 1:2traj.dim, traj.dim)] .+= μ∂²Bₖ
    end
    return μ∂²B
end

# -------------------------------------------------------------------------------- #
# Tests
# -------------------------------------------------------------------------------- #

@testitem "testing BilinearIntegrator" begin
    include("../../test/test_utils.jl")

    G, traj = bilinear_dynamics_and_trajectory()

    B = BilinearIntegrator(G, :x, :u, traj)

    test_integrator(B, traj, atol = 1e-3)
end

@testitem "testing multi-state BilinearIntegrator" begin
    include("../../test/test_utils.jl")
    using DirectTrajOpt: Solvers

    # Reference: the standard fixture, one 4-dim state component :x.
    G, traj_ref = bilinear_dynamics_and_trajectory()

    # Split: the same state data as two 2-dim components (:a, :b) occupying the
    # same flat-layout slots, so the stacked state is x = [a; b] and the flat
    # data vectors of both trajectories coincide.
    x_data = traj_ref[:x]
    traj_split = NamedTrajectory(
        (
            a = x_data[1:2, :],
            b = x_data[3:4, :],
            u = traj_ref[:u],
            du = traj_ref[:du],
            ddu = traj_ref[:ddu],
            Δt = traj_ref[:Δt],
        );
        controls = (:ddu, :Δt),
        timestep = :Δt,
    )

    B_single = BilinearIntegrator(G, :x, :u, traj_ref)
    B_multi = BilinearIntegrator(G, [:a, :b], :u, traj_split)

    @test B_multi.x_names == [:a, :b]
    @test B_multi.x_name == :a
    @test B_multi.x_dim == B_single.x_dim == 4
    @test B_multi.dim == B_single.dim

    # The generic integrator gates on the split trajectory
    test_integrator(B_multi, traj_split, atol = 1e-3)

    # Direct equivalence with the single-component reference: identical
    # residuals, Jacobians, and Lagrangian Hessians on coinciding flat data.
    δ_multi = zeros(B_multi.dim)
    δ_single = zeros(B_single.dim)
    evaluate!(δ_multi, B_multi, traj_split)
    evaluate!(δ_single, B_single, traj_ref)
    @test δ_multi ≈ δ_single atol = 1e-10

    J_multi = eval_jacobian(B_multi, traj_split)
    J_single = eval_jacobian(B_single, traj_ref)
    @test J_multi ≈ J_single atol = 1e-8

    μ = 2.0 .* (1:B_multi.dim)
    H_multi = eval_hessian_of_lagrangian(B_multi, traj_split, μ)
    H_single = eval_hessian_of_lagrangian(B_single, traj_ref, μ)
    @test H_multi ≈ H_single atol = 1e-8

    # The stacked dimension feeds get_nonlinear_constraints through x_names
    # (both fields present — x_names must win over x_name).
    prob_split = DirectTrajOptProblem(
        traj_split,
        QuadraticRegularizer(:u, traj_split, 1.0),
        [B_multi],
    )
    prob_ref =
        DirectTrajOptProblem(traj_ref, QuadraticRegularizer(:u, traj_ref, 1.0), [B_single])
    @test length(Solvers.get_nonlinear_constraints(prob_split)) ==
          length(Solvers.get_nonlinear_constraints(prob_ref))
end
