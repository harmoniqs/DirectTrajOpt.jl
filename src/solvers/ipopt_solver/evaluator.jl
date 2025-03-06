export IpoptEvaluator

using LinearAlgebra
using SparseArrays
using NamedTrajectories
using MathOptInterface
const MOI = MathOptInterface


using ..Objectives
using ..Integrators
using ..Dynamics
using ..Constraints
using ..Problems


function sparse_to_moi(A::SparseMatrixCSC)
    inds = collect(zip(findnz(A)...))
    vals = [A[i,j] for (i,j) ∈ inds]
    return (inds, vals)
end

mutable struct IpoptEvaluator <: MOI.AbstractNLPEvaluator
    trajectory::NamedTrajectory
    objective::Objective
    dynamics::TrajectoryDynamics
    constraints::Vector{<:AbstractNonlinearConstraint}
    jacobian_structure::Vector{Tuple{Int, Int}}
    hessian_structure::Vector{Tuple{Int, Int}}
    n_constraint_hessian_elements::Int
    n_dynamics_constraints::Int
    n_nonlinear_constraints::Int
    n_constraints::Int
    eval_hessian::Bool

    function IpoptEvaluator(
        prob::DirectCollocationProblem;
        eval_hessian=true
    )
        n_dynamics_constraints = prob.dynamics.dim * (prob.trajectory.T - 1)
        nonlinear_constraints = filter(c -> c isa AbstractNonlinearConstraint, prob.constraints)
        n_nonlinear_constraints = sum(c -> c.dim, nonlinear_constraints; init=0)

        ∂g = Dynamics.get_full_jacobian(prob.dynamics, prob.trajectory)

        for c ∈ nonlinear_constraints 
            ∂g = vcat(∂g, Constraints.get_full_jacobian(c, prob.trajectory))
        end

        jacobian_structure = collect(zip(findnz(∂g)[1:2]...))


        # dynamics hessian structure 
        hessian = prob.dynamics.μ∂²F_structure

        # nonlinear constraints hessian structure
        for con ∈ nonlinear_constraints 
            hessian .+= Constraints.get_full_hessian(con, prob.trajectory)
        end

        hessian_structure = collect(zip(findnz(hessian)[1:2]...))
        n_constraint_hessian_elements = length(hessian_structure)

        # objective hessian structure
        hessian_structure = vcat(
            hessian_structure, 
            prob.objective.∂²L_structure()
        )

        return new(
            prob.trajectory,
            prob.objective,
            prob.dynamics,
            AbstractNonlinearConstraint[nonlinear_constraints...],
            jacobian_structure,
            hessian_structure,
            n_constraint_hessian_elements,
            n_dynamics_constraints,
            n_nonlinear_constraints,
            n_dynamics_constraints + n_nonlinear_constraints,
            eval_hessian
        )
    end
end

MOI.initialize(::IpoptEvaluator, features) = nothing

function MOI.features_available(evaluator::IpoptEvaluator)
    if evaluator.eval_hessian 
        return [:Grad, :Jac, :Hess]
    else
        return [:Grad, :Jac]
    end
end


# objective and gradient

@views function MOI.eval_objective(
    evaluator::IpoptEvaluator,
    Z⃗::AbstractVector
)
    return evaluator.objective.L(Z⃗)
end

@views function MOI.eval_objective_gradient(
    evaluator::IpoptEvaluator,
    ∇::AbstractVector,
    Z⃗::AbstractVector
)
    ∇[:] = evaluator.objective.∇L(Z⃗)
end


# constraints and Jacobian

@views function MOI.eval_constraint(
    evaluator::IpoptEvaluator,
    g::AbstractVector,
    Z⃗::AbstractVector
)
    evaluator.dynamics.F!(g[1:evaluator.n_dynamics_constraints], Z⃗)

    # loop over nonlinear constraints, incrementing offset
    offset = evaluator.n_dynamics_constraints
    for con ∈ evaluator.constraints
        con.g!(g[offset .+ (1:con.dim)], Z⃗)
        offset += con.dim
    end

    return nothing
end

function MOI.jacobian_structure(evaluator::IpoptEvaluator)
    return evaluator.jacobian_structure
end

@views function MOI.eval_constraint_jacobian(
    evaluator::IpoptEvaluator,
    ∂::AbstractVector,
    Z⃗::AbstractVector
)
    evaluator.dynamics.∂F!(evaluator.dynamics.∂fs, Z⃗)

    ∂g = Dynamics.get_full_jacobian(evaluator.dynamics, evaluator.trajectory)

    for c ∈ evaluator.constraints
        c.∂g!(c.∂gs, Z⃗)
        ∂g = vcat(∂g, Constraints.get_full_jacobian(c, evaluator.trajectory))
    end 

    ∂[:] = [∂g[i, j] for (i, j) ∈ MOI.jacobian_structure(evaluator)] 

    return nothing
end


# Hessian of the Lagrangian

function MOI.hessian_lagrangian_structure(evaluator::IpoptEvaluator)
    return evaluator.hessian_structure
end

@views function MOI.eval_hessian_lagrangian(
    evaluator::IpoptEvaluator,
    H::AbstractVector{T},
    Z⃗::AbstractVector{T},
    σ::T,
    μ::AbstractVector{T}
) where T

    evaluator.dynamics.μ∂²F!(
        evaluator.dynamics.μ∂²fs, 
        Z⃗, 
        μ[1:evaluator.n_dynamics_constraints]
    )

    ∂²ℒ = Dynamics.get_full_hessian(evaluator.dynamics, evaluator.trajectory)

    # reset stored hessians
    for μ∂²f ∈ evaluator.dynamics.μ∂²fs
        μ∂²f .= 0.0
    end

    offset = evaluator.n_dynamics_constraints
    for con ∈ evaluator.constraints

        con.μ∂²g!(con.μ∂²gs, Z⃗, μ[offset .+ (1:con.dim)])

        ∂²ℒ .+= Constraints.get_full_hessian(con, evaluator.trajectory)

        # reset stored hessians
        for μ∂²g ∈ con.μ∂²gs
            μ∂²g .= 0.0
        end
        
        offset += con.dim
    end  

    H[1:evaluator.n_constraint_hessian_elements] = 
        [∂²ℒ[i, j] for (i, j) ∈ evaluator.hessian_structure[1:evaluator.n_constraint_hessian_elements]]

    H[evaluator.n_constraint_hessian_elements+1:end] = 
        σ * evaluator.objective.∂²L(Z⃗)

    return nothing
end

@testitem "testing evaluator" begin
    import MathOptInterface as MOI

    include("../../../test/test_utils.jl")

    G, traj = bilinear_dynamics_and_trajectory()

    integrators = [
        BilinearIntegrator(G, traj, :x, :u, :Δt),
        DerivativeIntegrator(traj, :u, :du),
        DerivativeIntegrator(traj, :du, :ddu)
    ]

    J = TerminalLoss(x -> norm(x - traj.goal.x)^2, :x, traj)
    J += QuadraticRegularizer(:u, traj, 1.0) 
    J += QuadraticRegularizer(:du, traj, 1.0)
    J += MinimumTimeObjective(traj)

    g_u_norm = NonlinearKnotPointConstraint(u -> [norm(u) - 1.0], :u, traj; times=2:traj.T-1, equality=false)

    prob = DirectCollocationProblem(traj, J, integrators; constraints=AbstractConstraint[g_u_norm])

    evaluator = IpoptEvaluator(prob)

    @test MOI.eval_objective(evaluator, traj.datavec) ≈ J.L(traj.datavec)

    ∇ = zeros(length(traj.datavec))

    ∇L_autodiff = ForwardDiff.gradient(J.L, traj.datavec)

    MOI.eval_objective_gradient(evaluator, ∇, traj.datavec)

    @test ∇ ≈ ∇L_autodiff

    ĝ = Z⃗ -> begin 
        δ_dynamics = zeros(eltype(Z⃗), evaluator.n_dynamics_constraints) 
        evaluator.dynamics.F!(δ_dynamics, Z⃗)
        δ_nonlinear = zeros(eltype(Z⃗), 0)
        for con ∈ filter(c -> c isa AbstractNonlinearConstraint, evaluator.constraints)
            δ_con = zeros(eltype(Z⃗), con.dim)
            con.g!(δ_con, Z⃗)
            δ_nonlinear = vcat(δ_nonlinear, δ_con)
        end
        return vcat(δ_dynamics, δ_nonlinear)
    end

    # testing constraint evaluation
    g = zeros(evaluator.n_constraints)

    MOI.eval_constraint(evaluator, g, traj.datavec)

    @test g ≈ ĝ(traj.datavec)

    # testing constraint Jacobian
    ∂ĝ_structure = MOI.jacobian_structure(evaluator)

    ∂ĝ_values = zeros(length(∂ĝ_structure)) 

    MOI.eval_constraint_jacobian(evaluator, ∂ĝ_values, traj.datavec)

    ∂ĝ = dense(∂ĝ_values, ∂ĝ_structure, (evaluator.n_constraints, evaluator.trajectory.dim * evaluator.trajectory.T))

    ∂g_autodiff = ForwardDiff.jacobian(ĝ, traj.datavec)
    @test all(∂g_autodiff .≈ ∂ĝ)

    # testing Hessian of the Lagrangian
    μ = randn(evaluator.n_constraints)
    σ = randn()

    ∂²ℒ_structure = MOI.hessian_lagrangian_structure(evaluator) 

    ∂²ℒ_values = zeros(length(∂²ℒ_structure))

    MOI.eval_hessian_lagrangian(evaluator, ∂²ℒ_values, traj.datavec, σ, μ)

    ∂²ℒ = dense(∂²ℒ_values, ∂²ℒ_structure, (evaluator.trajectory.dim * evaluator.trajectory.T, evaluator.trajectory.dim * evaluator.trajectory.T))

    ∂²ℒ_autodiff = ForwardDiff.hessian(Z⃗ -> σ * J.L(Z⃗) + μ'ĝ(Z⃗), traj.datavec)
    
    @test all(isapprox(∂²ℒ, ∂²ℒ_autodiff, atol=1e-7))
end
