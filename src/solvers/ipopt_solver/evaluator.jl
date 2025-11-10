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
    objective::AbstractObjective
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
        prob::DirectTrajOptProblem;
        eval_hessian=true,
        verbose=false
    )
        # Create dynamics from integrators
        if verbose
            println("Creating TrajectoryDynamics from $(length(prob.integrators)) integrators...")
        end
        dynamics = TrajectoryDynamics(prob.integrators, prob.trajectory; verbose=verbose)
        
        n_dynamics_constraints = dynamics.dim * (prob.trajectory.N - 1)
        nonlinear_constraints = filter(c -> c isa AbstractNonlinearConstraint, prob.constraints)
        n_nonlinear_constraints = sum(c -> c.dim, nonlinear_constraints; init=0)

        ∂g = Dynamics.get_full_jacobian(dynamics, prob.trajectory)

        for c ∈ nonlinear_constraints 
            ∂g = vcat(∂g, Constraints.get_full_jacobian(c, prob.trajectory))
        end

        jacobian_structure = collect(zip(findnz(∂g)[1:2]...))


        # dynamics hessian structure 
        hessian = dynamics.μ∂²F_structure

        # nonlinear constraints hessian structure
        for con ∈ nonlinear_constraints 
            hessian .+= Constraints.get_full_hessian(con, prob.trajectory)
        end

        # objective hessian structure
        hessian .+= Objectives.get_full_hessian(prob.objective, prob.trajectory)

        hessian_structure = filter(((i, j),) -> i ≤ j, collect(zip(findnz(hessian)[1:2]...)))
        n_constraint_hessian_elements = length(hessian_structure)
        return new(
            prob.trajectory,
            prob.objective,
            dynamics,
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
    # Wrap Z⃗ as NamedTrajectory
    traj_data = Z⃗[1:evaluator.trajectory.dim * evaluator.trajectory.N]
    global_data = Z⃗[evaluator.trajectory.dim * evaluator.trajectory.N + 1:end]
    traj = NamedTrajectory(
        evaluator.trajectory;
        datavec=traj_data,
        global_data=global_data
    )
    
    return Objectives.objective_value(evaluator.objective, traj)
end

@views function MOI.eval_objective_gradient(
    evaluator::IpoptEvaluator,
    ∇::AbstractVector,
    Z⃗::AbstractVector
)
    # Wrap Z⃗ as NamedTrajectory
    traj_data = Z⃗[1:evaluator.trajectory.dim * evaluator.trajectory.N]
    global_data = Z⃗[evaluator.trajectory.dim * evaluator.trajectory.N + 1:end]
    traj = NamedTrajectory(
        evaluator.trajectory;
        datavec=traj_data,
        global_data=global_data
    )
    
    Objectives.gradient!(∇, evaluator.objective, traj)
end


# constraints and Jacobian

@views function MOI.eval_constraint(
    evaluator::IpoptEvaluator,
    g::AbstractVector,
    Z⃗::AbstractVector
)
    evaluator.dynamics.F!(g[1:evaluator.n_dynamics_constraints], Z⃗)

    # Wrap Z⃗ as NamedTrajectory for constraint evaluation
    # Extract trajectory data and global data separately
    traj_data = Z⃗[1:evaluator.trajectory.dim * evaluator.trajectory.N]
    global_data = Z⃗[evaluator.trajectory.dim * evaluator.trajectory.N + 1:end]
    
    traj = NamedTrajectory(
        evaluator.trajectory;
        datavec=traj_data,
        global_data=global_data
    )
    
    # loop over nonlinear constraints, incrementing offset
    offset = evaluator.n_dynamics_constraints
    for con ∈ evaluator.constraints
        g[offset .+ (1:con.dim)] .= Constraints.constraint_value(con, traj)
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
    # Wrap Z⃗ into trajectory for constraint evaluation
    Z = NamedTrajectory(
        evaluator.trajectory;
        datavec=Z⃗[1:evaluator.trajectory.dim * evaluator.trajectory.N],
        global_data=Z⃗[evaluator.trajectory.dim * evaluator.trajectory.N + 1:end]
    )

    evaluator.dynamics.∂F!(evaluator.dynamics.∂fs, Z⃗)

    ∂g = Dynamics.get_full_jacobian(evaluator.dynamics, evaluator.trajectory)

    for c ∈ evaluator.constraints
        Constraints.jacobian!(c, Z)
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

    # wrap datavec with stored trajectory 
    Z = NamedTrajectory(
        evaluator.trajectory;
        datavec=Z⃗[1:evaluator.trajectory.dim * evaluator.trajectory.N],
        global_data=Z⃗[evaluator.trajectory.dim * evaluator.trajectory.N + 1:end]
    )

    # evaluate dynamics hessian lagrangian
    evaluator.dynamics.μ∂²F!(evaluator.dynamics.μ∂²fs, Z⃗, μ[1:evaluator.n_dynamics_constraints])
    ∂²ℒ = Dynamics.get_full_hessian(evaluator.dynamics, evaluator.trajectory)

    # evaluate other nonlinear_constraints
    offset = evaluator.n_dynamics_constraints
    for con ∈ evaluator.constraints
        Constraints.hessian_of_lagrangian!(con, Z, μ[offset .+ (1:con.dim)])
        ∂²ℒ .+= Constraints.get_full_hessian(con, evaluator.trajectory)
        offset += con.dim
    end  

    # evaluate objective hessian
    Objectives.hessian!(evaluator.objective, Z)
    ∂²ℒ .+= σ * Objectives.get_full_hessian(evaluator.objective, Z)

    # fill in H
    H[:] = [∂²ℒ[i, j] for (i, j) ∈ evaluator.hessian_structure]


    return nothing
end

@testitem "testing evaluator" begin
    using FiniteDiff
    import MathOptInterface as MOI

    include("../../../test/test_utils.jl")

    G, traj = bilinear_dynamics_and_trajectory()

    integrators = AbstractIntegrator[
        # BilinearIntegrator(G, :x, :u),
        # DerivativeIntegrator(:u, :du),
        DerivativeIntegrator(:du, :ddu)
    ]

    J = TerminalObjective(x -> norm(x - traj.goal.x)^2, :x, traj)
    J += QuadraticRegularizer(:u, traj, 2.0e-1) 
    J += QuadraticRegularizer(:du, traj, 3.0e-1)
    J += QuadraticRegularizer(:ddu, traj, 4.0e-1)
    J += MinimumTimeObjective(traj)

    display(traj.components)

    g_u_norm = NonlinearKnotPointConstraint(u -> [norm(u) - 1.0], :u, traj; times=2:traj.N-1, equality=false)

    prob = DirectTrajOptProblem(traj, J, integrators; 
        constraints=AbstractConstraint[g_u_norm]
    )

    evaluator = IpoptEvaluator(prob)

    J_val = Objectives.objective_value(J, traj)
    @test MOI.eval_objective(evaluator, traj.datavec) ≈ J_val

    ∇ = zeros(length(traj.datavec))

    ∇L_finitediff = FiniteDiff.finite_difference_gradient(Z⃗ -> MOI.eval_objective(evaluator, Z⃗), traj.datavec)

    MOI.eval_objective_gradient(evaluator, ∇, traj.datavec)

    @test ∇ ≈ ∇L_finitediff

    ĝ = Z⃗ -> begin 
        δ_dynamics = zeros(eltype(Z⃗), evaluator.n_dynamics_constraints) 
        evaluator.dynamics.F!(δ_dynamics, Z⃗)
        δ_nonlinear = zeros(eltype(Z⃗), 0)
        for con ∈ filter(c -> c isa AbstractNonlinearConstraint, evaluator.constraints)
            traj_wrap = NamedTrajectory(Z⃗, evaluator.trajectory.components, evaluator.trajectory.N)
            δ_con = Constraints.constraint_value(con, traj_wrap)
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

    ∂ĝ = dense(∂ĝ_values, ∂ĝ_structure, (evaluator.n_constraints, evaluator.trajectory.dim * evaluator.trajectory.N))

    ∂g_finitediff = FiniteDiff.finite_difference_jacobian(ĝ, traj.datavec)
    @test all(isapprox.(∂g_finitediff, ∂ĝ, atol=1e-6, rtol=1e-6))

    # testing Hessian of the Lagrangian
    μ = 0.1 * ones(evaluator.n_constraints)
    σ = 2.0 

    ∂²ℒ_structure = MOI.hessian_lagrangian_structure(evaluator) 

    ∂²ℒ_values = zeros(length(∂²ℒ_structure))

    for (i, j) ∈ ∂²ℒ_structure
        if j < i
            println("Hessian index: (", i, ", ", j, ")")
        end
    end

    MOI.eval_hessian_lagrangian(evaluator, ∂²ℒ_values, traj.datavec, σ, μ)

    n_vars = evaluator.trajectory.dim * evaluator.trajectory.N + evaluator.trajectory.global_dim

    ∂²ℒ_I = [i for (i, j) ∈ ∂²ℒ_structure]
    ∂²ℒ_J = [j for (i, j) ∈ ∂²ℒ_structure]

    ∂²ℒ = sparse(∂²ℒ_I, ∂²ℒ_J, ∂²ℒ_values, n_vars, n_vars)

    # Collect to avoid LazyArrays issues with FiniteDiff
    Z⃗_vec = collect(traj.datavec)
    ∂²ℒ_finitediff = FiniteDiff.finite_difference_hessian(Z⃗_vec) do Z⃗
        traj_wrap = NamedTrajectory(traj; datavec=Z⃗)
        return σ * Objectives.objective_value(J, traj_wrap) + μ'ĝ(Z⃗)
    end
    
    @test all(isapprox.(triu(∂²ℒ), triu(sparse(∂²ℒ_finitediff)), atol=1e-2))
end
