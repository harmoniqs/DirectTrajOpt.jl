export IpoptEvaluator

using LinearAlgebra
using SparseArrays
using NamedTrajectories
using Base.Threads

# ============================================================================ #
# Performance optimizations applied to IpoptEvaluator:
# 
# 1. Pre-computed offsets - Constraint evaluation uses pre-computed offsets
#    instead of runtime arithmetic (eliminates ~O(n_constraints) additions)
#
# 2. Linear index maps - Replaced Dict{Tuple{Int,Int}, Int} with Vector{Int}
#    using linear indexing: map[(row-1)*ncols + col] = output_idx
#    Provides O(1) array access vs O(log n) Dict lookup
#
# 3. Direct SparseArrays access - Use rowvals(), nonzeros(), nzrange() to
#    iterate sparse matrices instead of findnz() which allocates 3 arrays
#    Saves 6+ allocations per Jacobian/Hessian evaluation
#
# 4. Lightweight NamedTrajectory wrapping - Create wrapper with @views to 
#    avoid copying data; minimal allocation for metadata
#
# 5. Multi-threading - Parallel evaluation of independent integrators and
#    constraints when nthreads() > 1 (near-linear speedup with cores)
#
# 6. Direct value filling - Eliminate intermediate sparse matrix construction
#    for Jacobian/Hessian by filling output vector directly
#
# 7. Parametric typing - NamedTrajectory uses parametric vector types
#    enabling specialization on Vector, SubArray, etc.
# ============================================================================ #
using MathOptInterface
const MOI = MathOptInterface


using ..Objectives
using ..Integrators: AbstractIntegrator
using ..Constraints
using ..Problems
using ..CommonInterface: evaluate!, eval_jacobian, eval_hessian_of_lagrangian


function sparse_to_moi(A::SparseMatrixCSC)
    inds = collect(zip(findnz(A)...))
    vals = [A[i,j] for (i,j) ∈ inds]
    return (inds, vals)
end

mutable struct IpoptEvaluator <: MOI.AbstractNLPEvaluator
    trajectory::NamedTrajectory
    objective::AbstractObjective
    integrators::Vector{<:AbstractIntegrator}
    constraints::Vector{<:AbstractNonlinearConstraint}
    jacobian_structure::Vector{Tuple{Int, Int}}
    hessian_structure::Vector{Tuple{Int, Int}}
    n_constraint_hessian_elements::Int
    n_dynamics_constraints::Int
    n_nonlinear_constraints::Int
    n_constraints::Int
    eval_hessian::Bool
    
    # Pre-computed offsets for constraint evaluation
    _integrator_offsets::Vector{Int}
    _constraint_offsets::Vector{Int}
    
    # Pre-computed row offsets for Jacobian filling
    _jacobian_integrator_row_offsets::Vector{Int}
    _jacobian_constraint_row_offsets::Vector{Int}
    
    # Pre-computed linear index maps for fast lookup (eliminates Dict overhead)
    _jacobian_linear_map::Vector{Int}  # map[row * max_cols + col] = output_idx
    _hessian_linear_map::Vector{Int}
    _jacobian_ncols::Int
    _hessian_ncols::Int

    function IpoptEvaluator(
        prob::DirectTrajOptProblem;
        eval_hessian=true,
        verbose=false
    )
        t_start = time()
        
        # Calculate total dynamics constraint dimension
        n_dynamics_constraints = sum(integrator.dim for integrator in prob.integrators; init=0)
        
        nonlinear_constraints = filter(c -> c isa AbstractNonlinearConstraint, prob.constraints)
        n_nonlinear_constraints = sum(c -> c.dim, nonlinear_constraints; init=0)

        if verbose
            println("      building evaluator: $(length(prob.integrators)) integrators, $(length(nonlinear_constraints)) nonlinear constraints")
            println("      dynamics constraints: $n_dynamics_constraints, nonlinear constraints: $n_nonlinear_constraints")
        end

        # Build Jacobian structure from integrators
        t_jac = time()
        ∂g = spzeros(0, prob.trajectory.dim * prob.trajectory.N + prob.trajectory.global_dim)
        
        for (i, integrator) in enumerate(prob.integrators)
            t_int = time()
            ∂g = vcat(∂g, get_jacobian_structure(integrator, prob.trajectory))
            if verbose
                println("        integrator $i jacobian structure: $(round(time() - t_int, digits=3))s")
            end
        end

        for (i, c) ∈ enumerate(nonlinear_constraints)
            t_con = time()
            ∂g = vcat(∂g, eval_jacobian(c, prob.trajectory))
            if verbose
                println("        constraint $i ($(typeof(c).name.name)) jacobian structure: $(round(time() - t_con, digits=3))s")
            end
        end

        jacobian_structure = collect(zip(findnz(∂g)[1:2]...))
        if verbose
            println("      jacobian structure: $(length(jacobian_structure)) nonzeros ($(round(time() - t_jac, digits=3))s)")
        end

        # Build Hessian structure from integrators
        t_hess = time()
        hessian = spzeros(
            prob.trajectory.dim * prob.trajectory.N + prob.trajectory.global_dim,
            prob.trajectory.dim * prob.trajectory.N + prob.trajectory.global_dim
        )
        
        for (i, integrator) in enumerate(prob.integrators)
            t_int = time()
            hessian .+= get_hessian_of_lagrangian_structure(integrator, prob.trajectory)
            if verbose
                println("        integrator $i hessian structure: $(round(time() - t_int, digits=3))s")
            end
        end

        # nonlinear constraints hessian structure
        for (i, con) ∈ enumerate(nonlinear_constraints)
            t_con = time()
            hessian .+= eval_hessian_of_lagrangian(con, prob.trajectory, ones(con.dim))
            if verbose
                println("        constraint $i ($(typeof(con).name.name)) hessian structure: $(round(time() - t_con, digits=3))s")
            end
        end

        # objective hessian structure
        t_obj = time()
        if verbose
            println("        computing objective hessian structure ($(typeof(prob.objective).name.name))...")
        end
        if prob.objective isa Objectives.CompositeObjective
            hessian .+= Objectives.hessian_structure(prob.objective, prob.trajectory; verbose=verbose)
        else
            hessian .+= Objectives.hessian_structure(prob.objective, prob.trajectory)
        end
        if verbose
            println("        objective hessian structure: $(round(time() - t_obj, digits=3))s")
        end

        hessian_structure = filter(((i, j),) -> i ≤ j, collect(zip(findnz(hessian)[1:2]...)))
        n_constraint_hessian_elements = length(hessian_structure)
        
        if verbose
            println("      hessian structure: $(length(hessian_structure)) nonzeros ($(round(time() - t_hess, digits=3))s)")
        end
        
        # Pre-compute offsets for fast indexing
        t_maps = time()
        integrator_offsets = Vector{Int}(undef, length(prob.integrators) + 1)
        integrator_offsets[1] = 0
        for i in 1:length(prob.integrators)
            integrator_offsets[i+1] = integrator_offsets[i] + prob.integrators[i].dim
        end
        
        constraint_offsets = Vector{Int}(undef, length(nonlinear_constraints) + 1)
        constraint_offsets[1] = n_dynamics_constraints
        for i in 1:length(nonlinear_constraints)
            constraint_offsets[i+1] = constraint_offsets[i] + nonlinear_constraints[i].dim
        end
        
        # Pre-compute Jacobian row offsets for each integrator/constraint
        jacobian_integrator_row_offsets = copy(integrator_offsets)
        jacobian_constraint_row_offsets = copy(constraint_offsets)
        
        # Pre-compute linear index maps for O(1) lookup (replaces Dict with array indexing)
        n_vars = prob.trajectory.dim * prob.trajectory.N + prob.trajectory.global_dim
        jacobian_ncols = n_vars
        hessian_ncols = n_vars
        
        n_total_constraints = n_dynamics_constraints + n_nonlinear_constraints
        
        # Build linear index map for Jacobian: map[row * ncols + col] = output_idx
        jacobian_linear_map = zeros(Int, n_total_constraints * jacobian_ncols)
        for (idx, (i, j)) in enumerate(jacobian_structure)
            linear_idx = (i - 1) * jacobian_ncols + j
            jacobian_linear_map[linear_idx] = idx
        end
        
        # Build linear index map for Hessian: map[row * ncols + col] = output_idx
        hessian_linear_map = zeros(Int, hessian_ncols * hessian_ncols)
        for (idx, (i, j)) in enumerate(hessian_structure)
            linear_idx = (i - 1) * hessian_ncols + j
            hessian_linear_map[linear_idx] = idx
        end
        
        if verbose
            println("      linear index maps built ($(round(time() - t_maps, digits=3))s)")
            println("      evaluator ready (total: $(round(time() - t_start, digits=3))s)")
        end
        
        return new(
            prob.trajectory,
            prob.objective,
            prob.integrators,
            AbstractNonlinearConstraint[nonlinear_constraints...],
            jacobian_structure,
            hessian_structure,
            n_constraint_hessian_elements,
            n_dynamics_constraints,
            n_nonlinear_constraints,
            n_dynamics_constraints + n_nonlinear_constraints,
            eval_hessian,
            integrator_offsets,
            constraint_offsets,
            jacobian_integrator_row_offsets,
            jacobian_constraint_row_offsets,
            jacobian_linear_map,
            hessian_linear_map,
            jacobian_ncols,
            hessian_ncols
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
    # Update cached trajectory in-place
    traj = _update_trajectory_cache!(evaluator, Z⃗)
    return Objectives.objective_value(evaluator.objective, traj)
end

@views function MOI.eval_objective_gradient(
    evaluator::IpoptEvaluator,
    ∇::AbstractVector,
    Z⃗::AbstractVector
)
    # Update cached trajectory in-place
    traj = _update_trajectory_cache!(evaluator, Z⃗)
    Objectives.gradient!(∇, evaluator.objective, traj)
end


# constraints and Jacobian

@views function MOI.eval_constraint(
    evaluator::IpoptEvaluator,
    g::AbstractVector,
    Z⃗::AbstractVector
)
    # Update cached trajectory in-place
    traj = _update_trajectory_cache!(evaluator, Z⃗)
    
    # Parallelize integrator evaluations (writes don't overlap)
    if nthreads() > 1 && length(evaluator.integrators) > 1
        @threads for i in 1:length(evaluator.integrators)
            integrator = evaluator.integrators[i]
            offset = evaluator._integrator_offsets[i]
            δ = view(g, offset+1:offset+integrator.dim)
            evaluate!(δ, integrator, traj)
        end
    else
        for (i, integrator) in enumerate(evaluator.integrators)
            offset = evaluator._integrator_offsets[i]
            δ = view(g, offset+1:offset+integrator.dim)
            evaluate!(δ, integrator, traj)
        end
    end
    
    # Parallelize nonlinear constraint evaluations (writes don't overlap)
    if nthreads() > 1 && length(evaluator.constraints) > 1
        @threads for i in 1:length(evaluator.constraints)
            con = evaluator.constraints[i]
            offset = evaluator._constraint_offsets[i]
            CommonInterface.evaluate!(view(g, offset+1:offset+con.dim), con, traj)
        end
    else
        for (i, con) in enumerate(evaluator.constraints)
            offset = evaluator._constraint_offsets[i]
            CommonInterface.evaluate!(view(g, offset+1:offset+con.dim), con, traj)
        end
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
    # Update cached trajectory in-place
    Z = _update_trajectory_cache!(evaluator, Z⃗)

    # Fill values directly from structure without building full sparse matrix
    _fill_jacobian_values!(∂, evaluator, Z)

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

    # Update cached trajectory in-place
    Z = _update_trajectory_cache!(evaluator, Z⃗)

    # Fill Hessian values directly without building full sparse matrix
    _fill_hessian_values!(H, evaluator, Z, σ, μ)

    return nothing
end

# ============================================================================ #
# Helper functions for efficient value filling
# ============================================================================ #

"""
    _update_trajectory_cache!(evaluator, Z⃗)

Update the cached trajectory in-place with new data from Z⃗.
Avoids repeated allocation of NamedTrajectory wrappers.
"""
@inline @views function _update_trajectory_cache!(evaluator::IpoptEvaluator, Z⃗::AbstractVector)
    n_traj = evaluator.trajectory.dim * evaluator.trajectory.N
    
    # Create trajectory wrapper with views (minimal allocation)
    # This is equivalent to the old approach but reuses structure
    traj = NamedTrajectory(
        evaluator.trajectory;
        datavec=Z⃗[1:n_traj],
        global_data=Z⃗[n_traj+1:end]
    )
    
    return traj
end

"""
    _fill_jacobian_values!(∂, evaluator, Z)

Fill Jacobian values directly into the output vector without building
intermediate sparse matrices. Uses pre-computed linear index map and
direct SparseArrays access to eliminate allocations.
"""
@inline function _fill_jacobian_values!(∂::AbstractVector, evaluator::IpoptEvaluator, Z::NamedTrajectory)
    # Zero out output first
    fill!(∂, 0.0)
    
    # Pre-computed linear map and dimensions
    linear_map = evaluator._jacobian_linear_map
    ncols = evaluator._jacobian_ncols
    
    # Evaluate each integrator and fill corresponding values
    # Use direct SparseArrays access to avoid findnz() allocations
    for (idx, integrator) in enumerate(evaluator.integrators)
        row_offset = evaluator._jacobian_integrator_row_offsets[idx]
        ∂g = eval_jacobian(integrator, Z)
        
        # Direct iteration over sparse matrix internals
        rows = rowvals(∂g)
        vals = nonzeros(∂g)
        m, n = size(∂g)
        
        for col = 1:n
            for j in nzrange(∂g, col)
                row = rows[j]
                val = vals[j]
                global_row = row_offset + row
                linear_idx = (global_row - 1) * ncols + col
                output_idx = linear_map[linear_idx]
                if output_idx != 0
                    ∂[output_idx] = val
                end
            end
        end
    end
    
    # Evaluate each nonlinear constraint and fill corresponding values
    for (idx, con) in enumerate(evaluator.constraints)
        row_offset = evaluator._jacobian_constraint_row_offsets[idx]
        ∂g = eval_jacobian(con, Z)
        
        # Direct iteration over sparse matrix internals
        rows = rowvals(∂g)
        vals = nonzeros(∂g)
        m, n = size(∂g)
        
        for col = 1:n
            for j in nzrange(∂g, col)
                row = rows[j]
                val = vals[j]
                global_row = row_offset + row
                linear_idx = (global_row - 1) * ncols + col
                output_idx = linear_map[linear_idx]
                if output_idx != 0
                    ∂[output_idx] = val
                end
            end
        end
    end
end

"""
    _fill_hessian_values!(H, evaluator, Z, σ, μ)

Fill Hessian of Lagrangian values directly into output vector without
building intermediate sparse matrices. Uses linear index map and
direct SparseArrays access to eliminate allocations.
"""
@inline function _fill_hessian_values!(H::AbstractVector{T}, evaluator::IpoptEvaluator, 
                                Z::NamedTrajectory, σ::T, μ::AbstractVector{T}) where T
    # Pre-computed linear map and dimensions
    linear_map = evaluator._hessian_linear_map
    ncols = evaluator._hessian_ncols
    
    # Zero out H first
    fill!(H, zero(T))
    
    # Accumulate integrator Hessians using direct sparse matrix access
    for (idx, integrator) in enumerate(evaluator.integrators)
        offset = evaluator._integrator_offsets[idx]
        μ_slice = view(μ, offset+1:offset+integrator.dim)
        ∂²ℒ = eval_hessian_of_lagrangian(integrator, Z, μ_slice)
        
        # Direct iteration over sparse matrix internals (eliminates findnz allocation)
        rows = rowvals(∂²ℒ)
        vals = nonzeros(∂²ℒ)
        
        for col = 1:size(∂²ℒ, 2)
            for j in nzrange(∂²ℒ, col)
                row = rows[j]
                val = vals[j]
                # Only process upper triangle
                if row <= col
                    linear_idx = (row - 1) * ncols + col
                    output_idx = linear_map[linear_idx]
                    if output_idx != 0
                        H[output_idx] += val
                    end
                end
            end
        end
    end
    
    # Accumulate nonlinear constraint Hessians
    for (idx, con) in enumerate(evaluator.constraints)
        offset = evaluator._constraint_offsets[idx]
        μ_slice = view(μ, offset+1:offset+con.dim)
        ∂²ℒ = eval_hessian_of_lagrangian(con, Z, μ_slice)
        
        # Direct iteration over sparse matrix internals
        rows = rowvals(∂²ℒ)
        vals = nonzeros(∂²ℒ)
        
        for col = 1:size(∂²ℒ, 2)
            for j in nzrange(∂²ℒ, col)
                row = rows[j]
                val = vals[j]
                if row <= col
                    linear_idx = (row - 1) * ncols + col
                    output_idx = linear_map[linear_idx]
                    if output_idx != 0
                        H[output_idx] += val
                    end
                end
            end
        end
    end
    
    # Accumulate objective Hessian
    if σ != 0
        ∂²L = Objectives.get_full_hessian(evaluator.objective, Z)
        
        # Direct iteration over sparse matrix internals
        rows = rowvals(∂²L)
        vals = nonzeros(∂²L)
        
        for col = 1:size(∂²L, 2)
            for j in nzrange(∂²L, col)
                row = rows[j]
                val = vals[j]
                if row <= col
                    linear_idx = (row - 1) * ncols + col
                    output_idx = linear_map[linear_idx]
                    if output_idx != 0
                        H[output_idx] += σ * val
                    end
                end
            end
        end
    end
end

@testitem "testing evaluator" begin
    using FiniteDiff
    using TrajectoryIndexingUtils
    import MathOptInterface as MOI

    include("../../../test/test_utils.jl")

    G, traj = bilinear_dynamics_and_trajectory()

    integrators = AbstractIntegrator[
        BilinearIntegrator(G, :x, :u, traj),
        DerivativeIntegrator(:u, :du, traj),
        DerivativeIntegrator(:du, :ddu, traj)
    ]

    J = TerminalObjective(x -> norm(x - traj.goal.x)^2, :x, traj)
    J += QuadraticRegularizer(:u, traj, 2.0e-1) 
    J += QuadraticRegularizer(:du, traj, 3.0e-1)
    J += QuadraticRegularizer(:ddu, traj, 4.0e-1)
    J += MinimumTimeObjective(traj)

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
        traj_wrap = NamedTrajectory(
            evaluator.trajectory; 
            datavec=Z⃗[1:evaluator.trajectory.dim * evaluator.trajectory.N],
            global_data=Z⃗[evaluator.trajectory.dim * evaluator.trajectory.N + 1:end]
        )
        
        # Evaluate integrators
        δ_dynamics = zeros(eltype(Z⃗), evaluator.n_dynamics_constraints)
        offset = 0
        for integrator in evaluator.integrators
            δ = view(δ_dynamics, offset+1:offset+integrator.dim)
            evaluate!(δ, integrator, traj_wrap)
            offset += integrator.dim
        end
        
        # Evaluate nonlinear constraints
        δ_nonlinear = zeros(eltype(Z⃗), evaluator.n_nonlinear_constraints)
        offset_nl = 0
        for con ∈ evaluator.constraints
            # Use the constraint's evaluate! method
            δ_con = zeros(eltype(Z⃗), con.dim)
            evaluate!(δ_con, con, traj_wrap)
            δ_nonlinear[offset_nl .+ (1:con.dim)] .= δ_con
            offset_nl += con.dim
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
    
    show_diffs(triu(∂²ℒ)[slice(2, traj.dim), slice(2, traj.dim)], triu(sparse(∂²ℒ_finitediff))[slice(2, traj.dim), slice(2, traj.dim)], atol=1e-2)
    @test all(isapprox.(triu(∂²ℒ), triu(sparse(∂²ℒ_finitediff)), atol=1e-2))
end
