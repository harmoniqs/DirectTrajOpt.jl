module Constraints

export AbstractConstraint

export AbstractLinearConstraint
export AbstractNonlinearConstraint

export AbstractConstraint
export EqualityConstraint
export BoundsConstraint
export NonlinearKnotPointConstraint
export NonlinearGlobalConstraint
export NonlinearGlobalKnotPointConstraint
export TimeConsistencyConstraint

export evaluate!
export test_constraint

using NamedTrajectories
using TrajectoryIndexingUtils
using ForwardDiff
using FiniteDiff
using SparseArrays
using TestItemRunner
using LinearAlgebra
using Test

# Import and extend the common interface
using ..CommonInterface
import ..CommonInterface: evaluate!, jacobian_structure, jacobian!, hessian_structure, hessian_of_lagrangian
import ..CommonInterface: eval_jacobian, eval_hessian_of_lagrangian

# ----------------------------------------------------------------------------- #
#                     Abstract Constraints                                      #
# ----------------------------------------------------------------------------- #

abstract type AbstractConstraint end
abstract type AbstractLinearConstraint <: AbstractConstraint end
abstract type AbstractNonlinearConstraint <: AbstractConstraint end

# ----------------------------------------------------------------------------- #
#                     Abstract Constraint Interface                             #
# ----------------------------------------------------------------------------- #

"""
    jacobian_structure(constraint, traj::NamedTrajectory)

Return the sparsity structure of the constraint Jacobian.
"""
function jacobian_structure end

"""
    jacobian!(constraint, traj::NamedTrajectory)

Compute the Jacobian of the constraint in-place.
"""
function jacobian! end

"""
    hessian_structure(constraint, traj::NamedTrajectory)

Return the sparsity structure of the constraint Hessian.
"""
function hessian_structure end

"""
    hessian_of_lagrangian!(constraint, traj::NamedTrajectory, μ::AbstractVector)

Compute the Hessian of the Lagrangian (μ'g) for the constraint in-place.
"""
function hessian_of_lagrangian! end

"""
    get_full_jacobian(constraint, traj::NamedTrajectory)

Assemble the full sparse Jacobian matrix from compact per-timestep blocks.
"""
function get_full_jacobian end

"""
    get_full_hessian(constraint, traj::NamedTrajectory)

Assemble the full sparse Hessian matrix from compact per-timestep blocks.
"""
function get_full_hessian end

# ----------------------------------------------------------------------------- #
#                     Testing Utilities                                         #
# ----------------------------------------------------------------------------- #

"""
    test_constraint(
        constraint::AbstractNonlinearConstraint,
        traj::NamedTrajectory;
        show_jacobian_diff=false,
        show_hessian_diff=false,
        test_equality=true,
        atol=1e-5,
        rtol=1e-5
    )

Test that constraint Jacobian and Hessian match finite difference approximations.

# Arguments
- `constraint`: Constraint to test
- `traj`: Trajectory to evaluate constraint on

# Keyword Arguments
- `show_jacobian_diff=false`: Print detailed Jacobian differences
- `show_hessian_diff=false`: Print detailed Hessian differences
- `test_equality=true`: Test element-wise equality (vs norm-based test)
- `atol=1e-5`: Absolute tolerance
- `rtol=1e-5`: Relative tolerance

# Returns
Tuple of (∂g, ∂g_finite_diff, μ∂²g, μ∂²g_finite_diff) for inspection

# Example
```julia
g(x) = [norm(x) - 1.0]
constraint = NonlinearKnotPointConstraint(g, :x, traj)
test_constraint(constraint, traj)
```
"""
function test_constraint(
    constraint::AbstractNonlinearConstraint,
    traj::NamedTrajectory;
    show_jacobian_diff=false,
    show_hessian_diff=false,
    test_equality=true,
    atol=1e-5,
    rtol=1e-5
)

    # Function to evaluate constraint via evaluate!
    # Use vec(traj) to include both datavec and global_data
    ĝ = Z⃗ -> begin
        # Split into datavec and global_data
        datavec_size = traj.dim * traj.N
        Z_traj = NamedTrajectory(
            traj; 
            datavec=Z⃗[1:datavec_size],
            global_data=Z⃗[datavec_size+1:end]
        )
        values = zeros(eltype(Z⃗), constraint.dim)
        CommonInterface.evaluate!(values, constraint, Z_traj)
        return values
    end

    # Test Jacobian
    ∂g = CommonInterface.eval_jacobian(constraint, traj)
    
    # Compute finite difference Jacobian using full vector (datavec + global_data)
    # Collect to convert from lazy ApplyArray to regular Vector
    ∂g_finite_diff = FiniteDiff.finite_difference_jacobian(ĝ, collect(vec(traj)))

    if show_jacobian_diff 
        println("\tDifference in Jacobian")
        for (i, (a, b)) in enumerate(zip(∂g, ∂g_finite_diff))
            inds = Tuple(CartesianIndices(∂g)[i])
            if !isapprox(a, b; atol=atol, rtol=rtol)
                println((a, b), " @ ", inds)
            end
        end
        println()
    end
    
    # Test Jacobian equality
    if test_equality
        @test all(isapprox.(∂g, ∂g_finite_diff, atol=atol, rtol=rtol))
    else
        if atol > 0.0
            @test norm(∂g - ∂g_finite_diff) < atol
        else
            @test norm(∂g - ∂g_finite_diff) / norm(∂g_finite_diff) < rtol
        end
    end

    # Test Hessian
    μ = rand(constraint.dim)
    
    μ∂²g = CommonInterface.eval_hessian_of_lagrangian(constraint, traj, μ)
    
    # Compute finite difference Hessian using full vector (datavec + global_data)
    # Collect to convert from lazy ApplyArray to regular Vector
    μ∂²g_finite_diff = FiniteDiff.finite_difference_hessian(Z⃗ -> μ'ĝ(Z⃗), collect(vec(traj)))

    if show_hessian_diff 
        println("\tDifference in Hessian")
        # Only show upper triangle since Hessian is symmetric
        for (i, (a, b)) in enumerate(zip(μ∂²g, μ∂²g_finite_diff))
            inds = Tuple(CartesianIndices(μ∂²g)[i])
            if !isapprox(a, b; atol=atol, rtol=rtol) && inds[1] ≤ inds[2]
                println((a, b), " @ ", inds)
            end
        end
        println()
    end
    
    # Test Hessian equality (only upper triangle since Hessian is symmetric)
    if test_equality
        @test all(isapprox.(triu(μ∂²g), triu(μ∂²g_finite_diff), atol=atol))
    else
        if atol > 0.0
            @test norm(μ∂²g - μ∂²g_finite_diff) < atol
        else
            @test norm(μ∂²g - μ∂²g_finite_diff) / norm(μ∂²g_finite_diff) < rtol
        end
    end

    return ∂g, ∂g_finite_diff, μ∂²g, μ∂²g_finite_diff
end

export test_constraint

# Linear constraints
include("linear/equality_constraint.jl")
include("linear/all_equal_constraint.jl")
include("linear/bounds_constraint.jl")
include("linear/total_constraint.jl")
include("linear/symmetry_constraint.jl")
include("linear/time_consistency_constraint.jl")

# Nonlinear constraints
include("nonlinear/knot_point_constraint.jl")
include("nonlinear/global_constraint.jl")
include("nonlinear/global_knot_point_constraint.jl")

end
