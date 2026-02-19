module Objectives

export AbstractObjective
export objective_value
export gradient!
export hessian_structure
export get_full_hessian
export test_objective
export NullObjective
export CompositeObjective
export QuadraticRegularizer
export LinearRegularizer
export MinimumTimeObjective
export KnotPointObjective
export GlobalObjective
export GlobalKnotPointObjective

using ..Constraints

using TrajectoryIndexingUtils
using NamedTrajectories
using LinearAlgebra
using SparseArrays
using ForwardDiff
using FiniteDiff
using TestItemRunner
using Test

# ----------------------------------------------------------------------------- #
#                     Abstract Objective Interface                              #
# ----------------------------------------------------------------------------- #

"""
    AbstractObjective

Abstract type for all objective functions in trajectory optimization.

Concrete objective types must implement:
- `objective_value(obj, traj)`: Evaluate the objective at trajectory
- `gradient!(∇, obj, traj)`: Compute gradient in-place (gradient is always dense)
- `hessian_structure(obj, traj)`: Return sparsity structure as sparse matrix
- `get_full_hessian(obj, traj)`: Return the full Hessian matrix

Objectives support addition and scalar multiplication through `CompositeObjective`.

Note: Unlike constraints and integrators, objective gradients are always dense, so no
gradient_structure method is needed. The gradient! method fills the entire ∇ vector.
"""
abstract type AbstractObjective end

# ----------------------------------------------------------------------------- #
#                     Objective Interface Methods                               #
# ----------------------------------------------------------------------------- #

"""
    objective_value(obj::AbstractObjective, traj::NamedTrajectory)

Evaluate the objective function at the given trajectory.
"""
function objective_value end

(obj::AbstractObjective)(traj::NamedTrajectory) = objective_value(obj, traj)

"""
    gradient!(∇::AbstractVector, obj::AbstractObjective, traj::NamedTrajectory)

Compute the gradient of the objective in-place. The gradient is always dense.
"""
function gradient! end

"""
    hessian_structure(obj::AbstractObjective, traj::NamedTrajectory)

Return the sparsity structure of the Hessian as a sparse matrix with non-zero
entries where the Hessian has non-zero values.
"""
function hessian_structure end

"""
    get_full_hessian(obj::AbstractObjective, traj::NamedTrajectory)

Compute and return the full Hessian matrix of the objective.
"""
function get_full_hessian end




# ----------------------------------------------------------------------------- #
#                        Composite Objectives                                   #
# ----------------------------------------------------------------------------- #

"""
    CompositeObjective <: AbstractObjective

Represents a weighted sum or composition of multiple objectives.

# Fields
- `objectives::Vector{AbstractObjective}`: Individual objectives to combine
- `weights::Vector{Float64}`: Weight for each objective
"""
struct CompositeObjective <: AbstractObjective
    objectives::Vector{AbstractObjective}
    weights::Vector{Float64}
end

function objective_value(obj::CompositeObjective, traj::NamedTrajectory)
    val = 0.0
    for (sub_obj, weight) in zip(obj.objectives, obj.weights)
        val += weight * objective_value(sub_obj, traj)
    end
    return val
end

function gradient!(∇::AbstractVector, obj::CompositeObjective, traj::NamedTrajectory)
    fill!(∇, 0.0)
    ∇_temp = similar(∇)
    for (sub_obj, weight) in zip(obj.objectives, obj.weights)
        fill!(∇_temp, 0.0)
        gradient!(∇_temp, sub_obj, traj)
        ∇ .+= weight .* ∇_temp
    end
    return nothing
end

function hessian_structure(obj::CompositeObjective, traj::NamedTrajectory)
    Z_dim = traj.dim * traj.N + traj.global_dim
    structure = spzeros(Z_dim, Z_dim)
    for sub_obj in obj.objectives
        structure .+= hessian_structure(sub_obj, traj)
    end
    return structure
end

function get_full_hessian(obj::CompositeObjective, traj::NamedTrajectory)
    Z_dim = traj.dim * traj.N + traj.global_dim
    ∂²L = spzeros(Z_dim, Z_dim)
    for (sub_obj, weight) in zip(obj.objectives, obj.weights)
        ∂²L .+= weight * get_full_hessian(sub_obj, traj)
    end
    return ∂²L
end

# ----------------------------------------------------------------------------- #
#                        Operators for Objectives                              #
# ----------------------------------------------------------------------------- #

"""
Add two objectives together. Flattens nested CompositeObjectives.
"""
function Base.:+(obj1::AbstractObjective, obj2::AbstractObjective)
    # Flatten composite objectives
    objs1 = obj1 isa CompositeObjective ? obj1.objectives : [obj1]
    weights1 = obj1 isa CompositeObjective ? obj1.weights : [1.0]

    objs2 = obj2 isa CompositeObjective ? obj2.objectives : [obj2]
    weights2 = obj2 isa CompositeObjective ? obj2.weights : [1.0]

    return CompositeObjective(vcat(objs1, objs2), vcat(weights1, weights2))
end

"""
Scale an objective by a constant.
"""
function Base.:*(num::Real, obj::AbstractObjective)
    if obj isa CompositeObjective
        return CompositeObjective(obj.objectives, num .* obj.weights)
    else
        return CompositeObjective([obj], [Float64(num)])
    end
end

Base.:*(obj::AbstractObjective, num::Real) = num * obj

function Base.show(io::IO, obj::CompositeObjective)
    n = length(obj.objectives)
    print(io, "CompositeObjective ($n terms)")
    for (sub_obj, w) in zip(obj.objectives, obj.weights)
        w_str = string(round(w, sigdigits = 4))
        print(io, "\n  $(lpad(w_str, 8)) * ")
        show(io, sub_obj)
    end
end

# ----------------------------------------------------------------------------- #
# Null objective                                      
# ----------------------------------------------------------------------------- #

"""
    NullObjective <: AbstractObjective

A zero objective that contributes nothing to the cost.
Useful as a placeholder or when only constraints matter.
"""
struct NullObjective <: AbstractObjective end

NullObjective(::NamedTrajectory) = NullObjective()

Base.show(io::IO, ::NullObjective) = print(io, "NullObjective")

objective_value(::NullObjective, ::NamedTrajectory) = 0.0

function gradient!(∇::AbstractVector, ::NullObjective, ::NamedTrajectory)
    fill!(∇, 0.0)
    return nothing
end

function hessian_structure(::NullObjective, traj::NamedTrajectory)
    Z_dim = traj.dim * traj.N + traj.global_dim
    return spzeros(Z_dim, Z_dim)
end

function get_full_hessian(::NullObjective, traj::NamedTrajectory)
    Z_dim = traj.dim * traj.N + traj.global_dim
    return spzeros(Z_dim, Z_dim)
end# ----------------------------------------------------------------------------- #
#                        Test Objective Utility                                #
# ----------------------------------------------------------------------------- #

"""
    test_objective(
        obj::AbstractObjective,
        traj::NamedTrajectory;
        show_gradient_diff=false,
        show_hessian_diff=false,
        test_equality=true,
        atol=1e-5,
        rtol=1e-5
    )

Test an objective's gradient and Hessian implementations against finite differences.

Similar to `test_integrator`, this validates that computed derivatives match
finite differences.

# Arguments
- `obj::AbstractObjective`: The objective to test
- `traj::NamedTrajectory`: Trajectory defining the problem structure

# Keyword Arguments
- `show_gradient_diff`: Print element-wise differences in gradient
- `show_hessian_diff`: Print element-wise differences in Hessian
- `test_equality`: Test element-wise equality (vs. norm-based)
- `atol`: Absolute tolerance for comparisons
- `rtol`: Relative tolerance for comparisons
"""
function test_objective(
    obj::AbstractObjective,
    traj::NamedTrajectory;
    show_gradient_diff = false,
    show_hessian_diff = false,
    test_equality = true,
    atol = 1e-5,
    rtol = 1e-5,
)
    # Collect to avoid LazyArrays issues
    Z⃗_vec = collect(vec(traj))
    Z_dim = length(Z⃗_vec)

    # Test objective value
    @test objective_value(obj, traj) isa Real

    # Test gradient
    ∇ = zeros(Z_dim)
    gradient!(∇, obj, traj)
    ∇_fd = FiniteDiff.finite_difference_gradient(Z⃗_vec) do Z⃗
        traj_data = Z⃗[1:(traj.dim*traj.N)]
        global_data = Z⃗[(traj.dim*traj.N+1):end]
        traj_wrapped =
            NamedTrajectory(traj; datavec = traj_data, global_data = global_data)
        return objective_value(obj, traj_wrapped)
    end

    if show_gradient_diff
        println("\tDifference in gradient")
        for i = 1:Z_dim
            if abs(∇[i] - ∇_fd[i]) > atol ||
               abs((∇[i] - ∇_fd[i]) / (∇_fd[i] + 1e-10)) > rtol
                println("\t  [$i]: $(∇[i]) vs $(∇_fd[i]) (diff: $(∇[i] - ∇_fd[i]))")
            end
        end
        println()
    else
        if test_equality
            @test all(isapprox.(∇, ∇_fd, atol = atol, rtol = rtol))
        else
            if atol > 0.0
                @test norm(∇ - ∇_fd) < atol
            else
                @test norm(∇ - ∇_fd) / norm(∇_fd) < rtol
            end
        end
    end

    # Test Hessian
    ∂²J = get_full_hessian(obj, traj)

    ∂²J_fd = FiniteDiff.finite_difference_hessian(Z⃗_vec) do Z⃗
        traj_data = Z⃗[1:(traj.dim*traj.N)]
        global_data = Z⃗[(traj.dim*traj.N+1):end]
        traj_wrapped =
            NamedTrajectory(traj; datavec = traj_data, global_data = global_data)
        return objective_value(obj, traj_wrapped)
    end

    if show_hessian_diff
        println("\tDifference in Hessian")
        for i = 1:Z_dim, j = 1:Z_dim
            if abs(∂²J[i, j] - ∂²J_fd[i, j]) > atol ||
               abs((∂²J[i, j] - ∂²J_fd[i, j]) / (∂²J_fd[i, j] + 1e-10)) > rtol
                println(
                    "\t  [$i, $j]: $(∂²J[i, j]) vs $(∂²J_fd[i, j]) (diff: $(∂²J[i, j] - ∂²J_fd[i, j]))",
                )
            end
        end
        println()
    else
        @test triu(∂²J) ≈ triu(∂²J_fd) atol=atol rtol=rtol
    end

    return nothing
end

# ----------------------------------------------------------------------------- #
# Additional objectives
# ----------------------------------------------------------------------------- #

include("knot_point_objectives.jl")
include("global_objectives.jl")
include("minimum_time_objective.jl")
include("regularizers.jl")

# ----------------------------------------------------------------------------- #
# Tests
# ----------------------------------------------------------------------------- #

@testitem "testing CompositeObjective" begin
    include("../../test/test_utils.jl")
    using DirectTrajOpt.Objectives

    _, traj = bilinear_dynamics_and_trajectory()

    # Create some simple objectives
    obj1 = QuadraticRegularizer(:u, traj, 1.0)
    obj2 = QuadraticRegularizer(:du, traj, 0.5)
    obj3 = MinimumTimeObjective(traj, D = 2.0)

    # Test addition
    obj_sum = obj1 + obj2
    @test obj_sum isa CompositeObjective
    @test length(obj_sum.objectives) == 2
    @test obj_sum.weights == [1.0, 1.0]

    # Test scalar multiplication
    obj_scaled = 2.0 * obj1
    @test obj_scaled isa CompositeObjective
    @test obj_scaled.weights == [2.0]

    # Test combination
    obj_combo = 2.0 * obj1 + 0.5 * obj2 + obj3
    @test obj_combo isa CompositeObjective
    @test length(obj_combo.objectives) == 3

    # Test that composite objective works correctly
    test_objective(obj_sum, traj)
    test_objective(obj_scaled, traj)
    test_objective(obj_combo, traj)

    # Test that values match
    val1 = objective_value(obj1, traj)
    val2 = objective_value(obj2, traj)
    val_sum = objective_value(obj_sum, traj)
    @test val_sum ≈ val1 + val2

    val_scaled = objective_value(obj_scaled, traj)
    @test val_scaled ≈ 2.0 * val1

    val3 = objective_value(obj3, traj)
    val_combo = objective_value(obj_combo, traj)
    @test val_combo ≈ 2.0 * val1 + 0.5 * val2 + val3
end

end
