export NonlinearKnotPointConstraint

using ..Constraints

# ----------------------------------------------------------------------------- #
# NonlinearKnotPointConstraint
# ----------------------------------------------------------------------------- #

"""
    NonlinearKnotPointConstraint{F} <: AbstractNonlinearConstraint

Constraint applied at individual knot points over a trajectory.

Computes Jacobians and Hessians on-the-fly using automatic differentiation.
For pre-allocated optimization, see Piccolissimo.OptimizedNonlinearKnotPointConstraint.

# Fields
- `g::F`: Constraint function mapping (variables..., params) -> constraint values
- `var_names::Vector{Symbol}`: Names of trajectory variables the constraint depends on
- `equality::Bool`: If true, g(x) = 0; if false, g(x) ≤ 0
- `times::Vector{Int}`: Time indices where constraint is applied
- `params::Vector`: Parameters for each time index (e.g., time-varying targets)
- `g_dim::Int`: Dimension of constraint output at each time step
- `var_dim::Int`: Combined dimension of all constrained variables
- `dim::Int`: Total constraint dimension (g_dim * length(times))
"""
struct NonlinearKnotPointConstraint{F} <: AbstractNonlinearConstraint
    g::F
    var_names::Vector{Symbol}
    equality::Bool
    times::Vector{Int}
    params::Vector
    g_dim::Int
    var_dim::Int
    dim::Int

    """
        NonlinearKnotPointConstraint(
            g::Function,
            names::Union{Symbol, AbstractVector{Symbol}},
            traj::NamedTrajectory;
            kwargs...
        )

    Create a NonlinearKnotPointConstraint object that represents a nonlinear constraint on a trajectory.

    # Arguments
    - `g::Function`: Function over knot point variable(s) that defines the constraint. 
      - For single variable: `g(x)` where `x` is the variable values at a knot point
      - For multiple variables: `g(x, u)` where each argument corresponds to a variable in `names`
    - `names::Union{Symbol, AbstractVector{Symbol}}`: Name(s) of the variable(s) to be constrained.
      - Single variable: `:x`
      - Multiple variables: `[:x, :u]`
    - `traj::NamedTrajectory`: The trajectory on which the constraint is defined.

    # Keyword Arguments
    - `equality::Bool=true`: If `true`, the constraint is `g(x) = 0`. Otherwise, the constraint is `g(x) ≤ 0`.
    - `times::AbstractVector{Int}=1:traj.N`: Time indices at which the constraint is enforced.
    - `params::AbstractVector=fill(nothing, length(times))`: Parameters for each time step (e.g., time-varying targets).

    # Examples
    ```julia
    # Single variable constraint
    constraint = NonlinearKnotPointConstraint(
        x -> [x[1]^2 + x[2]^2 - 1],
        :x, traj
    )

    # Multiple variable constraint
    constraint = NonlinearKnotPointConstraint(
        (x, u) -> [x[1] - u[1]^2],
        [:x, :u], traj
    )
    ```
    """
    function NonlinearKnotPointConstraint(
        g::Function,
        names::AbstractVector{Symbol},
        traj::NamedTrajectory,
        params::AbstractVector;
        equality::Bool = true,
        times::AbstractVector{Int} = 1:traj.N,
    )
        @assert length(params) == length(times) "params must have the same length as times"

        # Get component indices for all variables
        x_comps = vcat([traj.components[name] for name in names]...)
        var_dim = length(x_comps)

        # Determine constraint dimension by evaluating with first parameter
        Z⃗ = vec(traj)
        x_slice_test = slice(1, x_comps, traj.dim)
        @assert g(Z⃗[x_slice_test], params[1]) isa AbstractVector{<:Real}
        g_dim = length(g(Z⃗[x_slice_test], params[1]))

        return new{typeof(g)}(
            g,
            names,
            equality,
            times,
            params,
            g_dim,
            var_dim,
            g_dim * length(times),
        )
    end
end

# Convenience constructor without params - creates wrapper that ignores param argument
function NonlinearKnotPointConstraint(
    g::Function,
    names::AbstractVector{Symbol},
    traj::NamedTrajectory;
    times::AbstractVector{Int} = 1:traj.N,
    kwargs...,
)
    num_vars = length(names)

    if num_vars == 1
        # Single variable: g(x) where x is the variable values
        params = [nothing for _ in times]
        g_param = (x, _) -> g(x)
        return NonlinearKnotPointConstraint(
            g_param,
            names,
            traj,
            params;
            times = times,
            kwargs...,
        )
    else
        # Multiple variables: determine if g expects separate arguments or concatenated

        # Get component ranges for splitting concatenated vector
        comp_ranges = Vector{UnitRange{Int}}(undef, num_vars)
        offset = 0
        for (i, name) in enumerate(names)
            comp_len = length(traj.components[name])
            comp_ranges[i] = (offset+1):(offset+comp_len)
            offset += comp_len
        end

        params = [nothing for _ in times]

        # Test if g accepts separate arguments
        Z⃗ = vec(traj)
        x_comps = vcat([traj.components[name] for name in names]...)
        x_slice = slice(1, x_comps, traj.dim)
        test_vec = Z⃗[x_slice]
        test_args = [test_vec[r] for r in comp_ranges]

        accepts_separate_args = false
        try
            result = g(test_args...)
            accepts_separate_args = result isa AbstractVector
        catch
            accepts_separate_args = false
        end

        if accepts_separate_args
            # Wrapper that splits concatenated vector into separate arguments
            g_param = function (x_concat, _)
                args = [x_concat[r] for r in comp_ranges]
                return g(args...)
            end
        else
            # g expects a single concatenated vector
            g_param = (x, _) -> g(x)
        end

        return NonlinearKnotPointConstraint(
            g_param,
            names,
            traj,
            params;
            times = times,
            kwargs...,
        )
    end
end

function NonlinearKnotPointConstraint(
    g::Function,
    name::Symbol,
    traj::NamedTrajectory;
    kwargs...,
)
    return NonlinearKnotPointConstraint(g, [name], traj; kwargs...)
end

# ----------------------------------------------------------------------------- #
# Method Implementations for NonlinearKnotPointConstraint
# ----------------------------------------------------------------------------- #

"""
    (constraint::NonlinearKnotPointConstraint)(δ, zₖ::KnotPoint, k::Int)

Evaluate the constraint at a single knot point.
"""
function (constraint::NonlinearKnotPointConstraint)(
    δ::AbstractVector,
    zₖ::KnotPoint,
    k::Int,
)
    # Extract the relevant variable values from the knot point
    x_vals = vcat([zₖ[name] for name in constraint.var_names]...)

    # Find which index this timestep corresponds to in constraint.times
    time_idx = findfirst(==(k), constraint.times)
    if isnothing(time_idx)
        error("Timestep $k not in constraint times $(constraint.times)")
    end

    δ[:] = constraint.g(x_vals, constraint.params[time_idx])
    return nothing
end

"""
    evaluate!(values::AbstractVector, constraint::NonlinearKnotPointConstraint, traj::NamedTrajectory)

Evaluate the constraint at all specified time indices, storing results in-place in `values`.
This is part of the common interface with integrators.
"""
function CommonInterface.evaluate!(
    values::AbstractVector,
    constraint::NonlinearKnotPointConstraint,
    traj::NamedTrajectory,
)
    @views for (i, t) ∈ enumerate(constraint.times)
        # Extract the relevant variable values from the knot point
        x_vals = vcat([traj[t][name] for name in constraint.var_names]...)
        values[slice(i, constraint.g_dim)] = constraint.g(x_vals, constraint.params[i])
    end

    return nothing
end

"""
    eval_jacobian(constraint::NonlinearKnotPointConstraint, traj::NamedTrajectory)

Compute and return the full Jacobian using automatic differentiation.
"""
@views function CommonInterface.eval_jacobian(
    K::NonlinearKnotPointConstraint,
    traj::NamedTrajectory,
)
    ∂K = spzeros(K.dim, traj.dim * traj.N + traj.global_dim)
    x_comps = vcat([traj.components[name] for name ∈ K.var_names]...)
    for (i, k) ∈ enumerate(K.times)
        ForwardDiff.jacobian!(
            ∂K[slice(i, K.g_dim), slice(k, x_comps, traj.dim)],
            x -> K.g(x, K.params[i]),
            vcat([traj[k][name] for name in K.var_names]...),
        )
    end
    return ∂K
end

"""
    eval_hessian_of_lagrangian(constraint::NonlinearKnotPointConstraint, traj::NamedTrajectory, μ::AbstractVector)

Compute and return the full Hessian of the Lagrangian using automatic differentiation.
"""
@views function CommonInterface.eval_hessian_of_lagrangian(
    K::NonlinearKnotPointConstraint,
    traj::NamedTrajectory,
    μ::AbstractVector,
)
    μ∂²K = spzeros(traj.dim * traj.N + traj.global_dim, traj.dim * traj.N + traj.global_dim)
    x_comps = vcat([traj.components[name] for name ∈ K.var_names]...)

    for (i, k) ∈ enumerate(K.times)
        μₖ = μ[slice(i, K.g_dim)]
        block_range = slice(k, x_comps, traj.dim)
        ForwardDiff.hessian!(
            μ∂²K[block_range, block_range],
            x -> μₖ' * K.g(x, K.params[i]),
            vcat([traj[k][name] for name in K.var_names]...),
        )
    end

    return μ∂²K
end



# ============================================================================= #

@testitem "NonlinearKnotPointConstraint - single variable" begin
    include("../../../test/test_utils.jl")

    _, traj = bilinear_dynamics_and_trajectory()

    g(a) = [norm(a) - 1.0]

    NLC = NonlinearKnotPointConstraint(g, :u, traj; times = 1:traj.N, equality = false)

    # Test Jacobian and Hessian against finite differences
    test_constraint(NLC, traj; atol = 1e-3, show_jacobian_diff = true)
end

@testitem "NonlinearKnotPointConstraint - single variable with vector syntax" begin
    using DirectTrajOpt: CommonInterface

    include("../../../test/test_utils.jl")

    _, traj = bilinear_dynamics_and_trajectory()

    # Test that [:u] syntax works the same as :u
    g(a) = [norm(a) - 1.0]

    NLC1 = NonlinearKnotPointConstraint(g, :u, traj; equality = false)
    NLC2 = NonlinearKnotPointConstraint(g, [:u], traj; equality = false)

    δ1 = zeros(NLC1.dim)
    δ2 = zeros(NLC2.dim)
    CommonInterface.evaluate!(δ1, NLC1, traj)
    CommonInterface.evaluate!(δ2, NLC2, traj)

    @test δ1 ≈ δ2

    # Test both with finite differences
    test_constraint(NLC1, traj; atol = 1e-3)
    test_constraint(NLC2, traj; atol = 1e-3)
end

@testitem "NonlinearKnotPointConstraint - multiple variables concatenated" begin
    include("../../../test/test_utils.jl")

    _, traj = bilinear_dynamics_and_trajectory()

    # Constraint function that expects concatenated [x; u]
    g_concat(xu) = [xu[1]^2 + xu[2]^2 - 1.0, xu[3] - 0.5]

    NLC = NonlinearKnotPointConstraint(g_concat, [:x, :u], traj; equality = false)

    # Test Jacobian and Hessian against finite differences
    test_constraint(NLC, traj)
end

@testitem "NonlinearKnotPointConstraint - multiple variables separate arguments" begin
    include("../../../test/test_utils.jl")

    _, traj = bilinear_dynamics_and_trajectory()

    # Constraint function with SEPARATE arguments (nicer syntax!)
    g_separate(x, u) = [x[1]^2 + x[2]^2 - 1.0, u[1] - 0.5]

    # This should automatically detect and handle separate arguments
    NLC = NonlinearKnotPointConstraint(g_separate, [:x, :u], traj; equality = false)

    # Test Jacobian and Hessian against finite differences
    test_constraint(NLC, traj)
end

@testitem "NonlinearKnotPointConstraint - three variables separate arguments" begin
    using NamedTrajectories

    include("../../../test/test_utils.jl")

    # Create trajectory with 3 variables
    N = 10
    x_dim = 2
    u_dim = 1
    a_dim = 1  # Additional variable
    Δt = 0.1

    traj = NamedTrajectory(
        (x = randn(x_dim, N), u = randn(u_dim, N), a = randn(a_dim, N), Δt = fill(Δt, N));
        controls = (:u, :a),
        timestep = :Δt,
    )

    # Constraint with THREE separate arguments
    g_three(x, u, a) = [x[1] + u[1] + a[1] - 1.0, x[2]^2 - 0.5]

    NLC = NonlinearKnotPointConstraint(g_three, [:x, :u, :a], traj; equality = true)

    # Test Jacobian and Hessian against finite differences
    test_constraint(NLC, traj)
end

@testitem "NonlinearKnotPointConstraint - inequality vs equality" begin
    using DirectTrajOpt: CommonInterface

    include("../../../test/test_utils.jl")

    _, traj = bilinear_dynamics_and_trajectory()

    g(x, u) = [x[1] - u[1]]

    # Test inequality constraint
    NLC_ineq = NonlinearKnotPointConstraint(g, [:x, :u], traj; equality = false)
    @test NLC_ineq.equality == false

    # Test equality constraint (default)
    NLC_eq = NonlinearKnotPointConstraint(g, [:x, :u], traj)
    @test NLC_eq.equality == true

    # Both should compute same values, just interpreted differently
    δ_ineq = zeros(NLC_ineq.dim)
    δ_eq = zeros(NLC_eq.dim)
    CommonInterface.evaluate!(δ_ineq, NLC_ineq, traj)
    CommonInterface.evaluate!(δ_eq, NLC_eq, traj)

    @test δ_ineq ≈ δ_eq

    # Test both with finite differences
    test_constraint(NLC_ineq, traj)
    test_constraint(NLC_eq, traj)
end

@testitem "NonlinearKnotPointConstraint - subset of times" begin
    include("../../../test/test_utils.jl")

    _, traj = bilinear_dynamics_and_trajectory()

    g(x) = [norm(x) - 1.0]

    # Only constrain first and last time steps
    times = [1, traj.N]

    NLC = NonlinearKnotPointConstraint(g, [:x], traj; times = times, equality = false)

    @test NLC.times == times
    @test NLC.dim == length(g(traj.x[:, 1])) * length(times)

    # Test Jacobian and Hessian against finite differences
    test_constraint(NLC, traj)
end
