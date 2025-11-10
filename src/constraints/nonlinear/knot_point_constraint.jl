export NonlinearKnotPointConstraint

using ..Constraints

# ----------------------------------------------------------------------------- #
# NonlinearKnotPointConstraint
# ----------------------------------------------------------------------------- #

"""
    NonlinearKnotPointConstraint{F} <: AbstractNonlinearConstraint

Constraint applied at individual knot points over a trajectory.

Stores constraint function g, variable names, and pre-allocated storage for Jacobians/Hessians.
Each stored Jacobian is (g_dim × var_dim) for a single knot point, assembled into full structure by get_full_jacobian.

# Fields
- `g::F`: Constraint function mapping (variables..., params) -> constraint values
- `var_names::Vector{Symbol}`: Names of trajectory variables the constraint depends on
- `equality::Bool`: If true, g(x) = 0; if false, g(x) ≤ 0
- `times::Vector{Int}`: Time indices where constraint is applied
- `params::Vector`: Parameters for each time index (e.g., time-varying targets)
- `g_dim::Int`: Dimension of constraint output at each time step
- `var_dim::Int`: Combined dimension of all constrained variables
- `dim::Int`: Total constraint dimension (g_dim * length(times))
- `∂gs::Vector{SparseMatrixCSC{Float64, Int}}`: Pre-allocated Jacobian storage (g_dim × var_dim per timestep)
- `μ∂²gs::Vector{SparseMatrixCSC{Float64, Int}}`: Pre-allocated Hessian storage (var_dim × var_dim per timestep)
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
    ∂gs::Vector{SparseMatrixCSC{Float64, Int}}
    μ∂²gs::Vector{SparseMatrixCSC{Float64, Int}}

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
    - `jacobian_structure::Union{Nothing, SparseMatrixCSC}=nothing`: Optional sparse matrix defining Jacobian sparsity pattern (g_dim × var_dim).
    - `hessian_structure::Union{Nothing, SparseMatrixCSC}=nothing`: Optional sparse matrix defining Hessian sparsity pattern (var_dim × var_dim).

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

    # With custom sparsity structures
    ∂g_structure = sparse([1, 1], [1, 2], [1.0, 1.0], 1, 3)
    μ∂²g_structure = sparse([1, 2, 3], [1, 2, 3], [1.0, 1.0, 1.0], 3, 3)
    constraint = NonlinearKnotPointConstraint(
        g, [:x, :u], traj;
        jacobian_structure=∂g_structure,
        hessian_structure=μ∂²g_structure
    )
    ```
    """
    function NonlinearKnotPointConstraint(
        g::Function,
        names::AbstractVector{Symbol},
        traj::NamedTrajectory,
        params::AbstractVector;
        equality::Bool=true,
        times::AbstractVector{Int}=1:traj.N,
        jacobian_structure::Union{Nothing, SparseMatrixCSC}=nothing,
        hessian_structure::Union{Nothing, SparseMatrixCSC}=nothing,
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

        # Pre-allocate storage using provided structures or default sparse matrices filled with ones
        if !isnothing(jacobian_structure)
            @assert size(jacobian_structure) == (g_dim, var_dim) "jacobian_structure must be (g_dim=$g_dim × var_dim=$var_dim)"
            ∂gs = [copy(jacobian_structure) for _ in times]
        else
            # Default: sparse matrix filled with ones (indicates all entries may be non-zero)
            ∂g_default = sparse(ones(g_dim, var_dim))
            ∂gs = [copy(∂g_default) for _ in times]
        end

        if !isnothing(hessian_structure)
            @assert size(hessian_structure) == (var_dim, var_dim) "hessian_structure must be (var_dim=$var_dim × var_dim=$var_dim)"
            μ∂²gs = [copy(hessian_structure) for _ in times]
        else
            # Default: sparse matrix filled with ones (indicates all entries may be non-zero)
            μ∂²g_default = sparse(ones(var_dim, var_dim))
            μ∂²gs = [copy(μ∂²g_default) for _ in times]
        end

        return new{typeof(g)}(
            g,
            names,
            equality,
            times,
            params,
            g_dim,
            var_dim,
            g_dim * length(times),
            ∂gs,
            μ∂²gs
        )
    end
end

# Convenience constructor without params - creates wrapper that ignores param argument
function NonlinearKnotPointConstraint(
    g::Function,
    names::AbstractVector{Symbol},
    traj::NamedTrajectory;
    times::AbstractVector{Int}=1:traj.N,
    kwargs...
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
            times=times, 
            kwargs...
        )
    else
        # Multiple variables: determine if g expects separate arguments or concatenated
        
        # Get component ranges for splitting concatenated vector
        comp_ranges = Vector{UnitRange{Int}}(undef, num_vars)
        offset = 0
        for (i, name) in enumerate(names)
            comp_len = length(traj.components[name])
            comp_ranges[i] = (offset + 1):(offset + comp_len)
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
            g_param = function(x_concat, _)
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
            times=times, 
            kwargs...
        )
    end
end

function NonlinearKnotPointConstraint(g::Function, name::Symbol, traj::NamedTrajectory; kwargs...)
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
    k::Int
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
    constraint_value(constraint::NonlinearKnotPointConstraint, traj::NamedTrajectory)

Evaluate the constraint at all specified time indices using the trajectory data.
"""
function Constraints.constraint_value(
    constraint::NonlinearKnotPointConstraint,
    traj::NamedTrajectory
)
    δ = zeros(constraint.dim)
    @views for (i, t) ∈ enumerate(constraint.times)
        # Extract the relevant variable values from the knot point
        x_vals = vcat([traj[t][name] for name in constraint.var_names]...)
        δ[slice(i, constraint.g_dim)] = constraint.g(x_vals, constraint.params[i])
    end
    
    return δ
end

"""
    jacobian_structure(constraint::NonlinearKnotPointConstraint, traj::NamedTrajectory)

Return the sparsity structure of a single knot point constraint Jacobian.
"""
function Constraints.jacobian_structure(
    constraint::NonlinearKnotPointConstraint, 
    traj::NamedTrajectory
)
    x_comps = vcat([traj.components[name] for name in constraint.var_names]...)
    
    ∂g = spzeros(constraint.g_dim, traj.dim)
    ∂g[:, x_comps] .= 1.0
    
    return ∂g
end

"""
    jacobian!(constraint::NonlinearKnotPointConstraint, traj::NamedTrajectory)

Compute all Jacobians and store them in constraint.∂gs. Each stored Jacobian is (g_dim × var_dim).
"""
function Constraints.jacobian!(
    constraint::NonlinearKnotPointConstraint,
    traj::NamedTrajectory
)
    @views for (i, t) ∈ enumerate(constraint.times)
        zₖ = traj[t]
        # Extract relevant variables
        x_vals = vcat([zₖ[name] for name in constraint.var_names]...)
        
        # Compute Jacobian directly into sparse storage
        ForwardDiff.jacobian!(
            constraint.∂gs[i],
            x -> constraint.g(x, constraint.params[i]),
            x_vals
        )
    end
    return nothing
end

"""
    hessian_structure(constraint::NonlinearKnotPointConstraint, traj::NamedTrajectory)

Return the sparsity structure of a single knot point constraint Hessian.
"""
function Constraints.hessian_structure(
    constraint::NonlinearKnotPointConstraint,
    traj::NamedTrajectory
)
    x_comps = vcat([traj.components[name] for name in constraint.var_names]...)
    
    μ∂²g = spzeros(traj.dim, traj.dim)
    μ∂²g[x_comps, x_comps] .= 1.0
    
    return μ∂²g
end

"""
    hessian_of_lagrangian!(constraint::NonlinearKnotPointConstraint, traj::NamedTrajectory, μ::AbstractVector)

Compute all Hessians weighted by Lagrange multipliers and store in constraint.μ∂²gs. 
Each stored Hessian is (var_dim × var_dim).
"""
function Constraints.hessian_of_lagrangian!(
    constraint::NonlinearKnotPointConstraint,
    traj::NamedTrajectory,
    μ::AbstractVector
)
    @views for (i, t) ∈ enumerate(constraint.times)
        zₖ = traj[t]
        μₖ = μ[slice(i, constraint.g_dim)]
        
        # Extract relevant variables
        x_vals = vcat([zₖ[name] for name in constraint.var_names]...)
        
        # Compute Hessian directly into sparse storage
        ForwardDiff.hessian!(
            constraint.μ∂²gs[i],
            x -> μₖ' * constraint.g(x, constraint.params[i]),
            x_vals
        )
    end
    return nothing
end

# ----------------------------------------------------------------------------- #
# Full Jacobian and Hessian Assembly
# ----------------------------------------------------------------------------- #

"""
    get_full_jacobian(constraint::NonlinearKnotPointConstraint, traj::NamedTrajectory)

Assemble full sparse Jacobian from stored (g_dim × var_dim) blocks.
"""
function Constraints.get_full_jacobian(
    constraint::NonlinearKnotPointConstraint,
    traj::NamedTrajectory
)
    Z_dim = traj.dim * traj.N + traj.global_dim
    ∂g_full = spzeros(constraint.dim, Z_dim)
    
    # Get variable component indices
    x_comps = vcat([traj.components[name] for name in constraint.var_names]...)
    
    @views for (i, t) ∈ enumerate(constraint.times)
        # Rows: constraint equations for this timestep
        row_range = slice(i, constraint.g_dim)
        # Columns: constrained variables at timestep t
        col_range = slice(t, x_comps, traj.dim)
        ∂g_full[row_range, col_range] = constraint.∂gs[i]
    end
    
    return ∂g_full
end

"""
    get_full_hessian(constraint::NonlinearKnotPointConstraint, traj::NamedTrajectory)

Assemble full sparse Hessian from stored (var_dim × var_dim) blocks.
"""
function Constraints.get_full_hessian(
    constraint::NonlinearKnotPointConstraint, 
    traj::NamedTrajectory
)
    Z_dim = traj.dim * traj.N + traj.global_dim
    μ∂²g_full = spzeros(Z_dim, Z_dim)
    
    # Get variable component indices
    x_comps = vcat([traj.components[name] for name in constraint.var_names]...)
    
    @views for (i, t) ∈ enumerate(constraint.times)
        # Block diagonal structure: (var × var) blocks at each timestep
        block_range = slice(t, x_comps, traj.dim)
        μ∂²g_full[block_range, block_range] = constraint.μ∂²gs[i]
    end
    
    return μ∂²g_full
end



# ============================================================================= #

@testitem "NonlinearKnotPointConstraint - single variable" begin

    using TrajectoryIndexingUtils
    
    include("../../../test/test_utils.jl")

    _, traj = bilinear_dynamics_and_trajectory()

    g(a) = [norm(a) - 1.0]

    g_dim = 1
    times = 1:traj.N

    NLC = NonlinearKnotPointConstraint(g, :u, traj; times=times, equality=false)
    U_SLICE(k) = slice(k, traj.components[:u], traj.dim)

    ĝ(Z⃗) = vcat([g(Z⃗[U_SLICE(k)]) for k ∈ times]...)

    # Test constraint_value
    δ = Constraints.constraint_value(NLC, traj)
    @test δ ≈ ĝ(vec(traj))
    
    # Test jacobian!
    Constraints.jacobian!(NLC, traj)
    ∂g_full = Constraints.get_full_jacobian(NLC, traj)
    ∂g_autodiff = ForwardDiff.jacobian(ĝ, vec(traj))

    @test ∂g_full[:, 1:traj.dim * traj.N] ≈ ∂g_autodiff

    # Test hessian_of_lagrangian
    μ = randn(g_dim * traj.N)
    Constraints.hessian_of_lagrangian!(NLC, traj, μ)
    μ∂²g_full = Constraints.get_full_hessian(NLC, traj)
    hessian_autodiff = ForwardDiff.hessian(Z -> μ'ĝ(Z), vec(traj))

    @test μ∂²g_full[1:traj.dim * traj.N, 1:traj.dim * traj.N] ≈ hessian_autodiff
end

@testitem "NonlinearKnotPointConstraint - single variable with vector syntax" begin

    using TrajectoryIndexingUtils
    
    include("../../../test/test_utils.jl")

    _, traj = bilinear_dynamics_and_trajectory()

    # Test that [:x] syntax works the same as :x
    g(a) = [norm(a) - 1.0]

    g_dim = 1
    times = 1:traj.N

    NLC1 = NonlinearKnotPointConstraint(g, :u, traj; times=times, equality=false)
    NLC2 = NonlinearKnotPointConstraint(g, [:u], traj; times=times, equality=false)
    
    U_SLICE(k) = slice(k, traj.components[:u], traj.dim)
    ĝ(Z⃗) = vcat([g(Z⃗[U_SLICE(k)]) for k ∈ times]...)

    δ1 = Constraints.constraint_value(NLC1, traj)
    δ2 = Constraints.constraint_value(NLC2, traj)

    @test δ1 ≈ δ2
    @test δ1 ≈ ĝ(vec(traj))
end

@testitem "NonlinearKnotPointConstraint - multiple variables concatenated" begin

    using TrajectoryIndexingUtils
    
    include("../../../test/test_utils.jl")

    _, traj = bilinear_dynamics_and_trajectory()

    # Constraint function that expects concatenated [x; u]
    g_concat(xu) = [xu[1]^2 + xu[2]^2 - 1.0, xu[3] - 0.5]

    g_dim = 2
    times = 1:traj.N

    NLC = NonlinearKnotPointConstraint(g_concat, [:x, :u], traj; times=times, equality=false)
    
    x_comps = vcat(traj.components[:x], traj.components[:u])
    XU_SLICE(k) = slice(k, x_comps, traj.dim)

    ĝ(Z⃗) = vcat([g_concat(Z⃗[XU_SLICE(k)]) for k ∈ times]...)

    # Test constraint_value
    δ = Constraints.constraint_value(NLC, traj)
    @test δ ≈ ĝ(vec(traj))
    
    # Test jacobian!
    Constraints.jacobian!(NLC, traj)
    ∂g_full = Constraints.get_full_jacobian(NLC, traj)
    ∂g_autodiff = ForwardDiff.jacobian(ĝ, vec(traj))

    @test ∂g_full[:, 1:traj.dim * traj.N] ≈ ∂g_autodiff

    # Test hessian_of_lagrangian
    μ = randn(g_dim * traj.N)
    Constraints.hessian_of_lagrangian!(NLC, traj, μ)
    μ∂²g_full = Constraints.get_full_hessian(NLC, traj)
    hessian_autodiff = ForwardDiff.hessian(Z -> μ'ĝ(Z), vec(traj))

    @test μ∂²g_full[1:traj.dim * traj.N, 1:traj.dim * traj.N] ≈ hessian_autodiff
end

@testitem "NonlinearKnotPointConstraint - multiple variables separate arguments" begin

    using TrajectoryIndexingUtils
    
    include("../../../test/test_utils.jl")

    _, traj = bilinear_dynamics_and_trajectory()

    # Constraint function with SEPARATE arguments (nicer syntax!)
    g_separate(x, u) = [x[1]^2 + x[2]^2 - 1.0, u[1] - 0.5]

    g_dim = 2
    times = 1:traj.N

    # This should automatically detect and handle separate arguments
    NLC = NonlinearKnotPointConstraint(g_separate, [:x, :u], traj; times=times, equality=false)
    
    x_comps = vcat(traj.components[:x], traj.components[:u])
    XU_SLICE(k) = slice(k, x_comps, traj.dim)
    X_SLICE(k) = slice(k, traj.components[:x], traj.dim)
    U_SLICE(k) = slice(k, traj.components[:u], traj.dim)

    ĝ(Z⃗) = vcat([g_separate(Z⃗[X_SLICE(k)], Z⃗[U_SLICE(k)]) for k ∈ times]...)

    # Test constraint_value
    δ = Constraints.constraint_value(NLC, traj)
    @test δ ≈ ĝ(vec(traj))
    
    # Test jacobian!
    Constraints.jacobian!(NLC, traj)
    ∂g_full = Constraints.get_full_jacobian(NLC, traj)
    ∂g_autodiff = ForwardDiff.jacobian(ĝ, vec(traj))

    @test ∂g_full[:, 1:traj.dim * traj.N] ≈ ∂g_autodiff

    # Test hessian_of_lagrangian
    μ = randn(g_dim * traj.N)
    Constraints.hessian_of_lagrangian!(NLC, traj, μ)
    μ∂²g_full = Constraints.get_full_hessian(NLC, traj)
    hessian_autodiff = ForwardDiff.hessian(Z -> μ'ĝ(Z), vec(traj))

    @test μ∂²g_full[1:traj.dim * traj.N, 1:traj.dim * traj.N] ≈ hessian_autodiff
end

@testitem "NonlinearKnotPointConstraint - three variables separate arguments" begin

    using TrajectoryIndexingUtils
    using NamedTrajectories
    
    include("../../../test/test_utils.jl")

    # Create trajectory with 3 variables
    N = 10
    x_dim = 2
    u_dim = 1
    a_dim = 1  # Additional variable
    Δt = 0.1
    
    traj = NamedTrajectory(
        (
            x = randn(x_dim, N),
            u = randn(u_dim, N),
            a = randn(a_dim, N),
            Δt = fill(Δt, N),
        );
        controls=(:u, :a),
        timestep=:Δt,
    )

    # Constraint with THREE separate arguments
    g_three(x, u, a) = [x[1] + u[1] + a[1] - 1.0, x[2]^2 - 0.5]

    g_dim = 2
    times = 1:traj.N

    NLC = NonlinearKnotPointConstraint(g_three, [:x, :u, :a], traj; times=times, equality=true)
    
    X_SLICE(k) = slice(k, traj.components[:x], traj.dim)
    U_SLICE(k) = slice(k, traj.components[:u], traj.dim)
    A_SLICE(k) = slice(k, traj.components[:a], traj.dim)

    ĝ(Z⃗) = vcat([g_three(Z⃗[X_SLICE(k)], Z⃗[U_SLICE(k)], Z⃗[A_SLICE(k)]) for k ∈ times]...)

    # Test constraint_value
    δ = Constraints.constraint_value(NLC, traj)
    @test δ ≈ ĝ(vec(traj))
    
    # Test jacobian!
    Constraints.jacobian!(NLC, traj)
    ∂g_full = Constraints.get_full_jacobian(NLC, traj)
    ∂g_autodiff = ForwardDiff.jacobian(ĝ, vec(traj))

    @test ∂g_full[:, 1:traj.dim * traj.N] ≈ ∂g_autodiff

    # Test hessian_of_lagrangian
    μ = randn(g_dim * traj.N)
    Constraints.hessian_of_lagrangian!(NLC, traj, μ)
    μ∂²g_full = Constraints.get_full_hessian(NLC, traj)
    hessian_autodiff = ForwardDiff.hessian(Z -> μ'ĝ(Z), vec(traj))

    @test μ∂²g_full[1:traj.dim * traj.N, 1:traj.dim * traj.N] ≈ hessian_autodiff
end

@testitem "NonlinearKnotPointConstraint - inequality vs equality" begin

    using TrajectoryIndexingUtils
    
    include("../../../test/test_utils.jl")

    _, traj = bilinear_dynamics_and_trajectory()

    g(x, u) = [x[1] - u[1]]

    # Test inequality constraint
    NLC_ineq = NonlinearKnotPointConstraint(g, [:x, :u], traj; equality=false)
    @test NLC_ineq.equality == false

    # Test equality constraint (default)
    NLC_eq = NonlinearKnotPointConstraint(g, [:x, :u], traj)
    @test NLC_eq.equality == true

    # Both should compute same values, just interpreted differently
    δ_ineq = Constraints.constraint_value(NLC_ineq, traj)
    δ_eq = Constraints.constraint_value(NLC_eq, traj)
    
    @test δ_ineq ≈ δ_eq
end

@testitem "NonlinearKnotPointConstraint - subset of times" begin

    using TrajectoryIndexingUtils
    
    include("../../../test/test_utils.jl")

    _, traj = bilinear_dynamics_and_trajectory()

    g(x) = [norm(x) - 1.0]
    
    # Only constrain first and last time steps
    times = [1, traj.N]
    
    NLC = NonlinearKnotPointConstraint(g, [:x], traj; times=times, equality=false)
    
    @test NLC.times == times
    @test NLC.dim == length(g(traj.x[:, 1])) * length(times)
    
    X_SLICE(k) = slice(k, traj.components[:x], traj.dim)
    ĝ(Z⃗) = vcat([g(Z⃗[X_SLICE(k)]) for k ∈ times]...)

    δ = Constraints.constraint_value(NLC, traj)

    @test δ ≈ ĝ(vec(traj))
end

@testitem "NonlinearKnotPointConstraint - custom sparsity structures" begin

    using TrajectoryIndexingUtils
    
    include("../../../test/test_utils.jl")

    _, traj = bilinear_dynamics_and_trajectory()

    # Constraint that only depends on first component of u
    g(u) = [u[1]^2 - 1.0]
    
    g_dim = 1
    var_dim = length(traj.components[:u])
    
    # Define sparse Jacobian structure: only first column is non-zero
    ∂g_structure = spzeros(g_dim, var_dim)
    ∂g_structure[1, 1] = 1.0
    
    # Define sparse Hessian structure: only (1,1) entry is non-zero
    μ∂²g_structure = spzeros(var_dim, var_dim)
    μ∂²g_structure[1, 1] = 1.0
    
    NLC = NonlinearKnotPointConstraint(
        g, :u, traj;
        jacobian_structure=∂g_structure,
        hessian_structure=μ∂²g_structure,
        equality=false
    )
    
    # Check that storage was initialized with the structure
    @test size(NLC.∂gs[1]) == (g_dim, var_dim)
    @test size(NLC.μ∂²gs[1]) == (var_dim, var_dim)
    
    # Test that computation still works correctly
    U_SLICE(k) = slice(k, traj.components[:u], traj.dim)
    ĝ(Z⃗) = vcat([g(Z⃗[U_SLICE(k)]) for k ∈ 1:traj.N]...)
    
    δ = Constraints.constraint_value(NLC, traj)
    @test δ ≈ ĝ(vec(traj))
    
    # Test jacobian! with custom structure
    Constraints.jacobian!(NLC, traj)
    ∂g_full = Constraints.get_full_jacobian(NLC, traj)
    ∂g_autodiff = ForwardDiff.jacobian(ĝ, vec(traj))
    @test ∂g_full[:, 1:traj.dim * traj.N] ≈ ∂g_autodiff
end
