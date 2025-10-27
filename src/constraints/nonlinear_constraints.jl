export NonlinearKnotPointConstraint

# ----------------------------------------------------------------------------- #
# NonlinearKnotPointConstraint
# ----------------------------------------------------------------------------- #

struct NonlinearKnotPointConstraint{F1, F2, F3} <: AbstractNonlinearConstraint
    g!::F1
    ∂g!::F2
    ∂gs::Vector{SparseMatrixCSC}
    μ∂²g!::F3
    μ∂²gs::Vector{SparseMatrixCSC}
    equality::Bool
    times::Vector{Int}
    g_dim::Int
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
    - `times::AbstractVector{Int}=1:traj.T`: Time indices at which the constraint is enforced.
    - `jacobian_structure::Union{Nothing, SparseMatrixCSC}=nothing`: Structure of the Jacobian matrix of the constraint.
    - `hessian_structure::Union{Nothing, SparseMatrixCSC}=nothing`: Structure of the Hessian matrix of the constraint.

    # Examples
    ```julia
    # Single variable constraint
    constraint = NonlinearKnotPointConstraint(
        x -> [x[1]^2 + x[2]^2 - 1],
        [:x], traj
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
        equality::Bool=true,
        times::AbstractVector{Int}=1:traj.T,
        jacobian_structure::Union{Nothing, SparseMatrixCSC}=nothing,
        hessian_structure::Union{Nothing, SparseMatrixCSC}=nothing,
    )
        @assert length(params) == length(times) "params must have the same length as times"

        x_comps = vcat([traj.components[name] for name in names]...)
        x_slices = [slice(t, x_comps, traj.dim) for t in times]

        # inspect view of knot point data
        Z⃗ = vec(traj)
        @assert g(Z⃗[x_slices[1]], params[1]) isa AbstractVector{Float64}
        g_dim = length(g(Z⃗[x_slices[1]], params[1]))

        @views function g!(δ::AbstractVector, Z⃗::AbstractVector)
            for (i, x_slice) ∈ enumerate(x_slices)
                δ[slice(i, g_dim)] = g(Z⃗[x_slice], params[i])
            end
        end

        @views function ∂g!(∂gs::Vector{<:AbstractMatrix}, Z⃗::AbstractVector)
            for (i, (x_slice, ∂g)) ∈ enumerate(zip(x_slices, ∂gs))
                # Disjoint
                ForwardDiff.jacobian!(
                    ∂g[:, x_comps], 
                    x -> g(x, params[i]),
                    Z⃗[x_slice]
                )
            end
        end

        @views function μ∂²g!(
            μ∂²gs::Vector{<:AbstractMatrix},   
            Z⃗::AbstractVector, 
            μ::AbstractVector
        )
            for (i, (x_slice, μ∂²g)) ∈ enumerate(zip(x_slices, μ∂²gs))
                # Disjoint
                ForwardDiff.hessian!(
                    μ∂²g[x_comps, x_comps], 
                    x -> μ[slice(i, g_dim)]' * g(x, params[i]), 
                    Z⃗[x_slice]
                )
            end
        end

        if isnothing(jacobian_structure)
            jacobian_structure = spzeros(g_dim, traj.dim) 
            jacobian_structure[:, x_comps] .= 1.0
        else
            @assert size(jacobian_structure) == (g_dim, traj.dim)
        end

        ∂gs = [copy(jacobian_structure) for _ ∈ times]

        if isnothing(hessian_structure)
            hessian_structure = spzeros(traj.dim, traj.dim) 
            hessian_structure[x_comps, x_comps] .= 1.0
        else
            @assert size(hessian_structure) == (traj.dim, traj.dim)
        end

        μ∂²gs = [copy(hessian_structure) for _ ∈ times]

        return new{typeof(g!), typeof(∂g!), typeof(μ∂²g!)}(
            g!,
            ∂g!,
            ∂gs,
            μ∂²g!,
            μ∂²gs,
            equality,
            times,
            g_dim,
            g_dim * length(times)
        )
    end
end

function NonlinearKnotPointConstraint(
    g::Function,
    names::AbstractVector{Symbol},
    traj::NamedTrajectory;
    times::AbstractVector{Int}=1:traj.T,
    kwargs...
)
    # Determine if g expects separate arguments or a single concatenated vector
    # by checking the number of methods and their argument counts
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
        # Multiple variables: try to call g with separate arguments
        # Create wrapper that splits concatenated vector into separate variables
        
        # Get component ranges for each variable
        comp_ranges = Vector{UnitRange{Int}}(undef, num_vars)
        offset = 0
        for (i, name) in enumerate(names)
            comp_len = length(traj.components[name])
            comp_ranges[i] = (offset + 1):(offset + comp_len)
            offset += comp_len
        end
        
        # Try to determine if g expects separate arguments
        # We'll create a wrapper that handles both cases
        params = [nothing for _ in times]
        
        # Test with dummy data to see if g accepts separate arguments
        Z⃗ = vec(traj)
        x_comps = vcat([traj.components[name] for name in names]...)
        x_slice = slice(1, x_comps, traj.dim)
        test_vec = Z⃗[x_slice]
        
        # Split test vector according to component ranges
        test_args = [test_vec[r] for r in comp_ranges]
        
        accepts_separate_args = false
        try
            # Try calling with separate arguments
            result = g(test_args...)
            if result isa AbstractVector
                accepts_separate_args = true
            end
        catch
            # If that fails, it expects a single concatenated vector
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

function get_full_jacobian(
    NLC::NonlinearKnotPointConstraint, 
    traj::NamedTrajectory
)
    Z_dim = traj.dim * traj.T + traj.global_dim
    ∂g_full = spzeros(NLC.dim, Z_dim) 
    for (i, (k, ∂gₖ)) ∈ enumerate(zip(NLC.times, NLC.∂gs))
        # Disjoint
        ∂g_full[slice(i, NLC.g_dim), slice(k, traj.dim)] = ∂gₖ
    end
    return ∂g_full
end

function get_full_hessian(
    NLC::NonlinearKnotPointConstraint, 
    traj::NamedTrajectory
)
    Z_dim = traj.dim * traj.T + traj.global_dim
    μ∂²g_full = spzeros(Z_dim, Z_dim)
    for (k, μ∂²gₖ) ∈ zip(NLC.times, NLC.μ∂²gs)
        # Disjoint
        μ∂²g_full[slice(k, traj.dim), slice(k, traj.dim)] = μ∂²gₖ
    end
    return μ∂²g_full
end



# ============================================================================= #

@testitem "NonlinearKnotPointConstraint - single variable" begin

    using TrajectoryIndexingUtils
    
    include("../../test/test_utils.jl")

    _, traj = bilinear_dynamics_and_trajectory()

    g(a) = [norm(a) - 1.0]

    g_dim = 1
    times = 1:traj.T

    NLC = NonlinearKnotPointConstraint(g, :u, traj; times=times, equality=false)
    U_SLICE(k) = slice(k, traj.components[:u], traj.dim)

    ĝ(Z⃗) = vcat([g(Z⃗[U_SLICE(k)]) for k ∈ times]...)

    δ = zeros(g_dim * traj.T)

    NLC.g!(δ, vec(traj))

    @test δ ≈ ĝ(vec(traj))
    
    NLC.∂g!(NLC.∂gs, vec(traj))

    ∂g_full = Constraints.get_full_jacobian(NLC, traj)

    ∂g_autodiff = ForwardDiff.jacobian(ĝ, vec(traj))

    @test ∂g_full ≈ ∂g_autodiff

    μ = randn(g_dim * traj.T)

    NLC.μ∂²g!(NLC.μ∂²gs, vec(traj), μ)

    hessian_autodiff = ForwardDiff.hessian(Z -> μ'ĝ(Z), vec(traj))

    μ∂²g_full = Constraints.get_full_hessian(NLC, traj) 

    @test μ∂²g_full ≈ hessian_autodiff
end

@testitem "NonlinearKnotPointConstraint - single variable with vector syntax" begin

    using TrajectoryIndexingUtils
    
    include("../../test/test_utils.jl")

    _, traj = bilinear_dynamics_and_trajectory()

    # Test that [:x] syntax works the same as :x
    g(a) = [norm(a) - 1.0]

    g_dim = 1
    times = 1:traj.T

    NLC1 = NonlinearKnotPointConstraint(g, :u, traj; times=times, equality=false)
    NLC2 = NonlinearKnotPointConstraint(g, [:u], traj; times=times, equality=false)
    
    U_SLICE(k) = slice(k, traj.components[:u], traj.dim)
    ĝ(Z⃗) = vcat([g(Z⃗[U_SLICE(k)]) for k ∈ times]...)

    δ1 = zeros(g_dim * traj.T)
    δ2 = zeros(g_dim * traj.T)

    NLC1.g!(δ1, vec(traj))
    NLC2.g!(δ2, vec(traj))

    @test δ1 ≈ δ2
    @test δ1 ≈ ĝ(vec(traj))
end

@testitem "NonlinearKnotPointConstraint - multiple variables concatenated" begin

    using TrajectoryIndexingUtils
    
    include("../../test/test_utils.jl")

    _, traj = bilinear_dynamics_and_trajectory()

    # Constraint function that expects concatenated [x; u]
    g_concat(xu) = [xu[1]^2 + xu[2]^2 - 1.0, xu[3] - 0.5]

    g_dim = 2
    times = 1:traj.T

    NLC = NonlinearKnotPointConstraint(g_concat, [:x, :u], traj; times=times, equality=false)
    
    x_comps = vcat(traj.components[:x], traj.components[:u])
    XU_SLICE(k) = slice(k, x_comps, traj.dim)

    ĝ(Z⃗) = vcat([g_concat(Z⃗[XU_SLICE(k)]) for k ∈ times]...)

    δ = zeros(g_dim * traj.T)
    NLC.g!(δ, vec(traj))

    @test δ ≈ ĝ(vec(traj))
    
    NLC.∂g!(NLC.∂gs, vec(traj))
    ∂g_full = Constraints.get_full_jacobian(NLC, traj)
    ∂g_autodiff = ForwardDiff.jacobian(ĝ, vec(traj))

    @test ∂g_full ≈ ∂g_autodiff

    μ = randn(g_dim * traj.T)
    NLC.μ∂²g!(NLC.μ∂²gs, vec(traj), μ)
    hessian_autodiff = ForwardDiff.hessian(Z -> μ'ĝ(Z), vec(traj))
    μ∂²g_full = Constraints.get_full_hessian(NLC, traj) 

    @test μ∂²g_full ≈ hessian_autodiff
end

@testitem "NonlinearKnotPointConstraint - multiple variables separate arguments" begin

    using TrajectoryIndexingUtils
    
    include("../../test/test_utils.jl")

    _, traj = bilinear_dynamics_and_trajectory()

    # Constraint function with SEPARATE arguments (nicer syntax!)
    g_separate(x, u) = [x[1]^2 + x[2]^2 - 1.0, u[1] - 0.5]

    g_dim = 2
    times = 1:traj.T

    # This should automatically detect and handle separate arguments
    NLC = NonlinearKnotPointConstraint(g_separate, [:x, :u], traj; times=times, equality=false)
    
    x_comps = vcat(traj.components[:x], traj.components[:u])
    XU_SLICE(k) = slice(k, x_comps, traj.dim)
    X_SLICE(k) = slice(k, traj.components[:x], traj.dim)
    U_SLICE(k) = slice(k, traj.components[:u], traj.dim)

    ĝ(Z⃗) = vcat([g_separate(Z⃗[X_SLICE(k)], Z⃗[U_SLICE(k)]) for k ∈ times]...)

    δ = zeros(g_dim * traj.T)
    NLC.g!(δ, vec(traj))

    @test δ ≈ ĝ(vec(traj))
    
    NLC.∂g!(NLC.∂gs, vec(traj))
    ∂g_full = Constraints.get_full_jacobian(NLC, traj)
    ∂g_autodiff = ForwardDiff.jacobian(ĝ, vec(traj))

    @test ∂g_full ≈ ∂g_autodiff

    μ = randn(g_dim * traj.T)
    NLC.μ∂²g!(NLC.μ∂²gs, vec(traj), μ)
    hessian_autodiff = ForwardDiff.hessian(Z -> μ'ĝ(Z), vec(traj))
    μ∂²g_full = Constraints.get_full_hessian(NLC, traj) 

    @test μ∂²g_full ≈ hessian_autodiff
end

@testitem "NonlinearKnotPointConstraint - three variables separate arguments" begin

    using TrajectoryIndexingUtils
    using NamedTrajectories
    
    include("../../test/test_utils.jl")

    # Create trajectory with 3 variables
    T = 10
    x_dim = 2
    u_dim = 1
    a_dim = 1  # Additional variable
    Δt = 0.1
    
    traj = NamedTrajectory(
        (
            x = randn(x_dim, T),
            u = randn(u_dim, T),
            a = randn(a_dim, T),
            Δt = fill(Δt, T),
        );
        controls=(:u, :a),
        timestep=:Δt,
    )

    # Constraint with THREE separate arguments
    g_three(x, u, a) = [x[1] + u[1] + a[1] - 1.0, x[2]^2 - 0.5]

    g_dim = 2
    times = 1:traj.T

    NLC = NonlinearKnotPointConstraint(g_three, [:x, :u, :a], traj; times=times, equality=true)
    
    X_SLICE(k) = slice(k, traj.components[:x], traj.dim)
    U_SLICE(k) = slice(k, traj.components[:u], traj.dim)
    A_SLICE(k) = slice(k, traj.components[:a], traj.dim)

    ĝ(Z⃗) = vcat([g_three(Z⃗[X_SLICE(k)], Z⃗[U_SLICE(k)], Z⃗[A_SLICE(k)]) for k ∈ times]...)

    δ = zeros(g_dim * traj.T)
    NLC.g!(δ, vec(traj))

    @test δ ≈ ĝ(vec(traj))
    
    NLC.∂g!(NLC.∂gs, vec(traj))
    ∂g_full = Constraints.get_full_jacobian(NLC, traj)
    ∂g_autodiff = ForwardDiff.jacobian(ĝ, vec(traj))

    @test ∂g_full ≈ ∂g_autodiff

    μ = randn(g_dim * traj.T)
    NLC.μ∂²g!(NLC.μ∂²gs, vec(traj), μ)
    hessian_autodiff = ForwardDiff.hessian(Z -> μ'ĝ(Z), vec(traj))
    μ∂²g_full = Constraints.get_full_hessian(NLC, traj) 

    @test μ∂²g_full ≈ hessian_autodiff
end

@testitem "NonlinearKnotPointConstraint - inequality vs equality" begin

    using TrajectoryIndexingUtils
    
    include("../../test/test_utils.jl")

    _, traj = bilinear_dynamics_and_trajectory()

    g(x, u) = [x[1] - u[1]]

    # Test inequality constraint
    NLC_ineq = NonlinearKnotPointConstraint(g, [:x, :u], traj; equality=false)
    @test NLC_ineq.equality == false

    # Test equality constraint (default)
    NLC_eq = NonlinearKnotPointConstraint(g, [:x, :u], traj)
    @test NLC_eq.equality == true

    # Both should compute same values, just interpreted differently
    δ_ineq = zeros(NLC_ineq.dim)
    δ_eq = zeros(NLC_eq.dim)
    
    NLC_ineq.g!(δ_ineq, vec(traj))
    NLC_eq.g!(δ_eq, vec(traj))
    
    @test δ_ineq ≈ δ_eq
end

@testitem "NonlinearKnotPointConstraint - subset of times" begin

    using TrajectoryIndexingUtils
    
    include("../../test/test_utils.jl")

    _, traj = bilinear_dynamics_and_trajectory()

    g(x) = [norm(x) - 1.0]
    
    # Only constrain first and last time steps
    times = [1, traj.T]
    
    NLC = NonlinearKnotPointConstraint(g, [:x], traj; times=times, equality=false)
    
    @test NLC.times == times
    @test NLC.dim == length(g(traj.x[:, 1])) * length(times)
    
    X_SLICE(k) = slice(k, traj.components[:x], traj.dim)
    ĝ(Z⃗) = vcat([g(Z⃗[X_SLICE(k)]) for k ∈ times]...)

    δ = zeros(NLC.dim)
    NLC.g!(δ, vec(traj))

    @test δ ≈ ĝ(vec(traj))
end

