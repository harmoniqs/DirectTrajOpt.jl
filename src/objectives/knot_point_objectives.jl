export KnotPointObjective
export TerminalObjective


# ----------------------------------------------------------------------------- #
# KnotPointObjective
# ----------------------------------------------------------------------------- #

"""
    KnotPointObjective(
        ℓ::Function,
        names::AbstractVector{Symbol},
        traj::NamedTrajectory,
        params::AbstractVector;
        kwargs...
    )
    KnotPointObjective(
        ℓ::Function,
        names::AbstractVector{Symbol},
        traj::NamedTrajectory;
        kwargs...
    )
    KnotPointObjective(
        ℓ::Function,
        name::Symbol,
        args...;
        kwargs...
    )

Create a knot point summed objective function for trajectory optimization, where `ℓ(x, p)` 
on trajectory knot point variables `x` with parameters `p`. If the parameters argument is 
omitted, `ℓ(x)` is assumed to be a function of `x` only.

For multiple variables, the function `ℓ` can accept either:
- Separate arguments: `ℓ(x, u)` for variables `[:x, :u]`
- Concatenated vector: `ℓ(xu)` where `xu = [x; u]`

The constructor automatically detects which form `ℓ` expects.

# Arguments
- `ℓ::Function`: Function that defines the objective, `ℓ(x, p)` or `ℓ(x)`.
  - For single variable: `ℓ(x)` where `x` is the variable values at a knot point
  - For multiple variables: `ℓ(x, u)` or `ℓ(xu)` depending on preference
- `names::AbstractVector{Symbol}`: Names of the trajectory variables to be optimized.
- `traj::NamedTrajectory`: The trajectory on which the objective is defined.
- `params::AbstractVector`: Parameters `p` for the objective function ℓ, for each time.

# Keyword Arguments
- `times::AbstractVector{Int}=1:traj.N`: Time indices at which the objective is evaluated.
- `Qs::AbstractVector{Float64}=ones(traj.N)`: Weights for the objective function at each time.

# Examples
```julia
# Single variable objective
obj = KnotPointObjective(
    x -> norm(x)^2,
    [:x], traj
)

# Multiple variables with separate arguments (recommended)
obj = KnotPointObjective(
    (x, u) -> x[1]^2 + u[1]^2,
    [:x, :u], traj
)

# Multiple variables with concatenated vector
obj = KnotPointObjective(
    xu -> xu[1]^2 + xu[3]^2,  # xu = [x[1], x[2], u[1]]
    [:x, :u], traj
)
```
"""
function KnotPointObjective(
    ℓ::Function,
    names::AbstractVector{Symbol},
    traj::NamedTrajectory,
    params::AbstractVector;
    times::AbstractVector{Int}=1:traj.N,
    Qs::Union{Nothing, AbstractVector{Float64}}=nothing,
)
    # Default Qs to ones matching the length of times
    if isnothing(Qs)
        Qs = ones(length(times))
    end
    
    @assert length(Qs) == length(times) "Qs must have the same length as times"
    @assert length(params) == length(times) "params must have the same length as times"

    Z_dim = traj.dim * traj.N + traj.global_dim
    # Compute component indices once for use in closures
    x_comps = vcat([traj.components[name] for name in names]...)

    function L(Z⃗::AbstractVector{<:Real})
        loss = 0.0
        for (i, t) in enumerate(times)
            x_slice = slice(t, x_comps, traj.dim)
            x = Z⃗[x_slice]
            loss += Qs[i] * ℓ(x, params[i])
        end
        return loss
    end

    @views function ∇L(Z⃗::AbstractVector{<:Real})
        ∇ = zeros(Z_dim)
        for (i, t) in enumerate(times)
            x_slice = slice(t, x_comps, traj.dim)
            # Disjoint
            ForwardDiff.gradient!(
                ∇[x_slice], 
                x -> Qs[i] * ℓ(x, params[i]), 
                Z⃗[x_slice]
            )
        end
        return ∇
    end

    function ∂²L_structure()
        structure = spzeros(Z_dim, Z_dim)
        for t in times
            x_slice = slice(t, x_comps, traj.dim)
            structure[x_slice, x_slice] .= 1.0
        end
        structure_pairs = collect(zip(findnz(structure)[1:2]...))
        return structure_pairs
    end

    @views function ∂²L(Z⃗::AbstractVector{<:Real})
        ∂²L_values = zeros(length(∂²L_structure()))
        ∂²ℓ_length = length(x_comps)^2
        for (i, t) in enumerate(times)
            x_slice = slice(t, x_comps, traj.dim)
            # Disjoint
            ForwardDiff.hessian!(
                ∂²L_values[(i - 1) * ∂²ℓ_length + 1:i * ∂²ℓ_length],
                x -> Qs[i] * ℓ(x, params[i]), 
                Z⃗[x_slice]
            )
        end
        return ∂²L_values
    end

    return Objective(L, ∇L, ∂²L, ∂²L_structure)
end

function KnotPointObjective(
    ℓ::Function,
    names::AbstractVector{Symbol},
    traj::NamedTrajectory;
    times::AbstractVector{Int}=1:traj.N,
    kwargs...
)
    # Determine if ℓ expects separate arguments or a single concatenated vector
    num_vars = length(names)
    
    if num_vars == 1
        # Single variable: ℓ(x) where x is the variable values
        params = [nothing for _ in times]
        ℓ_param = (x, _) -> ℓ(x)
        return KnotPointObjective(ℓ_param, names, traj, params; times=times, kwargs...)
    else
        # Multiple variables: try to detect if ℓ expects separate arguments
        
        # Get component ranges for each variable
        comp_ranges = Vector{UnitRange{Int}}(undef, num_vars)
        offset = 0
        for (i, name) in enumerate(names)
            comp_len = length(traj.components[name])
            comp_ranges[i] = (offset + 1):(offset + comp_len)
            offset += comp_len
        end
        
        # Test with dummy data to see if ℓ accepts separate arguments
        Z⃗ = vec(traj)
        x_comps = vcat([traj.components[name] for name in names]...)
        x_slice = slice(1, x_comps, traj.dim)
        test_vec = Z⃗[x_slice]
        
        # Split test vector according to component ranges
        test_args = [test_vec[r] for r in comp_ranges]
        
        accepts_separate_args = false
        try
            # Try calling with separate arguments
            result = ℓ(test_args...)
            if result isa Real
                accepts_separate_args = true
            end
        catch
            # If that fails, it expects a single concatenated vector
            accepts_separate_args = false
        end
        
        params = [nothing for _ in times]
        
        if accepts_separate_args
            # Wrapper that splits concatenated vector into separate arguments
            ℓ_param = function(x_concat, _)
                args = [x_concat[r] for r in comp_ranges]
                return ℓ(args...)
            end
        else
            # ℓ expects a single concatenated vector
            ℓ_param = (x, _) -> ℓ(x)
        end
        
        return KnotPointObjective(ℓ_param, names, traj, params; times=times, kwargs...)
    end
end

function KnotPointObjective(ℓ::Function, name::Symbol, traj::NamedTrajectory; kwargs...)
    return KnotPointObjective(ℓ, [name], traj; kwargs...)
end

function KnotPointObjective(ℓ::Function, name::Symbol, traj::NamedTrajectory, params::AbstractVector; kwargs...)
    return KnotPointObjective(ℓ, [name], traj, params; kwargs...)
end

function TerminalObjective(
    ℓ::Function,
    name::Symbol,
    traj::NamedTrajectory;
    Q::Float64=1.0,
    kwargs...
)
    return KnotPointObjective(
        ℓ,
        name,
        traj;
        Qs=[Q],
        times=[traj.N],
        kwargs...
    )
end


# ============================================================================ #

@testitem "testing KnotPointObjective" begin

    using TrajectoryIndexingUtils
    
    include("../../test/test_utils.jl")

    _, traj = bilinear_dynamics_and_trajectory()

    L(a) = norm(a)

    Qs = [1.0, 2.0]
    times = [1, traj.N]

    OBJ = KnotPointObjective(L, :u, traj, times=times, Qs=Qs)

    L̂(Z⃗) = sum(Q * L(Z⃗[slice(k, traj.components[:u], traj.dim)]) for (Q, k) ∈ zip(Qs, times))

    @test OBJ.L(traj.datavec) ≈ L̂(traj.datavec)
    
    ∂L_autodiff = ForwardDiff.gradient(L̂, traj.datavec)
    @test OBJ.∇L(traj.datavec) ≈ ∂L_autodiff

    ∂²L_autodiff = ForwardDiff.hessian(L̂, traj.datavec)

    ∂²L_full = zeros(size(∂²L_autodiff))
    for (index, entry) in zip(OBJ.∂²L_structure(), OBJ.∂²L(traj.datavec))
        i, j = index
        ∂²L_full[i, j] = entry
    end

    @test ∂²L_full ≈ ∂²L_autodiff
end

@testitem "testing KnotPointObjective with parameters" begin

    using TrajectoryIndexingUtils
    
    include("../../test/test_utils.jl")

    _, traj = bilinear_dynamics_and_trajectory()

    L(x, p) = norm(x) + p

    Qs = [1.0, 2.0]
    times = [1, traj.N]
    params = [1.0, 2.0]

    OBJ = KnotPointObjective(L, :u, traj, params; times=times, Qs=Qs)

    L̂(Z⃗) = sum(Q * L(Z⃗[slice(k, traj.components[:u], traj.dim)], p) for (Q, k, p) ∈ zip(Qs, times, params))

    @test OBJ.L(traj.datavec) ≈ L̂(traj.datavec)
    
    ∂L_autodiff = ForwardDiff.gradient(L̂, traj.datavec)
    @test OBJ.∇L(traj.datavec) ≈ ∂L_autodiff

    ∂²L_autodiff = ForwardDiff.hessian(L̂, traj.datavec)
    ∂²L_full = zeros(size(∂²L_autodiff))
    for (index, entry) in zip(OBJ.∂²L_structure(), OBJ.∂²L(traj.datavec))
        i, j = index
        ∂²L_full[i, j] = entry
    end
    @test ∂²L_full ≈ ∂²L_autodiff
end