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
omitted, `ℓ(x)` is assumed  to be a function of `x` only.

# Arguments
- `ℓ::Function`: Function that defines the objective, ℓ(x, p) or ℓ(x).
- `names::AbstractVector{Symbol}`: Names of the trajectory variables to be optimized.
- `traj::NamedTrajectory`: The trajectory on which the objective is defined.
- `params::AbstractVector`: Parameters `p` for the objective function ℓ, for each time.

# Keyword Arguments
- `times::AbstractVector{Int}=1:traj.T`: Time indices at which the objective is evaluated.
- `Qs::AbstractVector{Float64}=ones(traj.T)`: Weights for the objective function at each time.
"""
function KnotPointObjective(
    ℓ::Function,
    names::AbstractVector{Symbol},
    traj::NamedTrajectory,
    params::AbstractVector;
    times::AbstractVector{Int}=1:traj.T,
    Qs::AbstractVector{Float64}=ones(traj.T),
)
    @assert length(Qs) == length(times) "Qs must have the same length as times"
    @assert length(params) == length(times) "params must have the same length as times"

    Z_dim = traj.dim * traj.T + traj.global_dim
    x_comps = vcat([traj.components[name] for name in names]...)
    x_slices = [slice(t, x_comps, traj.dim) for t in times]

    function L(Z⃗::AbstractVector{<:Real})
        loss = 0.0
        for (i, x_slice) in enumerate(x_slices)
            x = Z⃗[x_slice]
            loss += Qs[i] * ℓ(x, params[i])
        end
        return loss
    end

    @views function ∇L(Z⃗::AbstractVector{<:Real})
        ∇ = zeros(Z_dim)
        for (i, x_slice) in enumerate(x_slices)
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
        for x_slice in x_slices
            structure[x_slice, x_slice] .= 1.0
        end
        structure_pairs = collect(zip(findnz(structure)[1:2]...))
        return structure_pairs
    end

    @views function ∂²L(Z⃗::AbstractVector{<:Real})
        ∂²L_values = zeros(length(∂²L_structure()))
        ∂²ℓ_length = length(x_comps)^2
        for (i, x_slice) in enumerate(x_slices)
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
    times::AbstractVector{Int}=1:traj.T,
    kwargs...
)
    params = [nothing for _ in times]
    ℓ_param = (x, _) -> ℓ(x)
    return KnotPointObjective(ℓ_param, names, traj, params; times=times, kwargs...)
end

function KnotPointObjective(ℓ::Function,  name::Symbol,  args...;  kwargs...)
    return KnotPointObjective(ℓ, [name], args...; kwargs...)
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
        times=[traj.T],
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
    times = [1, traj.T]

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
    times = [1, traj.T]
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