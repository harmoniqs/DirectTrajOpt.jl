export NonlinearGlobalConstraint
export NonlinearGlobalKnotPointConstraint


# ----------------------------------------------------------------------------- #
# NonlinearGlobalConstraint
# ----------------------------------------------------------------------------- #

struct NonlinearGlobalConstraint <: AbstractNonlinearConstraint
    g!::Function
    ∂g!::Function
    ∂gs::SparseMatrixCSC
    μ∂²g!::Function
    μ∂²gs::SparseMatrixCSC
    equality::Bool
    dim::Int

    """
        Create a NonlinearGlobalConstraint object with global components.

    # Arguments
    - `g::Function`: Function over knot point and global variable(s) that defines the constraint, `g(vcat(x, globals))`.
    - `global_names::AbstractVector{Symbol}`: Name(s) of the variable(s) to be constrained.
    - `traj::NamedTrajectory`: The trajectory on which the constraint is defined.

    # Keyword Arguments
    - `equality::Bool=true`: If `true`, the constraint is `g(x) = 0`. Otherwise, the constraint is `g(x) ≤ 0`.
    """
    function NonlinearGlobalConstraint(
        g::Function,
        global_names::AbstractVector{Symbol},
        traj::NamedTrajectory;
        equality::Bool=true,
    )
        global_comps = vcat([traj.global_components[name] for name in global_names]...)
        offset_global_comps = traj.dim * traj.T .+ global_comps

        g_eval = g(traj.global_data[global_comps])
        @assert g_eval isa AbstractVector{Float64}
        g_dim = length(g_eval)
        
        @views function g!(δ::AbstractVector, Z⃗::AbstractVector)
            δ[:] = g(Z⃗[offset_global_comps])
            return nothing
        end

        @views function ∂g!(∂g::AbstractMatrix, Z⃗::AbstractVector)
            ForwardDiff.jacobian!(
                ∂g, 
                x -> g(x),
                Z⃗[offset_global_comps]
            )
        end

        # global subspace
        jacobian_structure = spzeros(g_dim, traj.global_dim) 
        jacobian_structure[:, global_comps] .= 1.0

        @views function μ∂²g!(
            μ∂²g::AbstractMatrix,   
            Z⃗::AbstractVector, 
            μ::AbstractVector
        )
            ForwardDiff.hessian!(
                μ∂²g[global_comps, global_comps], 
                x -> μ'g(x), 
                Z⃗[offset_global_comps]
            )
        end

        # global subspace
        hessian_structure = spzeros(traj.global_dim, traj.global_dim) 
        hessian_structure[global_comps, global_comps] .= 1.0

        return new(
            g!,
            ∂g!,
            jacobian_structure,
            μ∂²g!,
            hessian_structure,
            equality,
            g_dim,
        )
    end

    function NonlinearGlobalConstraint(
        g::Function,
        global_name::Symbol,
        traj::NamedTrajectory;
        kwargs...
    )
        return NonlinearGlobalConstraint(g, [global_name], traj; kwargs...)
    end
end

function get_full_jacobian(
    NLC::NonlinearGlobalConstraint, 
    traj::NamedTrajectory
)
    Z_dim = traj.dim * traj.T + traj.global_dim
    ∂g_full = spzeros(NLC.dim, Z_dim)
    ∂g_full[1:NLC.dim, traj.dim * traj.T + 1:Z_dim] = NLC.∂gs
    return ∂g_full
end

function get_full_hessian(
    NLC::NonlinearGlobalConstraint, 
    traj::NamedTrajectory
)
    Z_dim = traj.dim * traj.T + traj.global_dim
    μ∂²g_full = spzeros(Z_dim, Z_dim)
    g_comps = traj.dim * traj.T + 1:Z_dim
    μ∂²g_full[g_comps, g_comps] = NLC.μ∂²gs
    return μ∂²g_full
end

# --------------------------------------------------------------------------- #
# NonlinearGlobalKnotPointConstraint
# --------------------------------------------------------------------------- #

struct NonlinearGlobalKnotPointConstraint <: AbstractNonlinearConstraint
    g!::Function
    ∂g!::Function
    ∂gs::Vector{SparseMatrixCSC}
    μ∂²g!::Function
    μ∂²gs::Vector{SparseMatrixCSC}
    times::Vector{Int}
    equality::Bool
    g_dim::Int
    dim::Int

    """
        Create a NonlinearKnotPointConstraint object with global components.

        TODO: Consolidate with NonlinearKnotPointConstraint?
    """
    function NonlinearGlobalKnotPointConstraint(
        g::Function,
        names::AbstractVector{Symbol},
        global_names::AbstractVector{Symbol},
        traj::NamedTrajectory,
        params::AbstractVector;
        times::AbstractVector{Int}=1:traj.T,
        equality::Bool=true,
        jacobian_structure::Union{Nothing, SparseMatrixCSC}=nothing,
        hessian_structure::Union{Nothing, SparseMatrixCSC}=nothing,
    )
        @assert length(params) == length(times) "params must have the same length as times"

        # collect the components and global components
        x_comps = vcat([traj.components[name] for name in names]...)
        global_comps = vcat([traj.global_components[name] for name in global_names]...)
        offset_global_comps = traj.dim * traj.T .+ global_comps

        # append global data to the trajectory (each slice indexes into Z⃗, creating xg)
        xg_slices = [vcat(slice(t, x_comps, traj.dim), offset_global_comps) for t in times]

        # append global data comps to each knot point (indexes into knot points)
        xg_comps = vcat(x_comps, traj.dim .+ global_comps)
        z_dim = traj.dim + traj.global_dim

        Z⃗ = vec(traj)
        @assert g(Z⃗[xg_slices[1]], params[1]) isa AbstractVector{Float64}
        g_dim = length(g(Z⃗[xg_slices[1]], params[1]))

        @views function g!(δ::AbstractVector, Z⃗::AbstractVector)
            for (i, xg_slice) ∈ enumerate(xg_slices)
                δ[slice(i, g_dim)] = g(Z⃗[xg_slice], params[i])
            end
        end

        @views function ∂g!(∂gs::Vector{<:AbstractMatrix}, Z⃗::AbstractVector)
            for (i, (xg_slice, ∂g)) ∈ enumerate(zip(xg_slices, ∂gs))
                # Disjoint
                ForwardDiff.jacobian!(
                    ∂g[:, xg_comps], 
                    x -> g(x, params[i]),
                    Z⃗[xg_slice]
                )
            end
        end

        @views function μ∂²g!(
            μ∂²gs::Vector{<:AbstractMatrix},   
            Z⃗::AbstractVector, 
            μ::AbstractVector
        )
            for (i, (xg_slice, μ∂²g)) ∈ enumerate(zip(xg_slices, μ∂²gs))
                # Disjoint
                ForwardDiff.hessian!(
                    μ∂²g[xg_comps, xg_comps], 
                    xg -> μ[slice(i, g_dim)]' * g(xg, params[i]), 
                    Z⃗[xg_slice]
                )
            end
        end

        if isnothing(jacobian_structure)
            jacobian_structure = spzeros(g_dim, z_dim) 
            jacobian_structure[:, xg_comps] .= 1.0
        else
            @assert size(jacobian_structure) == (g_dim, z_dim)
        end

        ∂gs = [copy(jacobian_structure) for _ ∈ times]

        if isnothing(hessian_structure)
            hessian_structure = spzeros(z_dim, z_dim) 
            hessian_structure[xg_comps, xg_comps] .= 1.0
        else
            @assert size(hessian_structure) == (z_dim, z_dim)
        end

        μ∂²gs = [copy(hessian_structure) for _ ∈ times]

        return new(
            g!,
            ∂g!,
            ∂gs,
            μ∂²g!,
            μ∂²gs,
            times,
            equality,
            g_dim,
            g_dim * length(times)
        )
    end

    function NonlinearGlobalKnotPointConstraint(
        g::Function,
        names::AbstractVector{Symbol},
        global_names::AbstractVector{Symbol},
        traj::NamedTrajectory;
        times::AbstractVector{Int}=1:traj.T,
        kwargs...
    )
        params = [nothing for _ in times]
        g_param = (x, _) -> g(x)
        return NonlinearGlobalKnotPointConstraint(
            g_param, 
            names,
            global_names,
            traj, 
            params; 
            times=times, 
            kwargs...
        )
    end

    function NonlinearGlobalKnotPointConstraint(g::Function, name::Symbol, args...; kwargs...)
        return NonlinearGlobalKnotPointConstraint(g, [name], args...; kwargs...)
    end

end

function get_full_jacobian(
    NLC::NonlinearGlobalKnotPointConstraint, 
    traj::NamedTrajectory
)
    Z_dim = traj.dim * traj.T + traj.global_dim
    global_slice = traj.dim * traj.T .+ (1:traj.global_dim)
    ∂g_full = spzeros(NLC.dim, Z_dim) 
    for (i, (k, ∂gₖ)) ∈ enumerate(zip(NLC.times, NLC.∂gs))
        # Disjoint
        zg_slice = vcat(slice(k, traj.dim), global_slice)
        ∂g_full[slice(i, NLC.g_dim), zg_slice] .= ∂gₖ
    end
    return ∂g_full
end

function get_full_hessian(
    NLC::NonlinearGlobalKnotPointConstraint, 
    traj::NamedTrajectory
)
    Z_dim = traj.dim * traj.T + traj.global_dim
    global_slice = traj.dim * traj.T .+ (1:traj.global_dim)
    μ∂²g_full = spzeros(Z_dim, Z_dim)
    for (k, μ∂²gₖ) ∈ zip(NLC.times, NLC.μ∂²gs)
        # Overlapping
        zg_slice = vcat(slice(k, traj.dim), global_slice)
        μ∂²g_full[zg_slice, zg_slice] .+= μ∂²gₖ
    end
    return μ∂²g_full
end

# ============================================================================ #

@testitem "testing NonlinearGlobalConstraint" begin    
    include("../../test/test_utils.jl")

    _, traj = bilinear_dynamics_and_trajectory(add_global=true)

    g_fn(g) = [norm(g) - 1.0]

    g_dim = 1

    NLC = NonlinearGlobalConstraint(g_fn, :g, traj; equality=false)
    G_DIM = traj.dim * traj.T .+ traj.global_components[:g]

    ĝ(Z⃗) = g_fn(Z⃗[G_DIM])

    δ = zeros(g_dim)

    NLC.g!(δ, vec(traj))

    @test δ ≈ ĝ(vec(traj))

    NLC.∂g!(NLC.∂gs, vec(traj))

    ∂g_full = Constraints.get_full_jacobian(NLC, traj)

    ∂g_autodiff = ForwardDiff.jacobian(ĝ, vec(traj))

    @test ∂g_full ≈ ∂g_autodiff

    μ = randn(g_dim)

    NLC.μ∂²g!(NLC.μ∂²gs, vec(traj), μ)

    hessian_autodiff = ForwardDiff.hessian(Z -> μ'ĝ(Z), vec(traj))

    μ∂²g_full = Constraints.get_full_hessian(NLC, traj)

    @test μ∂²g_full ≈ hessian_autodiff
end

@testitem "testing NonlinearGlobalKnotPointConstraint" begin
    using TrajectoryIndexingUtils
    
    include("../../test/test_utils.jl")

    _, traj = bilinear_dynamics_and_trajectory(add_global=true)

    function g_fn(ug)
        u, g = ug[1:traj.dims[:u]], ug[traj.dims[:u] + 1:end]
        return [norm(u) - 1.0; norm(u) * norm(g) - 1.0]
    end

    g_dim = 2
    times = 1:traj.T

    NLC = NonlinearGlobalKnotPointConstraint(g_fn, [:u], [:g], traj; times=times, equality=false)
    U_DIM(k) = slice(k, traj.components[:u], traj.dim)
    G_DIM = traj.dim * traj.T .+ traj.global_components[:g]

    ĝ(Z⃗) = vcat([g_fn(Z⃗[vcat(U_DIM(k), G_DIM)]) for k ∈ times]...)

    δ = zeros(g_dim * traj.T)

    NLC.g!(δ, vec(traj))

    @test δ ≈ ĝ(vec(traj))

    NLC.∂g!(NLC.∂gs, vec(traj))

    ∂g_full = Constraints.get_full_jacobian(NLC, traj)

    ∂g_autodiff = ForwardDiff.jacobian(ĝ, vec(traj))
    
    display(∂g_full)
    println()
    display(∂g_autodiff)

    @test ∂g_full ≈ ∂g_autodiff

    μ = randn(g_dim * traj.T)

    NLC.μ∂²g!(NLC.μ∂²gs, vec(traj), μ)

    hessian_autodiff = ForwardDiff.hessian(Z -> μ'ĝ(Z), vec(traj))

    μ∂²g_full = Constraints.get_full_hessian(NLC, traj)

    @test μ∂²g_full ≈ hessian_autodiff
end