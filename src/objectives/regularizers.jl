export QuadraticRegularizer
export PairwiseQuadraticRegularizer
# ----------------------------------------------------------------------------- #
# Quadratic Regularizer
# ----------------------------------------------------------------------------- #

"""
    QuadraticRegularizer

A quadratic regularizer for a trajectory component.

Fields:
    `name`: the name of the trajectory component to regularize
    `traj`: the trajectory
    `R`: the regularization matrix diagonal
    `baseline`: the baseline values for the trajectory component
    `times`: the times at which to evaluate the regularizer
"""
function QuadraticRegularizer(
    name::Symbol,
    traj::NamedTrajectory,
    R::AbstractVector{<:Real};
    baseline::AbstractMatrix{<:Real}=zeros(traj.dims[name], traj.T),
    times::AbstractVector{Int}=1:traj.T,
)
    @assert length(R) == traj.dims[name] "R must have the same length as the dimension of the trajectory component"

    @views function L(Z⃗::AbstractVector{<:Real})
        J = 0.0
        for t ∈ times
            if traj.timestep isa Symbol
                Δt = Z⃗[slice(t, traj.components[traj.timestep], traj.dim)]
            else
                Δt = traj.timestep
            end

            vₜ = Z⃗[slice(t, traj.components[name], traj.dim)]
            Δv = vₜ - baseline[:, t]

            rₜ = Δt .* Δv
            J += 0.5 * rₜ' * (R .* rₜ)
        end
        return J
    end

    @views function ∇L(Z⃗::AbstractVector)
        ∇ = zeros(traj.dim * traj.T + traj.global_dim)
        Threads.@threads for t ∈ times
            vₜ_slice = slice(t, traj.components[name], traj.dim)
            Δv = Z⃗[vₜ_slice] .- baseline[:, t]

            if traj.timestep isa Symbol
                Δt_slice = slice(t, traj.components[traj.timestep], traj.dim)
                Δt = Z⃗[Δt_slice]
                ∇[Δt_slice] .= Δv' * (R .* (Δt .* Δv))
            else
                Δt = traj.timestep
            end

            ∇[vₜ_slice] .= R .* (Δt.^2 .* Δv)
        end
        return ∇
    end

    function ∂²L_structure()
        structure = []
        # Hessian structure (eq. 17)
        for t ∈ times
            vₜ_slice = slice(t, traj.components[name], traj.dim)
            vₜ_vₜ_inds = collect(zip(vₜ_slice, vₜ_slice))
            append!(structure, vₜ_vₜ_inds)

            if traj.timestep isa Symbol
                Δt_slice = slice(t, traj.components[traj.timestep], traj.dim)
                # ∂²_vₜ_Δt
                vₜ_Δt_inds = [(i, j) for i ∈ vₜ_slice for j ∈ Δt_slice]
                append!(structure, vₜ_Δt_inds)
                # ∂²_Δt_vₜ
                Δt_vₜ_inds = [(i, j) for i ∈ Δt_slice for j ∈ vₜ_slice]
                append!(structure, Δt_vₜ_inds)
                # ∂²_Δt_Δt
                Δt_Δt_inds = collect(zip(Δt_slice, Δt_slice))
                append!(structure, Δt_Δt_inds)
            end
        end
        return structure
    end

    function ∂²L(Z⃗::AbstractVector) 
        values = []
        # Match Hessian structure indices
        for t ∈ times
            if traj.timestep isa Symbol
                Δt = Z⃗[slice(t, traj.components[traj.timestep], traj.dim)]
                append!(values, R .* Δt.^2)
                # ∂²_vₜ_Δt, ∂²_Δt_vₜ
                vₜ = Z⃗[slice(t, traj.components[name], traj.dim)]
                Δv = vₜ .- baseline[:, t]
                append!(values, 2 * (R .* (Δt .* Δv)))
                append!(values, 2 * (R .* (Δt .* Δv)))
                # ∂²_Δt_Δt
                append!(values, Δv' * (R .* Δv))
            else
                Δt = traj.timestep
                append!(values, R .* Δt.^2)
            end
        end
        return values
    end

    return Objective(L, ∇L, ∂²L, ∂²L_structure)
end

function QuadraticRegularizer(
    name::Symbol,
    traj::NamedTrajectory,
    R::Real;
    kwargs...
)
    return QuadraticRegularizer(name, traj, R * ones(traj.dims[name]); kwargs...)
end



#####
function PairwiseQuadraticRegularizer(
    name1::Symbol,
    name2::Symbol,
    traj::NamedTrajectory,
    R::Float64;
)
    if traj.timestep isa Symbol
        J = KnotPointObjective(
        (v,p) -> 1/2 * sum(abs2, (v[1:length(v) ÷ 2]-v[length(v) ÷ 2+1:end] ) * p ) ,
        [name1,name2],
        traj,
        [t[1] for t in eachcol(traj[traj.timestep])];
        times =  1:traj.T,
        Qs = R * ones(traj.T)
        )
        
    else
        Δt  = traj.timestep
        J = KnotPointObjective(
        v -> 1/2 * sum(abs2, (v[1:length(v) ÷ 2]-v[length(v) ÷ 2+1:end]) * Δt),
        [name1,name2],
        traj,
        [nothing for _ in 1:traj.T];
        times =  1:traj.T,
        Qs = R * ones(traj.T)
        )
    end 
end