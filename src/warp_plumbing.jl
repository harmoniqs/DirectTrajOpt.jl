#
# Warp plumbing — DTO-side packed-coordinate helpers for trajectories that carry a
# NamedTrajectories time warp (NamedTrajectories#161).
#
# Under a warp the timestep rows are DERIVED: present as components (evaluation code
# that reads `traj[k].timestep` keeps working unchanged) but excluded from the packed
# decision vector. The packed layout of `vec(traj)` is
#
#     [ non-derived rows of knot 1 … knot N ] ++ [ global data ] ++ [ warp params ]
#
# with `n_params(warp)` trailing entries — for `GlobalScale`, the scalar duration `T`
# that replaces the per-knot Δt block of a free-time formulation. Every DTO surface
# that addresses the packed vector (evaluator structures, integrator Jacobians,
# objective gradients/Hessians, linear-constraint application) must use these packed
# positions. Without a warp every helper below returns exactly the historical
# `TrajectoryIndexingUtils` value — warp-free behavior is bit-unchanged.
#
module WarpPlumbing

using NamedTrajectories
using TrajectoryIndexingUtils: slice, index
using TestItems

export has_time_warp
export packed_knot_dim, packed_length, packed_globals_base, packed_warp_base
export packed_row_index, packed_slice, packed_two_knot_window
export warp_param_indices, derived_row_chain

"""
    has_time_warp(traj::NamedTrajectory)::Bool

Whether the trajectory carries a time warp (its timestep rows are derived).
"""
has_time_warp(traj::NamedTrajectory) = traj.warp !== nothing

"""
    packed_knot_dim(traj::NamedTrajectory)::Int

Per-knot width of the packed vector: all component rows except the derived timestep.
Without a warp this is `traj.dim` — identical to the historical layout.
"""
function packed_knot_dim(traj::NamedTrajectory)
    return traj.warp === nothing ? traj.dim : traj.dim - traj.dims[traj.timestep]
end

"""
    packed_length(traj::NamedTrajectory)::Int

Total length of the packed decision vector: `== length(traj)`. Without a warp this is
the historical `traj.dim * traj.N + traj.global_dim`.
"""
packed_length(traj::NamedTrajectory) = length(traj)

"""
    packed_globals_base(traj::NamedTrajectory)::Int

Number of packed positions preceding the global-data block (0-based base of the
globals). Without a warp: `traj.dim * traj.N`.
"""
packed_globals_base(traj::NamedTrajectory) = packed_knot_dim(traj) * traj.N

"""
    packed_warp_base(traj::NamedTrajectory)::Int

0-based base of the trailing warp-parameter block in the packed vector (`0` without a
warp). Warp parameter `j` lives at packed position `packed_warp_base(traj) + j`.
"""
function packed_warp_base(traj::NamedTrajectory)
    traj.warp === nothing && return 0
    return packed_globals_base(traj) + traj.global_dim
end

"""
    packed_row_index(traj::NamedTrajectory, k::Int, r::Int)::Int

Packed position of knot `k`'s datavec row `r` (1-based within the knot). The derived
timestep row has NO packed position — it is never decision data — and asking for it
errors. Without a warp this is the historical `index(k, r, traj.dim)`.
"""
function packed_row_index(traj::NamedTrajectory, k::Int, r::Int)
    if traj.warp === nothing
        return index(k, r, traj.dim)
    end
    zdim = packed_knot_dim(traj)
    dt_first = first(traj.components[traj.timestep])
    dt_dim = traj.dims[traj.timestep]
    dt_first ≤ r < dt_first + dt_dim && error(
        "the derived timestep row has no packed index — it is never decision data " *
        "under a time warp (NamedTrajectories#161)",
    )
    return (k - 1) * zdim + (r ≤ dt_first ? r : r - dt_dim)
end

"""
    packed_slice(traj::NamedTrajectory, k::Int, comps::AbstractVector{Int})::Vector{Int}

Packed positions of knot `k`'s component rows `comps` (datavec rows, 1-based within a
knot, none derived). Without a warp this is the historical
`slice(k, comps, traj.dim)` — bit-identical.
"""
function packed_slice(traj::NamedTrajectory, k::Int, comps::AbstractVector{Int})
    traj.warp === nothing && return slice(k, comps, traj.dim)
    return [packed_row_index(traj, k, r) for r in comps]
end

"""
    packed_two_knot_window(traj::NamedTrajectory, k::Int)::Vector{Int}

Packed positions of the two-knot datavec window `[knot k; knot k+1]` (rows
`1:2*traj.dim` in datavec coordinates), with derived timestep rows dropped. Without a
warp this is the historical `slice(k, 1:2traj.dim, traj.dim)` — bit-identical.
"""
function packed_two_knot_window(traj::NamedTrajectory, k::Int)
    traj.warp === nothing && return slice(k, 1:2traj.dim, traj.dim)
    dt_first = first(traj.components[traj.timestep])
    dt_dim = traj.dims[traj.timestep]
    rows = [r for r = 1:traj.dim if !(dt_first ≤ r < dt_first + dt_dim)]
    return vcat(packed_slice(traj, k, rows), packed_slice(traj, k + 1, rows))
end

"""
    warp_param_indices(traj::NamedTrajectory)::Vector{Int}

Packed positions of the warp parameters (empty without a warp) — the trailing block
of `vec(traj)`. For `GlobalScale` this is the single duration variable `T`.
"""
function warp_param_indices(traj::NamedTrajectory)
    warp = traj.warp
    warp === nothing && return Int[]
    return [packed_warp_base(traj) + j for j = 1:n_params(warp)]
end

"""
    derived_row_chain(warp, N::Int)::Matrix

Chain rule `∂Δt_row(k)/∂θⱼ` for the DERIVED (padded) timestep row of length `N`:
`ddeltats_dparams(warp, N)` stacked with its own last row (the padded row `N` repeats
interval `N-1`, mirroring `deltats_row`). Exact for `GlobalScale` (NT's override:
`∂Δtₖ/∂T = wₖ`, the rational lattice weight).
"""
function derived_row_chain(warp::AbstractTimeWarp, N::Int)
    M = ddeltats_dparams(warp, N)
    return [M; M[end:end, :]]
end

# ============================================================================ #
# Tests
# ============================================================================ #

@testitem "warp plumbing: packed index map agrees with NT's packed vec layout" begin
    using NamedTrajectories
    using DirectTrajOpt.WarpPlumbing
    using Test

    # A trajectory whose timestep is NOT the first component (the harder case:
    # the packing drops a MIDDLE row per knot, shifting everything after it).
    N = 6
    T0 = 2.5
    warp = GlobalScale(T0)
    traj = NamedTrajectory(
        (x = randn(2, N), u = randn(1, N), Δt = fill(T0 / (N - 1), 1, N));
        controls = (:u,),
        timestep = :Δt,
        warp = warp,
    )

    @test has_time_warp(traj)
    @test packed_knot_dim(traj) == 3                      # x (2) + u (1), Δt derived
    @test packed_length(traj) == length(vec(traj))
    @test packed_length(traj) == 3 * N + n_params(warp)   # no globals here
    @test packed_globals_base(traj) == 3 * N
    @test packed_warp_base(traj) == 3 * N
    @test warp_param_indices(traj) == [3 * N + 1]         # GlobalScale: the scalar T

    # Ground truth: NT's own `vec` packing. The packed position of knot k's
    # datavec row r must equal the position vec(traj) assigns to that value.
    z = collect(vec(traj))
    dt_idx = first(traj.components[:Δt])
    for k = 1:N, r = 1:traj.dim
        r == dt_idx && continue   # the derived row has no packed position
        z_packed = z[packed_row_index(traj, k, r)]
        z_datavec = traj.datavec[(k-1)*traj.dim+r]
        @test z_packed == z_datavec
    end
    @test_throws ErrorException packed_row_index(traj, 2, dt_idx)

    # Globals and warp params trail the knots.
    ztraj_g = NamedTrajectory(
        (x = randn(2, N), u = randn(1, N), Δt = fill(T0 / (N - 1), 1, N)),
        (g = [1.0, -1.0],);
        controls = (:u,),
        timestep = :Δt,
        warp = warp,
    )
    zg = collect(vec(ztraj_g))
    @test packed_globals_base(ztraj_g) == 3 * N
    @test zg[packed_globals_base(ztraj_g) .+ (1:2)] == ztraj_g.global_data
    @test zg[packed_warp_base(ztraj_g)+1] == T0
    @test warp_param_indices(ztraj_g) == [3 * N + 3]

    # derived_row_chain: the padded row repeats interval N-1 (exact for GlobalScale).
    chain = derived_row_chain(warp, N)
    @test size(chain) == (N, 1)
    @test chain == fill(1 / (N - 1), N, 1)
    @test sum(chain[1:(N-1)]) ≈ 1.0

    # Warp-free: every helper is the historical TrajectoryIndexingUtils value.
    plain = NamedTrajectory(
        (x = randn(2, N), u = randn(1, N), Δt = fill(0.2, 1, N));
        controls = (:u,),
        timestep = :Δt,
    )
    @test !has_time_warp(plain)
    @test packed_knot_dim(plain) == plain.dim
    @test packed_length(plain) == plain.dim * plain.N + plain.global_dim
    @test packed_row_index(plain, 3, 2) == (3 - 1) * plain.dim + 2
    @test packed_slice(plain, 3, [1, 3]) == [2 * plain.dim + 1, 2 * plain.dim + 3]
    @test packed_two_knot_window(plain, 2) == collect((2-1)*plain.dim .+ (1:2plain.dim))
    @test isempty(warp_param_indices(plain))
    @test packed_warp_base(plain) == 0
end

end
