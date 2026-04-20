# Diagnose global phase bifurcation between Ipopt and MadNLP
#
# Run this AFTER the main comparison script has populated:
#   qcp_ipopt, qcp_madnlp, integrator, qtraj
#
# Or run standalone — it sets everything up.

import Ipopt
import MadNLP

using DirectTrajOpt
using Piccolo
using Piccolissimo
using LinearAlgebra

function _get_MadNLPSolverExt()
    cur_mods = reverse(Base.loaded_modules_order)
    candidate_mods = [n for n = cur_mods if Symbol(n) == :MadNLPSolverExt]
    length(candidate_mods) == 1 || error("Expected 1 MadNLPSolverExt, found $(length(candidate_mods))")
    return candidate_mods[1]
end
const MadNLPSolverExt = _get_MadNLPSolverExt()

# ---------------------------------------------------------------------------- #
# Setup: same problem for both solvers
# ---------------------------------------------------------------------------- #

H_drift = 0.5 * PAULIS.Z
H_drives = [PAULIS.X, PAULIS.Y]
drive_bounds = [1., 1.]
sys = QuantumSystem(H_drift, H_drives, drive_bounds)
T = 10.0
N = 100
U_goal = GATES.X

qtraj = UnitaryTrajectory(sys, U_goal, T)
integrator = HermitianExponentialIntegrator(qtraj, N)

# Build two identical problems
qcp = SmoothPulseProblem(qtraj, N; Q=1e2, R=1e-2, ddu_bound=1e0, Δt_bounds=(0.5, 1.5), integrator=integrator)
qcp_ipopt = deepcopy(qcp)
qcp_madnlp = deepcopy(qcp)

# ---------------------------------------------------------------------------- #
# Solve
# ---------------------------------------------------------------------------- #

println("\n" * "="^70)
println("Solving with Ipopt...")
println("="^70)
solve!(qcp_ipopt; options=IpoptOptions(max_iter=100))

println("\n" * "="^70)
println("Solving with MadNLP...")
println("="^70)
solve!(qcp_madnlp; options=MadNLPSolverExt.MadNLPOptions(max_iter=100))

# ---------------------------------------------------------------------------- #
# Extract final unitaries from the solver trajectory data
# ---------------------------------------------------------------------------- #

function get_final_unitary(qcp)
    traj = get_trajectory(qcp)
    # The state Ũ⃗ at the final knot point
    Ũ⃗_final = traj[:Ũ⃗][:, end]
    return iso_vec_to_operator(Ũ⃗_final)
end

U_ipopt = get_final_unitary(qcp_ipopt)
U_madnlp = get_final_unitary(qcp_madnlp)

# ---------------------------------------------------------------------------- #
# Diagnose: what phase did each solver converge to?
# ---------------------------------------------------------------------------- #

println("\n" * "="^70)
println("PHASE ANALYSIS")
println("="^70)

# The trace inner product tr(U_goal' * U_final)
# For a perfect solution: tr(X' * (α·X)) = α·tr(I) = 2α
# where α = ±i (from Schrödinger evolution: U = exp(-iHt))
trace_ipopt = tr(U_goal' * U_ipopt)
trace_madnlp = tr(U_goal' * U_madnlp)

println("\ntr(U_goal' * U_final):")
println("  Ipopt:  $trace_ipopt")
println("  MadNLP: $trace_madnlp")

println("\nPhase angle (arg of trace):")
println("  Ipopt:  $(angle(trace_ipopt)) rad  ($(rad2deg(angle(trace_ipopt)))°)")
println("  MadNLP: $(angle(trace_madnlp)) rad  ($(rad2deg(angle(trace_madnlp)))°)")

println("\nPhase sign diagnostic (im * tr / |tr|):")
phase_ipopt = trace_ipopt / abs(trace_ipopt)
phase_madnlp = trace_madnlp / abs(trace_madnlp)
println("  Ipopt:  $phase_ipopt")
println("  MadNLP: $phase_madnlp")

println("\nFidelity (abs2-based, phase-insensitive):")
n = size(U_goal, 1)
fid_ipopt = abs2(trace_ipopt) / n^2
fid_madnlp = abs2(trace_madnlp) / n^2
println("  Ipopt:  $fid_ipopt")
println("  MadNLP: $fid_madnlp")

println("\nFidelity via Piccolo API:")
println("  Ipopt:  $(fidelity(qcp_ipopt))")
println("  MadNLP: $(fidelity(qcp_madnlp))")

# ---------------------------------------------------------------------------- #
# Check if solvers converged to different phase basins
# ---------------------------------------------------------------------------- #

phase_diff = angle(trace_ipopt) - angle(trace_madnlp)
# Normalize to [-π, π]
phase_diff = mod(phase_diff + π, 2π) - π

println("\n" * "="^70)
println("PHASE DIFFERENCE BETWEEN SOLVERS")
println("="^70)
println("  Δφ = $(phase_diff) rad  ($(rad2deg(phase_diff))°)")
if abs(phase_diff) > π/2
    println("  ⚠ SOLVERS CONVERGED TO DIFFERENT PHASE BASINS")
    println("    This is the bifurcation Gennadi identified.")
    println("    The abs2 in the objective makes ±(im·U_goal) equivalent.")
else
    println("  ✓ Solvers converged to the same phase basin this time.")
    println("    Re-run to potentially observe bifurcation (depends on numerics).")
end

# ---------------------------------------------------------------------------- #
# Show the actual final unitaries for inspection
# ---------------------------------------------------------------------------- #

println("\n" * "="^70)
println("FINAL UNITARIES")
println("="^70)

println("\nU_goal = X gate:")
display(U_goal)

println("\n\nU_final (Ipopt):")
display(U_ipopt)

println("\n\nU_final (MadNLP):")
display(U_madnlp)

println("\n\nU_final / (im * U_goal) — should be ≈ ±1 * I if converged:")
println("\n  Ipopt:  U / (i·X) =")
display(U_ipopt / (im * U_goal))
println("\n  MadNLP: U / (i·X) =")
display(U_madnlp / (im * U_goal))

# ---------------------------------------------------------------------------- #
# Constraint violations (for completeness)
# ---------------------------------------------------------------------------- #

println("\n" * "="^70)
println("CONSTRAINT VIOLATIONS")
println("="^70)

deltas_ipopt = zeros(integrator.dim)
deltas_madnlp = zeros(integrator.dim)
DirectTrajOpt.evaluate!(deltas_ipopt, integrator, get_trajectory(qcp_ipopt))
DirectTrajOpt.evaluate!(deltas_madnlp, integrator, get_trajectory(qcp_madnlp))

println("  max |constraint| Ipopt:  $(maximum(abs.(deltas_ipopt)))")
println("  max |constraint| MadNLP: $(maximum(abs.(deltas_madnlp)))")

# ---------------------------------------------------------------------------- #
# Datavec comparison
# ---------------------------------------------------------------------------- #

println("\n" * "="^70)
println("TRAJECTORY DIVERGENCE")
println("="^70)

data_ipopt = get_trajectory(qcp_ipopt).data
data_madnlp = get_trajectory(qcp_madnlp).data
max_diff = maximum(abs.(data_ipopt .- data_madnlp))
println("  max |data_ipopt - data_madnlp| = $max_diff")

println("\n" * "="^70)
println("DIAGNOSIS COMPLETE")
println("="^70)
println("""
The objective `abs2(tr(U_goal' * U)) / n^2` at:
  qc2/src/quantum_objectives.jl:49

is phase-insensitive: abs2(z) = abs2(-z) = abs2(iz) = ...
Both +im*U_goal and -im*U_goal are equally valid optima.

When Δt is a free variable, the expanded search space creates
multiple basins of attraction. The solver's early numerical
trajectory (from initialization + step selection) determines
which basin it falls into.

With fixed Δt, the landscape is more constrained and both
solvers consistently find the same basin.
""")
