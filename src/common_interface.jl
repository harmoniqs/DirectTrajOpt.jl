"""
Common interface for components (integrators and constraints).

This module defines the generic interface functions that both integrators
and constraints implement through multiple dispatch. This avoids naming
conflicts when both modules are imported.
"""
module CommonInterface

export evaluate!
export jacobian_structure
export jacobian!
export hessian_structure
export hessian_of_lagrangian
export hessian_of_lagrangian!
export eval_jacobian
export eval_hessian_of_lagrangian

using NamedTrajectories
using SparseArrays

"""
    evaluate!(values, component, traj::NamedTrajectory)

Evaluate the component (constraint or dynamics) and store the result in-place in `values`.

For integrators: Computes dynamics violations δ = f(x_{k+1}, x_k, u_k, ...) for all timesteps.
For constraints: Computes constraint violations g(x) for all applicable timesteps/variables.

# Arguments
- `values`: Pre-allocated vector to store the evaluation results
- `component`: An integrator or constraint
- `traj`: The trajectory providing values and structure

# Returns
- Nothing (modifies `values` in-place)
"""
function evaluate! end

"""
    jacobian_structure(component, traj::NamedTrajectory)

Return the sparsity structure of the Jacobian for the given component.
This should return a sparse matrix with the same structure as the Jacobian,
where non-zero entries indicate where partial derivatives exist.

# Arguments
- `component`: An integrator, constraint, or other component
- `traj`: The trajectory providing dimension information

# Returns
- A sparse matrix representing the Jacobian structure
"""
function jacobian_structure end

"""
    jacobian!(∂f, component, args...)

Compute the Jacobian of the component in-place, storing the result in ∂f.

# Arguments
- `∂f`: Pre-allocated sparse matrix for the Jacobian
- `component`: An integrator, constraint, or other component  
- `args...`: Component-specific arguments (e.g., knot points, trajectory values)
"""
function jacobian! end

"""
    hessian_structure(component, traj::NamedTrajectory)

Return the sparsity structure of the Hessian of the Lagrangian for the given component.

# Arguments
- `component`: An integrator, constraint, or other component
- `traj`: The trajectory providing dimension information

# Returns
- A sparse matrix representing the Hessian structure
"""
function hessian_structure end

"""
    hessian_of_lagrangian(component, μ, args...)

Compute the Hessian of the Lagrangian (weighted by multipliers μ) for the component.

# Arguments
- `component`: An integrator, constraint, or other component
- `μ`: Lagrange multipliers
- `args...`: Component-specific arguments (e.g., knot points, trajectory values)

# Returns
- A sparse matrix representing μ'∇²f
"""
function hessian_of_lagrangian end

"""
    hessian_of_lagrangian!(μ∂²f, component, μ, args...)

Compute the Hessian of the Lagrangian in-place, storing the result in μ∂²f.

# Arguments
- `μ∂²f`: Pre-allocated sparse matrix for the Hessian
- `component`: An integrator, constraint, or other component
- `μ`: Lagrange multipliers
- `args...`: Component-specific arguments (e.g., knot points, trajectory values)
"""
function hessian_of_lagrangian! end

"""
    eval_jacobian(component, traj::NamedTrajectory)

High-level method to evaluate and return the full Jacobian for the component.

For integrators: Computes the Jacobian using ForwardDiff across all timesteps.
For constraints: Calls jacobian! to fill compact storage, then assembles the full sparse Jacobian.

# Arguments
- `component`: An integrator or constraint
- `traj`: The trajectory providing values and structure

# Returns
- A sparse matrix representing the full Jacobian
"""
function eval_jacobian end

"""
    eval_hessian_of_lagrangian(component, traj::NamedTrajectory, μ::AbstractVector)

High-level method to evaluate and return the full Hessian of the Lagrangian for the component.

For integrators: Computes the Hessian using ForwardDiff across all timesteps.
For constraints: Calls hessian_of_lagrangian to fill compact storage, then assembles the full sparse Hessian.

# Arguments
- `component`: An integrator or constraint
- `traj`: The trajectory providing values and structure
- `μ`: Lagrange multipliers

# Returns
- A sparse matrix representing the full Hessian μ'∇²f
"""
function eval_hessian_of_lagrangian end

end
