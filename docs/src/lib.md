#  Library

```@meta
CollapsedDocStrings = true
```

## Common Interface
```@autodocs
Modules = [DirectTrajOpt.CommonInterface]
```

## Constraints
```@autodocs
Modules = [DirectTrajOpt.Constraints]
Filter = t -> t ∉ [DirectTrajOpt.CommonInterface.jacobian_structure, DirectTrajOpt.CommonInterface.jacobian!, DirectTrajOpt.CommonInterface.hessian_structure, DirectTrajOpt.CommonInterface.hessian_of_lagrangian!]
```

## Integrators
```@autodocs
Modules = [DirectTrajOpt.Integrators]
``` 

## Objectives
```@autodocs
Modules = [DirectTrajOpt.Objectives]
```

## Problems
```@autodocs
Modules = [DirectTrajOpt.Problems]
```

## Solvers
```@autodocs
Modules = [DirectTrajOpt.Solvers]
```

## Ipopt Solver
```@autodocs
Modules = [DirectTrajOpt.IpoptSolverExt]
```