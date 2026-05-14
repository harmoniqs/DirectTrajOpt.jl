@testitem "Aqua quality assurance" tags=[:aqua] begin
    using Aqua, DirectTrajOpt

    Aqua.test_all(
        DirectTrajOpt;
        # `hessian_structure` is exported by three different DirectTrajOpt
        # submodules (CommonInterface, Constraints, Integrators); the conflict
        # makes it appear undefined at DirectTrajOpt's surface even though all
        # three sub-definitions exist. TODO: pick a single canonical owner and
        # have the other submodules `import ..CommonInterface: hessian_structure`
        # rather than re-export.
        undefined_exports = (broken = true,),
    )
end
