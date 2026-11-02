@testitem "Aqua quality assurance" tags=[:aqua] begin
    using Aqua, DirectTrajOpt

    Aqua.test_all(
        DirectTrajOpt;
        deps_compat = (check_extras = false,),
        # `hessian_structure` is exported by three different DirectTrajOpt
        # submodules (CommonInterface, Constraints, Integrators); the conflict
        # makes it appear undefined at DirectTrajOpt's surface even though all
        # three sub-definitions exist. TODO: pick a single canonical owner and
        # have the other submodules `import ..CommonInterface: hessian_structure`
        # rather than re-export.
        undefined_exports = (broken = true,),
        # persistent_tasks cannot be asserted on 1.12: task detection is
        # nondeterministic BOTH directions — when tasks are detected the
        # broken-marked test holds (broken), when the scheduler comes up
        # clean Aqua reports 'Unexpected Pass' as an ERROR (observed across
        # #133, #135, #136, the 0.10.0 release PR, and bit-identically on
        # pre-MadNLP-flip main @ 699a518 in the 2026-10 migration campaign's
        # characterization runs). Since #144 the CI matrix is 1.12-ONLY, so
        # every configuration of this check reddens CI at random. It asserts
        # nothing stable — skipped, not silenced-by-broken. Revisit when the
        # Aqua/1.12 persistent-task interaction is understood.
        persistent_tasks = false,
    )
end
