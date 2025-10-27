using DirectTrajOpt
using PiccoloDocsTemplate

pages = [
    "Home" => "index.md",
    "Getting Started" => [
        "Quickstart" => "generated/quickstart.md",
    ],
    "Core Concepts" => [
        "Problem Formulation" => "generated/concepts/problem_formulation.md",
        "Trajectories" => "generated/concepts/trajectories.md",
        "Integrators" => "generated/concepts/integrators.md",
        "Objectives" => "generated/concepts/objectives.md",
        "Constraints" => "generated/concepts/constraints.md",
    ],
    "Library" => "lib.md",
]

generate_docs(
    @__DIR__,
    "DirectTrajOpt",
    DirectTrajOpt,
    pages;
    format_kwargs = (canonical = "https://docs.harmoniqs.co/DirectTrajOpt.jl",),
)