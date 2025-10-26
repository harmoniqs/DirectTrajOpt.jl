using DirectTrajOpt
using PiccoloDocsTemplate

pages = [
    "Home" => "index.md",
    "Guide" => "generated/explanation.md",
    "Library" => "lib.md",
]

generate_docs(
    @__DIR__,
    "DirectTrajOpt",
    DirectTrajOpt,
    pages;
    format_kwargs = (canonical = "https://docs.harmoniqs.co/DirectTrajOpt.jl",),
)