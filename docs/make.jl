using DirectTrajOpt
using PiccoloDocsTemplate

pages = [
    "Home" => "index.md",
    "Lib" => "lib.md",
    "Explanation" => "generated/explanation.md",
]

generate_docs(
    @__DIR__,
    "DirectTrajOpt",
    DirectTrajOpt,
    pages;
    format_kwargs = (canonical = "https://docs.harmoniqs.co/DirectTrajOpt.jl",),
)