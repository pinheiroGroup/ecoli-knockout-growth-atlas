#!/usr/bin/env julia
# figures.jl — Static publication figures for the Keio knockout growth atlas
#
# Reads docs/data/curves_data.json (produced by analyse.jl) and writes:
#   docs/figures/fig_elbow.pdf   — WCSS elbow plot (Fig 2)
#   docs/figures/fig_elbow.png
#
# Run:
#   julia --project=. figures.jl

using Pkg
Pkg.activate(@__DIR__)

using JSON3
using CairoMakie

const DATA_PATH = joinpath(@__DIR__, "..", "docs", "data", "curves_data.json")
const FIG_DIR   = joinpath(@__DIR__, "..", "docs", "figures")

# ─────────────────────────────────────────────────────────────────────────────
# Fig 2: WCSS elbow plot — LB and M63 on the same axes
# ─────────────────────────────────────────────────────────────────────────────

function fig_elbow(d)
    lb  = d["wcss_sweep"]["LB"]
    m63 = d["wcss_sweep"]["M63"]

    ks_lb   = Int.(lb["ks"])
    wcss_lb = Float64.(lb["wcss"])
    opt_lb  = Int(d["optimal_k"]["LB"])

    ks_m63   = Int.(m63["ks"])
    wcss_m63 = Float64.(m63["wcss"])
    opt_m63  = Int(d["optimal_k"]["M63"])

    fig = Figure(size = (560, 380), fontsize = 13)
    ax  = Axis(fig[1, 1];
        xlabel      = "Number of clusters (k)",
        ylabel      = "WCSS",
        xticks      = ks_lb,
        title       = "KinBiont k-means sweep — WCSS elbow",
        titlesize   = 13,
    )

    # LB line + markers
    lines!(ax,  ks_lb,  wcss_lb;  color = :steelblue, linewidth = 2, label = "LB")
    scatter!(ax, ks_lb, wcss_lb;  color = :steelblue, markersize = 7)

    # M63 line + markers
    lines!(ax,  ks_m63, wcss_m63; color = :tomato,    linewidth = 2, label = "M63")
    scatter!(ax, ks_m63, wcss_m63; color = :tomato,   markersize = 7)

    # Elbow markers
    idx_lb  = findfirst(==(opt_lb),  ks_lb)
    idx_m63 = findfirst(==(opt_m63), ks_m63)
    scatter!(ax, [opt_lb],  [wcss_lb[idx_lb]];
             color = :steelblue, markersize = 16, marker = :star5,
             label = "Optimal k=$(opt_lb) (LB)")
    scatter!(ax, [opt_m63], [wcss_m63[idx_m63]];
             color = :tomato,    markersize = 16, marker = :star5,
             label = "Optimal k=$(opt_m63) (M63)")

    axislegend(ax; position = :rt, framevisible = false, labelsize = 11)
    return fig
end

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

function main()
    mkpath(FIG_DIR)

    @info "Loading $DATA_PATH"
    d = open(DATA_PATH) do io; JSON3.read(io); end

    @info "Generating fig_elbow…"
    f = fig_elbow(d)
    save(joinpath(FIG_DIR, "fig_elbow.pdf"), f; pt_per_unit = 1)
    save(joinpath(FIG_DIR, "fig_elbow.png"), f; px_per_unit = 2)
    @info "  → docs/figures/fig_elbow.pdf / .png"
end

main()
