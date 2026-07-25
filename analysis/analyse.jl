#!/usr/bin/env julia
# analyse.jl — Growth curve analysis for E. coli Keio knockout dataset
#
# Setup (run once):
#   julia --project=. -e '
#     using Pkg
#     Pkg.develop(path="../../KinBiont.jl")
#     Pkg.instantiate()
#   '
#
# Run:
#   julia --project=. analyse.jl
#
# Output: ../docs/data/curves_data.json

using XLSX
using CSV
using DataFrames
using Statistics
using JSON3
using Kinbiont

const DATA_DIR    = joinpath(@__DIR__, "../data")
const RESULTS_DIR = joinpath(@__DIR__, "../results")
# The tracked file the frontend and analysis/enrichment_nongrowers.py read.
# This used to resolve under DATA_DIR, i.e. ../data/docs/data/, so runs wrote to
# an untracked copy and every consumer kept reading a stale checked-in file.
const OUT_PATH    = joinpath(@__DIR__, "..", "docs", "data", "curves_data.json")

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

_as_float(v) = (ismissing(v) || v === nothing) ? NaN : Float64(v)
_as_str(v)   = (ismissing(v) || v === nothing) ? ""  : string(v)

# Linearly interpolate a curve (times_in, vals_in) onto a new time grid (times_out).
# Only interpolates within [first_finite, last_finite] of the input; positions outside
# that range are filled with NaN.
function _interp1(times_in::Vector{Float64}, vals_in::Vector{Float64},
                  times_out::Vector{Float64})::Vector{Float64}
    n_in  = length(times_in)
    n_out = length(times_out)
    out   = fill(NaN, n_out)

    # Find valid (finite) range of input
    fi = findfirst(i -> isfinite(times_in[i]) && isfinite(vals_in[i]), 1:n_in)
    li = findlast( i -> isfinite(times_in[i]) && isfinite(vals_in[i]), 1:n_in)
    (fi === nothing || li === nothing) && return out

    t_start = times_in[fi]; t_end = times_in[li]

    for k in 1:n_out
        t = times_out[k]
        (t < t_start || t > t_end) && continue
        # Binary search for bracketing interval
        lo = fi; hi = li
        while hi - lo > 1
            mid = (lo + hi) ÷ 2
            times_in[mid] <= t ? (lo = mid) : (hi = mid)
        end
        dt = times_in[hi] - times_in[lo]
        if dt < 1e-12
            out[k] = vals_in[lo]
        else
            α = (t - times_in[lo]) / dt
            out[k] = (1 - α) * vals_in[lo] + α * vals_in[hi]
        end
    end
    return out
end

# Interpolate a matrix of gene mean curves (n_genes × n_tp_in) with the given time vector
# to a common uniform grid covering the intersection of all curves' valid ranges.
# n_grid: number of grid points.
# Returns (grid_times, interpolated_matrix).
# Replace NaN/Inf in each column with the column finite mean (or 0 if all non-finite).
function _fill_nan_colmean(m::Matrix{Float64})::Matrix{Float64}
    out = copy(m)
    for j in axes(m, 2)
        col    = m[:, j]; finite = filter(isfinite, col)
        fill_v = isempty(finite) ? 0.0 : Statistics.mean(finite)
        for i in axes(m, 1)
            isfinite(out[i, j]) || (out[i, j] = fill_v)
        end
    end
    return out
end

# Load the replicate-averaged matrix that scripts/01_build_gene_means.py writes.
#
# Clustering (and therefore the non-growing pre-screen) must run on these curves,
# not on a matrix re-aggregated here from the raw workbooks. The Python builder
# extends every replicate to the full run before averaging (leading values back-
# filled, internal gaps carried forward, post-final values carried forward), so
# each strain is judged over the whole 0.25–50 h window. Re-aggregating instead
# truncates each strain at the earliest replicate dropout, which cuts late
# growers short and makes them look flat: strains such as ptsH, ybhM and yccE
# reach OD 0.6–0.8 only after 23 h and were previously mis-assigned to the
# non-growing sentinel. `genes` fixes the row order.
function load_gene_mean_matrix(
    medium::String,
    genes::Vector{String},
)::Tuple{Vector{Float64}, Matrix{Float64}}
    path = joinpath(RESULTS_DIR, "keio_$(lowercase(medium))_gene_means.csv")
    isfile(path) || error(
        "Replicate-averaged matrix not found: $path\n" *
        "Run `python scripts/01_build_gene_means.py` first."
    )
    df    = CSV.read(path, DataFrame)
    times = Float64.(df[!, 1])

    missing_genes = filter(g -> !hasproperty(df, Symbol(g)), genes)
    isempty(missing_genes) || error(
        "$medium: $(length(missing_genes)) gene(s) absent from $path: " *
        join(first(missing_genes, 10), ", ")
    )

    mat = Matrix{Float64}(undef, length(genes), length(times))
    for (i, g) in enumerate(genes)
        mat[i, :] = Float64.(df[!, Symbol(g)])
    end
    return times, mat
end

# Build a clustering matrix on a robust common time grid.
# t_end is chosen as the `end_quantile` quantile of per-gene valid endpoints,
# so a few genes with very short replicates do not truncate the entire dataset.
# Genes that don't reach t_end get NaN in the remaining columns; those are then
# imputed with _fill_nan_colmean so k-means receives a fully finite matrix.
function _interp_to_cluster_grid(
    times::Vector{Float64},
    mat::Matrix{Float64};
    n_grid::Int        = 100,
    end_quantile::Float64 = 0.05,   # use 5th-percentile t_end so 95% of genes fully covered
)::Tuple{Vector{Float64}, Matrix{Float64}}
    n_genes = size(mat, 1)

    t_starts = Float64[]; t_ends = Float64[]
    for i in 1:n_genes
        row = mat[i, :]
        fi = findfirst(j -> isfinite(row[j]), eachindex(row))
        li = findlast( j -> isfinite(row[j]), eachindex(row))
        (fi === nothing || li === nothing) && continue
        push!(t_starts, times[fi]); push!(t_ends, times[li])
    end
    isempty(t_starts) && return (times, mat)

    t0 = maximum(t_starts)
    t1 = quantile(t_ends, end_quantile)   # robust: not the global minimum
    t0 >= t1 && (t1 = minimum(t_ends))    # fallback if quantile still fails

    grid = collect(range(t0, t1; length = n_grid))

    out = Matrix{Float64}(undef, n_genes, n_grid)
    for i in 1:n_genes
        out[i, :] = _interp1(times, mat[i, :], grid)
    end
    # Impute any remaining NaN (genes whose valid range ends before t1)
    out = _fill_nan_colmean(out)
    return grid, out
end

# Emit the centroid table Figure 2c reads: one row per cluster per scale, with
# the cluster size and the centroid sampled on the clustering grid.
const CENTROID_CSV_DIR = "/media/aivuk/64fce268-2613-4033-b39f-537ae2d28805/roms/pinheiroTech/GUIbiontPaper/scripts"

function write_centroid_csv(medium::String, times, curves, cl)
    isdir(CENTROID_CSV_DIR) || return
    zs   = Kinbiont._zscore_rows(curves)
    cols = ["centroid_t_$(round(t; digits=4))" for t in times]
    rows = NamedTuple[]
    for scale in ("raw", "normalized")
        src = scale == "raw" ? curves : zs
        for k in 1:cl.optimal_k
            idx = findall(==(k), cl.clusters)
            isempty(idx) && continue
            c = vec(mean(src[idx, :], dims=1))
            push!(rows, (; cluster=k, n_series=length(idx), type=scale,
                         NamedTuple{Tuple(Symbol.(cols))}(Tuple(round.(c; digits=6)))...))
        end
    end
    path = joinpath(CENTROID_CSV_DIR, "cluster_centroids_raw_and_normalized_$(medium).csv")
    CSV.write(path, DataFrame(rows))
    @info "  Centroid table written to $path"
end

function find_elbow(ks, wcss_vals)
    # Largest second-difference (maximum curvature in the elbow)
    length(ks) < 3 && return ks[end]
    d2 = diff(diff(wcss_vals))
    return ks[argmax(d2) + 1]
end

function raw_centroids(curves::Matrix{Float64}, labels::Vector{Int}, n_k::Int)
    n_tp = size(curves, 2)
    cents = zeros(n_k, n_tp)
    for k in 1:n_k
        idx = findall(==(k), labels)
        isempty(idx) && continue
        cents[k, :] = vec(mean(curves[idx, :], dims=1))
    end
    return cents
end

# ─────────────────────────────────────────────────────────────────────────────
# 1. Metadata: curve_id → (gene, jw_id, medium)
# ─────────────────────────────────────────────────────────────────────────────

function load_metadata()
    path = joinpath(DATA_DIR, "Curves_knockouts_media.xlsx")
    @info "Reading metadata from $path"
    xf = XLSX.readxlsx(path)
    sh = xf[1]
    data = sh[:]

    meta = Dict{String, @NamedTuple{gene::String, jw_id::String, medium::String}}()
    for i in 2:size(data, 1)
        row = data[i, :]
        any(j -> ismissing(row[j]), 1:5) && continue
        curve_id  = _as_str(row[1])
        jw_id     = _as_str(row[2])
        gene_name = _as_str(row[3])
        # col 4 = gene_category (unused here)
        medium    = _as_str(row[5])
        isempty(curve_id) && continue
        meta[curve_id] = (; gene=gene_name, jw_id, medium)
    end
    @info "  $(length(meta)) curve-to-gene mappings loaded"
    return meta
end

# ─────────────────────────────────────────────────────────────────────────────
# 2. Growth curves for one medium file
# ─────────────────────────────────────────────────────────────────────────────

function load_curves(path::String, sheet_name::String, wanted::Set{String})
    @info "Reading curves from $path (sheet=$sheet_name, want=$(length(wanted)) curves)"
    xf   = XLSX.readxlsx(path)
    sh   = xf[sheet_name]
    data = sh[:]
    nrows, ncols = size(data)

    # Row 1: column headers  (Time, Curve00001, …)
    headers = [_as_str(data[1, j]) for j in 1:ncols]

    # Collect valid time rows
    times     = Float64[]
    row_index = Int[]
    for i in 2:nrows
        t = _as_float(data[i, 1])
        isfinite(t) || continue
        push!(times, t)
        push!(row_index, i)
    end

    # Extract only curves in `wanted`
    curves = Dict{String, Vector{Float64}}()
    for j in 2:ncols
        hdr = headers[j]
        hdr in wanted || continue
        vals = [_as_float(data[row_index[k], j]) for k in eachindex(row_index)]
        # Keep NaN for missing/empty cells — downstream handles them
        curves[hdr] = vals
    end

    @info "  $(length(times)) time points, $(length(curves)) curves extracted"
    return times, curves
end

# ─────────────────────────────────────────────────────────────────────────────
# 3. Aggregate replicates → mean ± std per (gene, medium)
# ─────────────────────────────────────────────────────────────────────────────

function aggregate_by_gene(meta, curves_dict::Dict{String,Vector{Float64}})
    # curves_dict: curve_id → OD vector (all same medium)
    groups = Dict{String, Vector{Vector{Float64}}}()   # gene → list of replicate vectors
    jw_map = Dict{String, String}()

    for (curve_id, info) in meta
        haskey(curves_dict, curve_id) || continue
        gene = info.gene
        if !haskey(groups, gene)
            groups[gene]  = Vector{Float64}[]
            jw_map[gene]  = info.jw_id
        end
        push!(groups[gene], curves_dict[curve_id])
    end

    result = Dict{String, @NamedTuple{mean::Vector{Float64}, std::Vector{Float64},
                                       n::Int, jw_id::String,
                                       replicates::Vector{Vector{Float64}}}}()
    for (gene, replicates) in groups
        mat = reduce(hcat, replicates)   # n_tp × n_replicates
        n_rep = size(mat, 2)

        # Per-replicate valid range: last timepoint where each replicate has finite data.
        # The gene mean is only valid up to the minimum last-valid index across all
        # replicates; beyond that, one or more replicates have dropped out and the mean
        # would be biased by whichever replicate happens to run longest.
        last_valid = n_rep > 1 ? minimum(
            let li = findlast(isfinite, mat[:, j])
                li === nothing ? 0 : li
            end
            for j in 1:n_rep
        ) : size(mat, 1)

        μ = map(1:size(mat, 1)) do i
            i > last_valid && return NaN
            vs = filter(!isnan, mat[i, :])
            isempty(vs) ? NaN : Statistics.mean(vs)
        end
        # SEM (std / sqrt(n)) shows uncertainty in the mean (more meaningful at n=3).
        # NaN beyond last_valid so interpolation respects the gene's true range.
        sem = map(1:size(mat, 1)) do i
            i > last_valid && return NaN
            vs = filter(!isnan, mat[i, :])
            isempty(vs) ? NaN :
            length(vs) > 1 ? Statistics.std(vs) / sqrt(length(vs)) : 0.0
        end
        result[gene] = (; mean=μ, std=sem, n=n_rep, jw_id=jw_map[gene],
                          replicates=[vec(mat[:, j]) for j in 1:n_rep])
    end
    return result
end

# ─────────────────────────────────────────────────────────────────────────────
# 4. KinBiont clustering with WCSS elbow sweep
# ─────────────────────────────────────────────────────────────────────────────

# Identify the non-growing strains with Kinbiont's own detector, run on the
# FULL-LENGTH replicate-averaged curves rather than on the shape-clustering grid.
#
# The two matrices serve different purposes and must not be conflated. The
# clustering grid is deliberately truncated at the 5th-percentile endpoint so a
# few short strains cannot cut the whole dataset; judging "does this strain
# grow?" on that window declares every late grower flat (ptsH, ybhM and yccE
# exceed OD 0.6, but only after the M63 grid ends at ~23 h). Conversely the
# full-length curves carry a long carried-forward tail, which flattens the
# z-scored shapes and collapses the LB k-means. So: growth/no-growth is decided
# on the whole time course, shape is clustered on the truncated grid.
function detect_non_growing(
    gene_means_full::Matrix{Float64},   # n_genes × n_tp, full length, finite
    times_full::Vector{Float64},
    gene_labels::Vector{String};
    tol_const::Float64 = 1.5,      # τ (Kinbiont default)
    q_low::Float64  = 0.05,        # lower pre-screen quantile (Kinbiont default)
    q_high::Float64 = 0.95,        # upper pre-screen quantile (Kinbiont default)
)::Vector{Bool}
    idx = detect_non_growing_indices(
        gene_means_full, times_full;
        prescreen_constant = true,
        prescreen_tol      = tol_const,
        prescreen_q_low    = q_low,
        prescreen_q_high   = q_high,
    )
    mask = falses(length(gene_labels))
    mask[idx] .= true
    return collect(mask)
end

# Curves whose signal range is negligible over the clustering grid. Same
# quantile-range test Kinbiont's pre-screen applies, evaluated here on the
# clustering matrix rather than on the full-length curves, so it catches
# "flat in this window" rather than "does not grow".
function _flat_on_grid(
    curves::Matrix{Float64};
    q_low::Float64 = 0.05,
    q_high::Float64 = 0.95,
    tol::Float64 = 0.01,
)::Vector{Bool}
    [quantile(curves[i, :], q_high) - quantile(curves[i, :], q_low) <= tol
     for i in axes(curves, 1)]
end

function cluster_gene_curves(
    gene_means::Matrix{Float64},   # n_genes × n_tp, shape-clustering grid
    times::Vector{Float64},
    gene_labels::Vector{String};
    k_range = 2:10,
    # Non-growing strains, decided beforehand on the full-length curves. They are
    # held out of the k-means and take the last cluster index, reproducing
    # Kinbiont's sentinel semantics: k-1 dynamic groups plus the sentinel.
    non_growing::Vector{Bool} = Bool[],
    force_k::Union{Nothing,Int} = nothing,  # override the auto-elbow selection
)
    n_genes = size(gene_means, 1)
    ng      = isempty(non_growing) ? falses(n_genes) : non_growing
    length(ng) == n_genes || throw(ArgumentError(
        "non_growing length $(length(ng)) != n_genes $n_genes"))

    prescreen = any(ng)
    grow_idx  = findall(!, ng)

    # A growing strain can still be flat *inside the clustering window* if it only
    # takes off later (ptsI and uidR are at baseline through 22.75 h and reach
    # OD ~0.27 by 50 h). Z-scoring such a curve rescales pure measurement noise
    # into an extreme shape: these sit 12-15 units from the centroid where the
    # median curve sits at 1.4, so k-means spends a whole cluster isolating two of
    # them and the dynamic structure collapses to 3786/2. They are excluded from
    # the FIT for that numerical reason only -- they are growers, not members of
    # the non-growing class -- and are assigned to their nearest centroid
    # afterwards so they still appear in a dynamic cluster.
    flat_in_window = _flat_on_grid(gene_means) .& .!ng
    fit_idx        = [i for i in grow_idx if !flat_in_window[i]]
    assign_idx     = [i for i in grow_idx if flat_in_window[i]]
    isempty(assign_idx) ||
        @info "  $(length(assign_idx)) growing strain(s) flat within the clustering " *
              "window held out of the fit, assigned to the nearest centroid"

    fit_gd = GrowthData(gene_means[fit_idx, :], times, gene_labels[fit_idx])

    # k in the reported sweep counts the sentinel, so the k-means over the
    # remaining dynamic curves is run with one cluster fewer.
    n_dynamic(k) = prescreen ? k - 1 : k

    @info "  Running WCSS sweep k=$(first(k_range))..$(last(k_range)) (prescreen=$prescreen)"
    ks        = collect(k_range)
    wcss_vals = Float64[]
    for k in ks
        proc = preprocess(fit_gd, FitOptions(cluster = true, n_clusters = n_dynamic(k)))
        push!(wcss_vals, something(proc.wcss, 0.0))
    end

    # Elbow (diagnostic aid). `force_k` overrides it when the analysis fixes k.
    auto_k = find_elbow(ks, wcss_vals)
    opt_k  = force_k === nothing ? auto_k : force_k
    @info "  Optimal k = $opt_k (auto-elbow suggested $auto_k)"

    # Final clustering, then splice the sentinel back in at index opt_k.
    proc_final = preprocess(fit_gd, FitOptions(cluster = true, n_clusters = n_dynamic(opt_k)))
    fit_labels = something(proc_final.clusters, ones(Int, length(fit_idx)))

    clusters = Vector{Int}(undef, n_genes)
    clusters[fit_idx] = fit_labels
    prescreen && (clusters[ng] .= opt_k)

    # Nearest-centroid assignment for the held-out late growers, in the same
    # z-scored space the k-means used.
    if !isempty(assign_idx)
        Z = Kinbiont._zscore_rows(gene_means)
        cents = [vec(mean(Z[fit_idx[fit_labels .== j], :], dims=1)) for j in 1:n_dynamic(opt_k)]
        for i in assign_idx
            clusters[i] = argmin([sum((Z[i, :] .- c) .^ 2) for c in cents])
        end
    end

    return (
        ks         = ks,
        wcss       = wcss_vals,
        optimal_k  = opt_k,
        prescreen  = prescreen,
        clusters   = clusters,
        # z-scored centroids (shape prototypes, scale-independent). Computed over
        # all curves, so the sentinel gets a centroid too; `proc_final.centroids`
        # only covers the dynamic curves the k-means actually saw. z-scoring uses
        # Kinbiont's own row normalization, including its constant-curve guard.
        centroids_z  = raw_centroids(Kinbiont._zscore_rows(gene_means), clusters, opt_k),
        # original-space centroids
        centroids_raw = raw_centroids(gene_means, clusters, opt_k),
    )
end

# ─────────────────────────────────────────────────────────────────────────────
# 5. Main
# ─────────────────────────────────────────────────────────────────────────────

function main()
    # Load metadata
    meta = load_metadata()

    # Determine which curve IDs we need per medium
    ids_lb  = Set(k for (k, v) in meta if v.medium == "LB")
    ids_m63 = Set(k for (k, v) in meta if v.medium == "M63")

    # Subset meta per medium (so aggregate_by_gene sees correct medium's curves)
    meta_lb  = Dict(k => v for (k, v) in meta if v.medium == "LB")
    meta_m63 = Dict(k => v for (k, v) in meta if v.medium == "M63")

    # Load raw curves
    times_lb, curves_lb = load_curves(
        joinpath(DATA_DIR, "Growth_curves_LB.xlsx"), "LB", ids_lb)
    times_m63, curves_m63 = load_curves(
        joinpath(DATA_DIR, "Growth_curves_M63.xlsx"), "M63", ids_m63)

    # Aggregate replicates (each medium uses its own full time vector)
    @info "Aggregating replicates by gene..."
    agg_lb  = aggregate_by_gene(meta_lb,  curves_lb)
    agg_m63 = aggregate_by_gene(meta_m63, curves_m63)

    # Union of genes present in both media
    genes_lb  = Set(keys(agg_lb))
    genes_m63 = Set(keys(agg_m63))
    all_genes = sort(collect(genes_lb ∪ genes_m63))
    @info "  $(length(all_genes)) unique genes (LB: $(length(genes_lb)), M63: $(length(genes_m63)))"

    # Build gene-level matrices per medium (only genes present in that medium)
    genes_lb_sorted  = sort(collect(genes_lb))
    genes_m63_sorted = sort(collect(genes_m63))

    mat_lb  = reduce(vcat, [agg_lb[g].mean'  for g in genes_lb_sorted])   # n_genes_lb  × n_tp_lb
    mat_m63 = reduce(vcat, [agg_m63[g].mean' for g in genes_m63_sorted])  # n_genes_m63 × n_tp_m63

    # ── Clustering ────────────────────────────────────────────────────────────
    # Shape is clustered on a robust common grid (5th-percentile t_end) so that
    # one gene with a very short replicate does not truncate the entire dataset.
    # Genes that don't reach the grid end are imputed with column mean (inside
    # _interp_to_cluster_grid).
    @info "Building clustering grid for LB..."
    times_lb_cl, mat_lb_cl   = _interp_to_cluster_grid(times_lb,  mat_lb)
    @info "  LB clustering grid: $(times_lb_cl[1])–$(times_lb_cl[end]) h, $(length(times_lb_cl)) points"

    @info "Building clustering grid for M63..."
    times_m63_cl, mat_m63_cl = _interp_to_cluster_grid(times_m63, mat_m63)
    @info "  M63 clustering grid: $(times_m63_cl[1])–$(times_m63_cl[end]) h, $(length(times_m63_cl)) points"

    # Growth/no-growth is decided separately, on the full-length replicate-
    # averaged curves from scripts/01_build_gene_means.py, so that strains which
    # only take off after the clustering grid ends are not called non-growing.
    @info "Pre-screening non-growing strains on the full-length curves..."
    times_lb_full,  mat_lb_full  = load_gene_mean_matrix("LB",  genes_lb_sorted)
    times_m63_full, mat_m63_full = load_gene_mean_matrix("M63", genes_m63_sorted)

    ng_lb  = detect_non_growing(mat_lb_full,  times_lb_full,  genes_lb_sorted)
    ng_m63 = detect_non_growing(mat_m63_full, times_m63_full, genes_m63_sorted)
    @info "  LB: $(count(ng_lb)) non-growing; M63: $(count(ng_m63)) non-growing (τ=1.5, q=0.05/0.95)"

    # Paper configuration: LB has no non-growing subpopulation, so no sentinel is
    # reserved and two dynamic clusters are used; M63 reserves the last cluster
    # index for the pre-screened strains, giving two dynamic clusters plus the
    # sentinel (k=3).
    @info "Clustering LB gene curves (no pre-screen, k=2)..."
    cl_lb  = cluster_gene_curves(mat_lb_cl,  times_lb_cl,  genes_lb_sorted;
                                 non_growing = ng_lb, force_k = 2)

    @info "Clustering M63 gene curves (pre-screen τ=1.5, k=3)..."
    cl_m63 = cluster_gene_curves(mat_m63_cl, times_m63_cl, genes_m63_sorted;
                                 non_growing = ng_m63, force_k = 3)

    # ── Visualisation ─────────────────────────────────────────────────────────
    # Per-gene curves are stored on the ORIGINAL time axis (downsampled 4x).
    # Each gene has NaN where its replicates had no data — the frontend draws
    # the curve only up to its individual valid range.
    lb_cluster_map  = Dict(zip(genes_lb_sorted,  cl_lb.clusters))
    m63_cluster_map = Dict(zip(genes_m63_sorted, cl_m63.clusters))

    # NaN → nothing so JSON3 writes null (Plotly draws a gap)
    to_json_vec(v) = Union{Float64,Nothing}[isnan(x) ? nothing : round(x; digits=6) for x in v]

    # Downsample original time axes 4x for JSON output
    ds = 4
    lb_ds_idx  = 1:ds:length(times_lb)
    m63_ds_idx = 1:ds:length(times_m63)
    times_lb_ds  = times_lb[lb_ds_idx]
    times_m63_ds = times_m63[m63_ds_idx]

    # Collect unique jw_ids (prefer LB, fallback to M63)
    jw_ids = Dict{String, String}()
    for (g, info) in agg_lb;  jw_ids[g] = info.jw_id; end
    for (g, info) in agg_m63; haskey(jw_ids, g) || (jw_ids[g] = info.jw_id); end

    gene_records = []
    for gene in all_genes
        rec = Dict{String, Any}(
            "gene"  => gene,
            "jw_id" => get(jw_ids, gene, ""),
        )
        if haskey(agg_lb, gene)
            rec["LB"] = Dict(
                "mean"         => to_json_vec(agg_lb[gene].mean[lb_ds_idx]),
                "std"          => to_json_vec(agg_lb[gene].std[lb_ds_idx]),
                "n_replicates" => agg_lb[gene].n,
                "cluster"      => lb_cluster_map[gene],
                "replicates"   => [to_json_vec(rep[lb_ds_idx]) for rep in agg_lb[gene].replicates],
            )
        end
        if haskey(agg_m63, gene)
            rec["M63"] = Dict(
                "mean"         => to_json_vec(agg_m63[gene].mean[m63_ds_idx]),
                "std"          => to_json_vec(agg_m63[gene].std[m63_ds_idx]),
                "n_replicates" => agg_m63[gene].n,
                "cluster"      => m63_cluster_map[gene],
                "replicates"   => [to_json_vec(rep[m63_ds_idx]) for rep in agg_m63[gene].replicates],
            )
        end
        push!(gene_records, rec)
    end

    # Assemble final JSON — separate time axes per medium since they may differ
    out = Dict(
        "metadata" => Dict(
            "n_genes"      => length(all_genes),
            "media"        => ["LB", "M63"],
            "source"       => "https://www.nature.com/articles/s41597-026-07075-9",
            "description"  => "E. coli Keio knockout collection growth curves (mean of replicates per gene)",
        ),
        "times_LB"  => round.(times_lb_ds;  digits=4),
        "times_M63" => round.(times_m63_ds; digits=4),
        # Keep legacy "times" key as the shorter of the two for backward compat
        "times"     => round.(length(times_lb_ds) <= length(times_m63_ds) ? times_lb_ds : times_m63_ds; digits=4),
        "wcss_sweep" => Dict(
            "LB"  => Dict("ks" => cl_lb.ks,  "wcss" => round.(cl_lb.wcss;  digits=4)),
            "M63" => Dict("ks" => cl_m63.ks, "wcss" => round.(cl_m63.wcss; digits=4)),
        ),
        "optimal_k" => Dict(
            "LB"  => cl_lb.optimal_k,
            "M63" => cl_m63.optimal_k,
        ),
        # Centroids are on their own clustering-grid time axes
        "centroid_times" => Dict(
            "LB"  => round.(times_lb_cl;  digits=4),
            "M63" => round.(times_m63_cl; digits=4),
        ),
        "centroids" => Dict(
            "LB"  => [to_json_vec(cl_lb.centroids_raw[k, :])  for k in 1:cl_lb.optimal_k],
            "M63" => [to_json_vec(cl_m63.centroids_raw[k, :]) for k in 1:cl_m63.optimal_k],
        ),
        "centroids_z" => Dict(
            "LB"  => [to_json_vec(cl_lb.centroids_z[k, :])  for k in 1:size(cl_lb.centroids_z, 1)],
            "M63" => [to_json_vec(cl_m63.centroids_z[k, :]) for k in 1:size(cl_m63.centroids_z, 1)],
        ),
        "genes" => gene_records,
        # With cluster_prescreen_constant=true, KinBiont assigns non-growing wells
        # to label n_clusters (the last index). When the pre-screen is disabled
        # (LB), no sentinel is reserved: report cluster 0 (matches no cluster) and
        # an empty gene list so the figure draws no non-growing cluster.
        "nongrowing_cluster" => Dict(
            "LB"  => cl_lb.prescreen  ? cl_lb.optimal_k  : 0,
            "M63" => cl_m63.prescreen ? cl_m63.optimal_k : 0,
        ),
        "nongrowing_genes" => Dict(
            "LB"  => cl_lb.prescreen  ? sort([g for g in genes_lb_sorted  if lb_cluster_map[g]  == cl_lb.optimal_k])  : String[],
            "M63" => cl_m63.prescreen ? sort([g for g in genes_m63_sorted if m63_cluster_map[g] == cl_m63.optimal_k]) : String[],
        ),
    )

    # Centroid tables for the paper's Figure 2c generator
    # (GUIbiontPaper/scripts/plot_keio_cluster_centroids_raw_svg_v2.py). Written
    # from the same run as the JSON so the panel can never drift from the
    # clustering it is supposed to depict.
    write_centroid_csv("lb",  times_lb_cl,  mat_lb_cl,  cl_lb)
    write_centroid_csv("m63", times_m63_cl, mat_m63_cl, cl_m63)

    mkpath(dirname(OUT_PATH))
    open(OUT_PATH, "w") do io
        JSON3.write(io, out)
    end
    @info "Output written to $OUT_PATH"
end

main()
