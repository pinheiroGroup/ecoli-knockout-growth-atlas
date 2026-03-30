#!/usr/bin/env julia
# batch_fit.jl — Batch-fit all gene-level mean curves with AICc model selection
#
# Reads docs/data/curves_data.json (produced by analyse.jl) and fits each
# gene × medium curve with four models: logistic, gompertz, baranyi_richards,
# aHPM. Best model is selected by AICc.
#
# Checkpointing: each row is appended immediately after fitting. On restart the
# script skips already-fitted (gene, medium) pairs.
#
# Run (multi-threaded):
#   julia --threads auto --project=. analysis/batch_fit.jl
#
# Output: results/keio_batch_fit_results.csv
# Columns: gene, jw_id, medium, cluster, n_replicates,
#           best_model, gr, N_max, lag, shape, aicc, converged

using Pkg
Pkg.activate(@__DIR__)

using JSON3
using Kinbiont
using CSV
using DataFrames

const REPO_DIR   = joinpath(@__DIR__, "..")
const DATA_PATH  = joinpath(REPO_DIR, "docs", "data", "curves_data.json")
const RESULTS_DIR = joinpath(REPO_DIR, "results")
const OUT_PATH   = joinpath(RESULTS_DIR, "keio_batch_fit_results.csv")

const CSV_HEADER = [:gene, :jw_id, :medium, :cluster, :n_replicates,
                    :best_model, :gr, :N_max, :lag, :shape, :aicc, :converged]

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

# logistic / gompertz: 2 params [gr, N_max]
# baranyi_richards:    4 params [gr, N_max, lag_time, shape]
# aHPM:                4 params [gr, exit_lag_rate, N_max, shape]

const MODELS = [
    MODEL_REGISTRY["logistic"],
    MODEL_REGISTRY["gompertz"],
    MODEL_REGISTRY["baranyi_richards"],
    MODEL_REGISTRY["aHPM"],
]
const PARAMS = [
    fill(1.0, 2),   # logistic
    fill(1.0, 2),   # gompertz
    fill(1.0, 4),   # baranyi_richards
    fill(1.0, 4),   # aHPM
]
const SPEC = ModelSpec(
    MODELS, PARAMS;
    lower = [fill(0.0, 2), fill(0.0, 2), fill(0.0, 4), fill(0.0, 4)],
    upper = [fill(50.0, 2), fill(50.0, 2), fill(50.0, 4), fill(50.0, 4)],
)

const OPTS = FitOptions(
    smooth                          = true,
    smooth_method                   = :rolling_avg,
    smooth_pt_avg                   = 5,    # fewer points than chemical media (50 vs 97)
    cut_stationary_phase            = true,
    stationary_percentile_thr       = 0.05,
    stationary_pt_smooth_derivative = 5,
    stationary_win_size             = 3,
    loss                            = "RE",
)

# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

function load_done(path::String)::Set{Tuple{String,String}}
    isfile(path) || return Set{Tuple{String,String}}()
    df = CSV.read(path, DataFrame; select=[:gene, :medium])
    return Set{Tuple{String,String}}(zip(df.gene, df.medium))
end

function open_checkpoint(path::String)::IO
    new_file = !isfile(path)
    io = open(path, "a")
    new_file && println(io, join(CSV_HEADER, ","))
    return io
end

function append_row!(io::IO, lk::ReentrantLock, row::NamedTuple)
    vals = [getfield(row, c) for c in CSV_HEADER]
    line = join(vals, ",")
    lock(lk) do
        println(io, line)
        flush(io)
    end
end

# ---------------------------------------------------------------------------
# Extract params from fit result (models differ in param layout)
# ---------------------------------------------------------------------------

function extract_params(model_name::String, params::Vector{Float64})
    # Returns (gr, N_max, lag, shape) — lag/shape are NaN when model has no such param
    if model_name in ("logistic", "gompertz")
        length(params) >= 2 || return (NaN, NaN, NaN, NaN)
        return (params[1], params[2], NaN, NaN)
    elseif model_name == "baranyi_richards"
        length(params) >= 4 || return (NaN, NaN, NaN, NaN)
        return (params[1], params[2], params[3], params[4])
    elseif model_name == "aHPM"
        # aHPM params: gr, exit_lag_rate, N_max, shape
        length(params) >= 4 || return (NaN, NaN, NaN, NaN)
        return (params[1], params[3], params[2], params[4])
    else
        return (NaN, NaN, NaN, NaN)
    end
end

# ---------------------------------------------------------------------------
# Fit one (gene, medium) curve
# ---------------------------------------------------------------------------

function fit_curve(gene::String, jw_id::String, medium::String,
                   cluster::Int, n_reps::Int,
                   times::Vector{Float64}, mean_od::Vector{Float64})::NamedTuple
    fail = (gene=gene, jw_id=jw_id, medium=medium,
             cluster=cluster, n_replicates=n_reps,
             best_model="", gr=NaN, N_max=NaN, lag=NaN, shape=NaN,
             aicc=NaN, converged=false)

    # Filter out NaN (null → NaN) and require at least 5 points
    mask  = .!isnan.(mean_od)
    sum(mask) < 5 && return fail

    t_valid  = times[mask]
    od_valid = mean_od[mask]

    try
        gd  = GrowthData(reshape(od_valid, 1, length(od_valid)), t_valid, [gene])
        res = kinbiont_fit(gd, SPEC, OPTS)
        r   = res[1]
        p   = Float64.(r.best_params)
        gr, N_max, lag, shape = extract_params(r.best_model.name, p)
        return (gene=gene, jw_id=jw_id, medium=medium,
                cluster=cluster, n_replicates=n_reps,
                best_model=r.best_model.name,
                gr=gr, N_max=N_max, lag=lag, shape=shape,
                aicc=Float64(r.best_aic), converged=true)
    catch
        return fail
    end
end

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

function main()
    mkpath(RESULTS_DIR)

    @info "Loading $DATA_PATH"
    d = open(DATA_PATH) do io; JSON3.read(io); end

    times = Float64.(d["times"])

    # Build todo list: (gene, medium) pairs not yet done
    done = load_done(OUT_PATH)
    !isempty(done) && @info "Resuming: $(length(done)) curves already fitted"

    todo = NamedTuple[]
    for g in d["genes"]
        gene  = String(g["gene"])
        jw_id = String(g["jw_id"])
        for med in ("LB", "M63")
            haskey(g, med) || continue
            (gene, med) in done && continue
            od_raw = g[med]["mean"]
            od = [x === nothing ? NaN : Float64(x) for x in od_raw]
            push!(todo, (gene=gene, jw_id=jw_id, medium=med,
                          cluster=Int(g[med]["cluster"]),
                          n_replicates=Int(g[med]["n_replicates"]),
                          od=od))
        end
    end

    n_todo = length(todo)
    @info "Fitting $n_todo gene × medium curves ($(Threads.nthreads()) thread(s))…"
    n_todo == 0 && return

    io = open_checkpoint(OUT_PATH)
    lk = ReentrantLock()
    done_count = Threads.Atomic{Int}(0)

    try
        Threads.@threads for i in 1:n_todo
            t = todo[i]
            row = fit_curve(t.gene, t.jw_id, t.medium, t.cluster, t.n_replicates,
                            times, t.od)
            append_row!(io, lk, row)
            c = Threads.atomic_add!(done_count, 1) + 1
            if c % 200 == 0 || c == n_todo
                print("\r  $(c)/$(n_todo) fitted…")
                flush(stdout)
            end
        end
    finally
        close(io)
    end
    println()

    # Summary
    results = CSV.read(OUT_PATH, DataFrame)
    conv    = filter(r -> r.converged, results)
    pct     = round(100 * nrow(conv) / nrow(results); digits=1)
    @info "Total: $(nrow(results)) fits | converged: $(nrow(conv)) ($(pct)%)"
    @info "Output: $OUT_PATH"

    for med in ("LB", "M63")
        sub = filter(r -> r.medium == med && r.converged, conv)
        isempty(sub) && continue
        @info "  $med — $(nrow(sub)) converged curves"
        @info "    model counts: $(sort(collect(countmap(sub.best_model)), by=x->-x[2]))"
    end
end

function countmap(v)
    d = Dict{eltype(v), Int}()
    for x in v; d[x] = get(d, x, 0) + 1; end
    d
end

main()
