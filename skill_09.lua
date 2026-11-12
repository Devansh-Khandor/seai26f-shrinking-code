#!/usr/bin/env lua
-- Skill 09 – Apache configuration anomaly detection & outlier settings
--
-- Usage:  lua skill_09.lua <csv path>
--
-- Requires the ezr library (ezr.lua, ezr-lib.lua) in the same directory.
--
-- APACHE-17: Detect rows whose recorded performance deviates unusually
--            from the value predicted by a decision tree built on all data.
-- APACHE-18: Flag rows that are very distant in configuration space from
--            their nearest neighbours, even if their performance is not poor.
--
local ezr = require("ezr")

-- ------------------------------------------------------------------
-- Helper: get row index in the original table
local function row_index(tbl, row)
  for i, r in ipairs(tbl.rows) do
    if r == row then return i end
  end
  return nil
end

-- ------------------------------------------------------------------
-- APACHE-17: performance anomaly detection
local function apache_17(tbl)
  print("== APACHE-17 ==")

  local perf_at = tbl.cols.y[1].at
  local tree = ezr.Tree(tbl, tbl.rows)

  -- Predict and compute residuals
  local residuals = {}
  for _, r in ipairs(tbl.rows) do
    local pred = tree:leaf(tbl, r)
    local actual = r[perf_at]
    local resid = actual - pred
    table.insert(residuals, {row=r, actual=actual, pred=pred, resid=resid})
  end

  -- Compute mean & std of residuals
  local sum, sumsq = 0, 0
  for _, d in ipairs(residuals) do
    sum = sum + d.resid
    sumsq = sumsq + d.resid * d.resid
  end
  local n = #residuals
  local mean = sum / n
  local var = n > 1 and (sumsq - n * mean * mean) / (n - 1) or 0
  local sd = math.sqrt(math.max(var, 0))

  -- Flag rows with |residual| > 2*sd
  local flagged = {}
  for _, d in ipairs(residuals) do
    if math.abs(d.resid) > 2 * sd then
      table.insert(flagged, d)
    end
  end

  -- If none, say nothing
  if #flagged == 0 then
    print("No performance anomalies detected.")
    return
  end

  -- Sort by absolute residual descending and show top 3
  table.sort(flagged, function(a,b)
    return math.abs(a.resid) > math.abs(b.resid) end)

  local to_show = math.min(3, #flagged)
  for i = 1, to_show do
    local d = flagged[i]
    local idx = row_index(tbl, d.row)
    print(string.format(
      "%d. Row %d: Actual=%.1f, Pred=%.1f, Residual=%.1f",
      i, idx, d.actual, d.pred, d.resid))
  end
  if to_show < #flagged then
    print(string.format("...and %d more anomalies.", #flagged - to_show))
  end
end

-- ------------------------------------------------------------------
-- APACHE-18: configuration-space outlier detection
local function apache_18(tbl)
  print("\n== APACHE-18 ==")

  local perf_at = tbl.cols.y[1].at

  -- Compute performance median to filter out obviously poor rows
  local perf_vals = ezr.map(tbl.rows, function(r) return r[perf_at] end)
  local sorted_perf = ezr.sorted(perf_vals)
  local median_perf = sorted_perf[math.floor(#sorted_perf / 2) + 1]

  -- Precompute pairwise distances for efficiency
  local rows = tbl.rows
  local n_rows = #rows
  local min_dists = {}
  for i = 1, n_rows do
    local r1 = rows[i]
    local best = math.huge
    for j = 1, n_rows do
      if i ~= j then
        local d = tbl:distx(r1, rows[j])
        if d < best then best = d end
      end
    end
    min_dists[i] = best
  end

  -- Median of minimum distances
  local sorted_dists = ezr.sorted(min_dists)
  local median_dists = sorted_dists[math.floor(#sorted_dists / 2) + 1]

  -- Flag rows that: (1) have performance better than median_perf
  --                 (2) min distance > 1.5 * median_dists
  local threshold = 1.5 * median_dists
  local flagged = {}
  for i, r in ipairs(rows) do
    if r[perf_at] > median_perf and min_dists[i] > threshold then
      table.insert(flagged, {row=r, dist=min_dists[i]})
    end
  end

  if #flagged == 0 then
    print("No configuration-space outliers detected.")
    return
  end

  -- Sort by distance descending, show top 3
  table.sort(flagged, function(a,b) return a.dist > b.dist end)
  local to_show = math.min(3, #flagged)
  for i = 1, to_show do
    local d = flagged[i]
    local idx = row_index(tbl, d.row)
    print(string.format(
      "%d. Row %d: MinDist=%.2f, Performance=%.1f",
      i, idx, d.dist, d.row[perf_at]))
  end
  if to_show < #flagged then
    print(string.format("...and %d more configuration outliers.", #flagged - to_show))
  end
end

-- ------------------------------------------------------------------
-- Main routine
local function main(csv_path)
  if not csv_path then
    print("Usage: lua skill_09.lua <csv path>")
    os.exit(1)
  end

  local tbl = ezr.Tbl(ezr.csv(csv_path))

  apache_17(tbl)
  apache_18(tbl)
end

main(arg[1])
