#!/usr/bin/env lua
-- Skill 09: Outlier detection and unusual configuration spotting
-- Reuses the ezr library and helper functions from previous skills.

local ezr = require "ezr"

-- ------------------------------------------------------------------
-- Pretty printer
-- ------------------------------------------------------------------
local function pretty(v) return ezr.show(v) end

-- ------------------------------------------------------------------
-- Build a tree and predict performance for all rows
-- ------------------------------------------------------------------
local function build_predictions(tbl)
  local tree = ezr.Tree(tbl, tbl.rows)
  local preds = {}
  for i, r in ipairs(tbl.rows) do
    local pred = tree:leaf(tbl, r)
    preds[i] = pred
  end
  return preds
end

-- ------------------------------------------------------------------
-- Residual (actual – predicted) statistics
-- ------------------------------------------------------------------
local function residual_stats(tbl, preds)
  local perf_at = tbl.cols.y[1].at
  local n = #tbl.rows
  local sum, sqsum = 0, 0
  local residuals = {}
  for i, r in ipairs(tbl.rows) do
    local actual = r[perf_at]
    local res = actual - preds[i]
    residuals[i] = {row=r, actual=actual, pred=preds[i], res=res}
    sum = sum + res
    sqsum = sqsum + res * res
  end
  local mean = sum / n
  local sd = math.sqrt(math.max(sqsum - sum*sum/n, 0) / n)
  return {residuals=residuals, mean=mean, sd=sd}
end

-- ------------------------------------------------------------------
-- Flag residual outliers (large absolute deviation from expectation)
-- ------------------------------------------------------------------
local function flag_residual_outliers(stats)
  local outliers = {}
  local mean, sd = stats.mean, stats.sd
  local threshold = mean + 2 * sd
  for _, info in ipairs(stats.residuals) do
    if math.abs(info.res) > threshold then
      local z = sd > 0 and (info.res - mean) / sd or 0
      table.insert(outliers, {row=info.row,
                              actual=info.actual,
                              pred=info.pred,
                              res=info.res,
                              z=z})
    end
  end
  return outliers, threshold, mean, sd
end

-- ------------------------------------------------------------------
-- Nearest‑neighbour distance statistics (over x columns)
-- ------------------------------------------------------------------
local function distance_stats(tbl)
  local n = #tbl.rows
  local dists = {}
  for i, r1 in ipairs(tbl.rows) do
    local mind = 1e9
    for j, r2 in ipairs(tbl.rows) do
      if i ~= j then
        local d = tbl:distx(r1, r2)
        if d < mind then mind = d end
      end
    end
    dists[i] = mind
  end
  local sum, sqsum = 0, 0
  for _, d in ipairs(dists) do
    sum = sum + d
    sqsum = sqsum + d*d
  end
  local mean = sum / n
  local sd = math.sqrt(math.max(sqsum - sum*sum/n, 0) / n)
  return dists, mean, sd
end

-- ------------------------------------------------------------------
-- Flag distance outliers (configurations far from all others)
-- ------------------------------------------------------------------
local function flag_distance_outliers(tbl, dists, mean, sd)
  local outliers = {}
  local threshold = mean + 2 * sd
  for i, d in ipairs(dists) do
    if d > threshold then
      local z = sd > 0 and (d - mean) / sd or 0
      table.insert(outliers, {row=tbl.rows[i], dist=d, z=z})
    end
  end
  return outliers, threshold, mean, sd
end

-- ------------------------------------------------------------------
-- Main routine
-- ------------------------------------------------------------------
local function main()
  local csv_path = arg[1]
  assert(csv_path, "Usage: lua skill_09.lua <csv path>")

  local tbl = ezr.Tbl(ezr.csv(csv_path))

  -- ----------------------------------------------------------------
  -- APACHE-17 : Residual outliers
  -- ----------------------------------------------------------------
  print("== APACHE-17 ==")
  local preds = build_predictions(tbl)
  local res_stats = residual_stats(tbl, preds)
  local outliers, thr, m, sd = flag_residual_outliers(res_stats)

  print(string.format("Residual mean: %s, sd: %s, outlier threshold: %s",
        pretty(m), pretty(sd), pretty(thr)))
  if #outliers == 0 then
    print("  No significant residual outliers detected.")
  else
    for _, o in ipairs(outliers) do
      local idx = table.find(tbl.rows, o.row)
      print(string.format("  Row %d: actual=%s, predicted=%s, residual=%s, z=%.2f",
            idx, pretty(o.actual), pretty(o.pred), pretty(o.res), o.z))
    end
  end

  -- ----------------------------------------------------------------
  -- APACHE-18 : Distance outliers
  -- ----------------------------------------------------------------
  print("\n== APACHE-18 ==")
  local dists, m_dist, sd_dist = distance_stats(tbl)
  local dist_out, thr_dist, m_d, sd_d = flag_distance_outliers(tbl, dists, m_dist, sd_dist)

  print(string.format("Distance mean: %s, sd: %s, outlier threshold: %s",
        pretty(m_d), pretty(sd_d), pretty(thr_dist)))
  if #dist_out == 0 then
    print("  No configurations with unusually large setting distance.")
  else
    for _, o in ipairs(dist_out) do
      local idx = table.find(tbl.rows, o.row)
      print(string.format("  Row %d: distance=%s, z=%.2f",
            idx, pretty(o.dist), o.z))
    end
  end
end

-- Execute if run directly
if not debug.getinfo(1, "S").short_src:match("[/\\]skill_09.lua$") then
  main()
end
