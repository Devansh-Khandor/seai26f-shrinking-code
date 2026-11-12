#!/usr/bin/env lua
-- skill_09.lua
-- Run: lua skill_09.lua <csv_path>
--
-- Provides two NASA requests:
--   NASA-17 – Flag historical projects whose combinations of project characteristics
--             are unusually different from the rest of our records.
--   NASA-18 – Identify project records whose reported effort or defect counts
--             differ substantially from what comparable projects would lead us to
--             expect, so we can investigate possible measurement or recording issues.

local ezr = require "ezr"

-- Helpers from ezr
local csv   = ezr.csv
local Tbl   = ezr.Tbl
local push  = ezr.push
local sorted = ezr.sorted
local cap   = math.floor
local sqrt  = math.sqrt

-- ------------------------------------------------------------------
-- Read data
-- ------------------------------------------------------------------
local path = arg[1]
if not path then
  io.stderr:write("Usage: lua skill_09.lua <csv_path>\n")
  os.exit(1)
end

local tbl = Tbl(csv(path))

-- ------------------------------------------------------------------
-- Utility helpers
-- ------------------------------------------------------------------
local function row_dist_x(row1, row2)
  return tbl:distx(row1, row2)
end

-- Median of an array
local function median(arr)
  local n = #arr
  if n == 0 then return nil end
  local s = sorted(arr)
  if n % 2 == 1 then
    return s[(n + 1) / 2]
  else
    return (s[n / 2] + s[n / 2 + 1]) / 2
  end
end

-- Compute distance from a row to the centroid (mid values) of all x cols
local function dist_to_centroid(row, tbl)
  local d = 0
  for _, c in ipairs(tbl.cols.x) do
    d = d + c:dist(row[c.at], c:mid())
  end
  return d
end

-- Find k nearest neighbours (by distx) to a row
local function k_neighbors(row, tbl, k)
  local all = {}
  for _, r in ipairs(tbl.rows) do
    if r ~= row then
      push(all, {row = r, dist = tbl:distx(row, r)})
    end
  end
  local sorted_all = sorted(all, function(a, b) return a.dist < b.dist end)
  local neigh = {}
  for i = 1, math.min(k, #sorted_all) do
    push(neigh, sorted_all[i].row)
  end
  return neigh
end

-- Compute mean of a numeric column over a set of rows
local function col_mean(rows, at)
  local sum, n = 0, 0
  for _, r in ipairs(rows) do
    local v = r[at]
    if v ~= "?" then
      sum = sum + v
      n = n + 1
    end
  end
  return n > 0 and sum / n or nil
end

-- ------------------------------------------------------------------
-- NASA-17: flag projects with unusual characteristic combinations
-- ------------------------------------------------------------------
print("== NASA-17 ==")

-- Compute distances to overall centroid
local dist_rows = {}
for _, r in ipairs(tbl.rows) do
  push(dist_rows, {row = r, dist = dist_to_centroid(r, tbl)})
end
-- Sort descending: most unusual first
local sorted_dist = sorted(dist_rows, function(a, b) return a.dist > b.dist end)

-- Take top 5 unusual projects (or fewer if data <5)
local top_n = math.min(5, #sorted_dist)
print(string.format("Top %d projects with most atypical feature combinations:", top_n))
for i = 1, top_n do
  local r = sorted_dist[i].row
  print(string.format(
    "%d) ID=%s  Distance=%.4f",
    i, r[1], sorted_dist[i].dist))
end
print()

-- ------------------------------------------------------------------
-- NASA-18: flag projects with anomalous effort or defect counts
-- ------------------------------------------------------------------
print("== NASA-18 ==")

local effort_at = 27   -- Effort-
local defect_at = 28   -- Defects-

-- Compute overall stats for effort and defect
local effort_stats = {mean = tbl.cols.all[effort_at]:mid(),
                      std  = tbl.cols.all[effort_at]:div()}
local defect_stats = {mean = tbl.cols.all[defect_at]:mid(),
                      std  = tbl.cols.all[defect_at]:div()}

-- Thresholds: 2 standard deviations
local eff_thresh = 2 * effort_stats.std
local def_thresh = 2 * defect_stats.std

-- Analyze each row
local effort_anoms = {}
local defect_anoms = {}

for _, r in ipairs(tbl.rows) do
  -- Skip rows with missing target
  if r[effort_at] ~= "?" and r[defect_at] ~= "?" then
    -- Nearest neighbours for prediction
    local neigh = k_neighbors(r, tbl, 3)
    local pred_eff = col_mean(neigh, effort_at)
    local pred_def = col_mean(neigh, defect_at)

    local diff_eff = r[effort_at] - pred_eff
    local diff_def = r[defect_at] - pred_def

    if math.abs(diff_eff) > eff_thresh then
      push(effort_anoms, {row = r, diff = diff_eff})
    end
    if math.abs(diff_def) > def_thresh then
      push(defect_anoms, {row = r, diff = diff_def})
    end
  end
end

-- Sort by magnitude of anomaly
effort_anoms = sorted(effort_anoms, function(a,b) return math.abs(a.diff) > math.abs(b.diff) end)
defect_anoms = sorted(defect_anoms, function(a,b) return math.abs(a.diff) > math.abs(b.diff) end)

print(string.format("Projects with effort unusually higher or lower than comparable peers (|Δ| > 2σ, σ=%.2f):", effort_stats.std))
local top_eff = math.min(5, #effort_anoms)
if top_eff == 0 then
  print("  None found.")
else
  for i = 1, top_eff do
    local r = effort_anoms[i].row
    print(string.format(
      "%d) ID=%s  Effort=%.1f  Δ=%.1f",
      i, r[1], r[effort_at], effort_anoms[i].diff))
  end
end
print()

print(string.format("Projects with defect count unusually higher or lower than comparable peers (|Δ| > 2σ, σ=%.2f):", defect_stats.std))
local top_def = math.min(5, #defect_anoms)
if top_def == 0 then
  print("  None found.")
else
  for i = 1, top_def do
    local r = defect_anoms[i].row
    print(string.format(
      "%d) ID=%s  Defects=%d  Δ=%d",
      i, r[1], r[defect_at], defect_anoms[i].diff))
  end
end
