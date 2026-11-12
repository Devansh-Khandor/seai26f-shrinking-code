#!/usr/bin/env lua
--  skill_09.lua – anomaly detection (CAR‑17 & CAR‑18)
--  Usage: lua skill_09.lua <csv_path>
--
--  CAR‑17: Find vehicles whose independent characteristics
--          are unusually far from the inventory centroid.
--          These rows are flagged as potential outliers.
--
--  CAR‑18: Identify vehicles whose fuel‑efficiency or weight
--          differs substantially from the prediction based on
--          their nearest neighbours.  Such rows are flagged
--          for possible data entry errors.
--
--  The script uses only the standard Lua libraries and the
--  ezr framework shipped with this project.

local ezr = require("ezr")

-- ------------------------------------------------------------------
-- Helper functions
-- ------------------------------------------------------------------
local function mean_sd(values)
  local n = #values
  if n == 0 then return 0, 0 end
  local sum = 0
  for _, v in ipairs(values) do sum = sum + v end
  local mu = sum / n
  local m2 = 0
  for _, v in ipairs(values) do
    local d = v - mu
    m2 = m2 + d * d
  end
  local sd = n > 1 and math.sqrt(m2 / (n - 1)) or 0
  return mu, sd
end

local function nearest_neighbors(tbl, row, k)
  local dists = {}
  for _, r in ipairs(tbl.rows) do
    if r ~= row then
      table.insert(dists, {row = r, d = tbl:distx(row, r)})
    end
  end
  table.sort(dists, function(a, b) return a.d < b.d end)
  local neighbors = {}
  for i = 1, math.min(k, #dists) do
    neighbors[i] = dists[i].row
  end
  return neighbors
end

local function predict_value(neigh, col, k)
  local values = {}
  for _, r in ipairs(neigh) do
    local v = r[col.at]
    if v ~= "?" then
      table.insert(values, v)
    end
  end
  local mu, sd = mean_sd(values)
  return mu, sd
end

-- ------------------------------------------------------------------
-- Load the inventory table
-- ------------------------------------------------------------------
local csv_path = arg[1]
if not csv_path then
  io.stderr:write("usage: lua skill_09.lua <csv_path>\n")
  os.exit(1)
end

local tbl = ezr.Tbl(ezr.csv(csv_path))
local rows = tbl.rows
local cols = tbl.cols
local Lbs_col = cols.all[5]   -- "Lbs-"
local Mpg_col = cols.all[8]   -- "Mpg+"

-- ------------------------------------------------------------------
-- CAR‑17: Unusually distant combinations
-- ------------------------------------------------------------------
print("== CAR-17 ==")

-- Synthetic centroid row (mid values)
local centroid = tbl:mids()

-- Compute distances to centroid
local dists = {}
for idx, r in ipairs(rows) do
  local d = tbl:distx(r, centroid)
  table.insert(dists, {idx = idx, row = r, d = d})
end

-- Statistics of distances
local dist_values = ezr.map(dists, "d")
local mean_d, sd_d = mean_sd(dist_values)

-- Threshold: mean + 2*sd (adjustable)
local thr_d = mean_d + 2 * sd_d
if sd_d == 0 then thr_d = mean_d end

-- Find outliers
local outliers = {}
for _, info in ipairs(dists) do
  if info.d > thr_d then
    table.insert(outliers, info)
  end
end

if #outliers == 0 then
  print("  None found.")
else
  print(string.format("  %d outlier(s) (distance > %.3f)", #outliers, thr_d))
  for _, o in ipairs(outliers) do
    print(string.format("    #%-3d  dist=%.3f  row=%s",
      o.idx, o.d, ezr.show(o.row)))
  end
end

-- ------------------------------------------------------------------
-- CAR‑18: Weight or MPG deviates from nearest‑neighbour prediction
-- ------------------------------------------------------------------
print("\n== CAR-18 ==")

local k = 3  -- number of neighbours to use

-- Containers for flagged rows
local weight_anoms = {}
local mpg_anoms   = {}

for idx, r in ipairs(rows) do
  -- Skip rows with missing values
  local actual_w = r[Lbs_col.at]
  local actual_m = r[Mpg_col.at]
  if actual_w ~= "?" and actual_m ~= "?" then
    local neigh = nearest_neighbors(tbl, r, k)

    -- Weight prediction
    local pred_w, sd_w = predict_value(neigh, Lbs_col, k)
    local diff_w = math.abs(actual_w - pred_w)
    local thr_w = sd_w * 1.5
    if sd_w == 0 then thr_w = 0 end
    if diff_w > thr_w then
      table.insert(weight_anoms, {
        idx = idx, row = r,
        actual = actual_w,
        pred   = pred_w,
        diff   = diff_w,
        thr    = thr_w,
      })
    end

    -- MPG prediction
    local pred_m, sd_m = predict_value(neigh, Mpg_col, k)
    local diff_m = math.abs(actual_m - pred_m)
    local thr_m = sd_m * 1.5
    if sd_m == 0 then thr_m = 0 end
    if diff_m > thr_m then
      table.insert(mpg_anoms, {
        idx = idx, row = r,
        actual = actual_m,
        pred   = pred_m,
        diff   = diff_m,
        thr    = thr_m,
      })
    end
  end
end

-- Report weight anomalies
print("\n  Weight anomalies:")
if #weight_anoms == 0 then
  print("    None found.")
else
  for _, a in ipairs(weight_anoms) do
    print(string.format(
      "    #%-3d  actual=%.2f  pred=%.2f  diff=%.2f  thr=%.2f  row=%s",
      a.idx, a.actual, a.pred, a.diff, a.thr, ezr.show(a.row)))
  end
end

-- Report MPG anomalies
print("\n  MPG anomalies:")
if #mpg_anoms == 0 then
  print("    None found.")
else
  for _, a in ipairs(mpg_anoms) do
    print(string.format(
      "    #%-3d  actual=%.2f  pred=%.2f  diff=%.2f  thr=%.2f  row=%s",
      a.idx, a.actual, a.pred, a.diff, a.thr, ezr.show(a.row)))
  end
end
