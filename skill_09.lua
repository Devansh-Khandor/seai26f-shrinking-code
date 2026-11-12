#!/usr/bin/env lua
-- Skill 09 – Apache configuration anomaly detection
-- Requires the ezr library (ezr.lua & ezr-lib.lua in the same dir)

local ezr = require"ezr"

------------------------------------------------------------------
-- Get the CSV file path from the command line
------------------------------------------------------------------
local csv_path = arg[1]
assert(csv_path, "Usage: lua skill_09.lua <csv path>")

------------------------------------------------------------------
-- Load the data into an ezr table
------------------------------------------------------------------
local tbl = ezr.Tbl(ezr.csv(csv_path))

------------------------------------------------------------------
-- Helpers ----------------------------------------------------------
------------------------------------------------------------------
local perf_at = tbl.cols.klass.at

-- Pretty print a row as "col=value" pairs
local function row_to_str(r)
  local parts = {}
  for _, col in ipairs(tbl.cols.x) do
    parts[#parts+1] = col.name .. "=" .. tostring(r[col.at])
  end
  parts[#parts+1] = tbl.cols.klass.name .. "=" .. tostring(r[perf_at])
  return table.concat(parts, ", ")
end

-- Distance in configuration space (independent columns)
local function cfg_distance(r1, r2)
  return tbl:distx(r1, r2)
end

-- Return the k nearest rows to a reference row
local function nearest_rows(ref, rows, k)
  local t = {}
  for _, r in ipairs(rows) do
    if r ~= ref then
      table.insert(t, {row=r, dist=cfg_distance(ref, r)})
    end
  end
  table.sort(t, function(a,b) return a.dist < b.dist end)
  local res = {}
  for i=1, math.min(k, #t) do
    res[#res+1] = t[i].row
  end
  return res, t[1] and t[1].dist or nil
end

-- Compute mean and standard deviation of a numeric table
local function mean_and_sd(vals)
  local n = #vals
  if n == 0 then return 0, 0 end
  local sum = 0
  for _, v in ipairs(vals) do sum = sum + v end
  local mu = sum / n
  local sqsum = 0
  for _, v in ipairs(vals) do sqsum = sqsum + (v - mu)^2 end
  local sd = n > 1 and math.sqrt(sqsum/(n-1)) or 0
  return mu, sd
end

------------------------------------------------------------------
-- APACHE-17 – Unexpected performance relative to neighbors
------------------------------------------------------------------
print("== APACHE-17 ==")

local neighbor_k = 5
local residuals = {}
local res_vals = {}

for idx, r in ipairs(tbl.rows) do
  local neigh, _ = nearest_rows(r, tbl.rows, neighbor_k)
  local sum, cnt = 0, 0
  for _, n in ipairs(neigh) do
    sum = sum + n[perf_at]
    cnt = cnt + 1
  end
  local mean_n = cnt > 0 and sum/cnt or r[perf_at]
  local resid = math.abs(r[perf_at] - mean_n)
  table.insert(residuals, {row=r, idx=idx, resid=resid,
                           perf=r[perf_at], mean_n=mean_n})
  table.insert(res_vals, resid)
end

local mean_res, sd_res = mean_and_sd(res_vals)
local thresh_res = mean_res + 2 * sd_res

-- Pick top 3 rows with residual > threshold, sorted by residual desc
table.sort(residuals, function(a,b) return a.resid > b.resid end)
local top_res = {}
for i=1, math.min(3, #residuals) do
  local r = residuals[i]
  if r.resid > thresh_res then
    table.insert(top_res, r)
  end
end

print(string.format("%-5s %-20s %-10s %-10s %-10s",
        "Row", "Configuration", "Perf", "Mean_Neigh", "Resid"))
for _, r in ipairs(top_res) do
  print(string.format("%-5d %-20s %-10s %-10s %-10s",
        r.idx, row_to_str(r.row),
        ezr.round(r.perf), ezr.round(r.mean_n), ezr.round(r.resid)))
end
if #top_res == 0 then
  print("No configurations exceeded the 2σ residual threshold.")
end

------------------------------------------------------------------
-- APACHE-18 – Outlier configuration similarity
------------------------------------------------------------------
print("\n== APACHE-18 ==")

local distances = {}
local dist_vals = {}

for idx, r in ipairs(tbl.rows) do
  local _, dist = nearest_rows(r, tbl.rows, 1)
  table.insert(distances, {row=r, idx=idx, dist=dist,
                           perf=r[perf_at]})
  table.insert(dist_vals, dist)
end

local mean_dist, sd_dist = mean_and_sd(dist_vals)
local thresh_dist = mean_dist + 2 * sd_dist

-- Pick top 3 rows with distance > threshold, sorted by distance desc
table.sort(distances, function(a,b) return a.dist > b.dist end)
local top_dist = {}
for i=1, math.min(3, #distances) do
  local r = distances[i]
  if r.dist > thresh_dist then
    table.insert(top_dist, r)
  end
end

print(string.format("%-5s %-20s %-10s %-10s",
        "Row", "Configuration", "Perf", "Nearest_Dist"))
for _, r in ipairs(top_dist) do
  print(string.format("%-5d %-20s %-10s %-10s",
        r.idx, row_to_str(r.row),
        ezr.round(r.perf), ezr.round(r.dist)))
end
if #top_dist == 0 then
  print("No configurations exceeded the 2σ distance threshold.")
end
