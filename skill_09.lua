#!/usr/bin/env lua
-- skill_09.lua
-- Implements the requested APACHE-17 and APACHE-18 analyses.
--
-- Usage: lua skill_09.lua <csv path>
--
-- Requires the ezr library from the provided codebase.

local ezr = require"ezr"

-- ------------------------------------------------------------------
-- Utility helpers (copied from earlier skills)

-- Round numbers using ezr's formatting
local function display(v)
  if type(v) == "number" then
    return ezr.round(v)
  else
    return tostring(v)
  end
end

-- Convert a row table to a CSV string
local function row_to_csv(row)
  local parts = {}
  for _,v in ipairs(row) do
    if v == "?" then
      table.insert(parts, "?")
    else
      table.insert(parts, tostring(v))
    end
  end
  return table.concat(parts, ",")
end

-- ------------------------------------------------------------------
-- Main logic

local csvfile = arg[1]
assert(csvfile, "Usage: lua skill_09.lua <csv path>")

-- Build table
local tbl = ezr.Tbl(ezr.csv(csvfile))
local perf_col = tbl.cols.y[1]   -- only goal column

-- Map rows to their original indices
local idx_of_row = {}
for i,r in ipairs(tbl.rows) do
  idx_of_row[r] = i
end

-- Helper to compute mean and std of a list of numbers
local function mean_and_std(values)
  local n = #values
  if n == 0 then return 0,0 end
  local sum, sumsq = 0, 0
  for _,v in ipairs(values) do
    sum = sum + v
    sumsq = sumsq + v*v
  end
  local mean = sum / n
  local var = n > 1 and (sumsq - n*mean*mean)/(n-1) or 0
  return mean, math.sqrt(var)
end

-- Pre‑compute overall performance statistics
local perf_vals = ezr.map(tbl.rows, function(r) return r[perf_col.at] end)
local mean_perf, std_perf = mean_and_std(perf_vals)

-- ------------------------------------------------------------------
-- APACHE-17: Unusually different performance from expectation

print("== APACHE-17 ==")

local k = math.min(5, #tbl.rows - 1)   -- nearest‑neighbour count

local flagged_17 = {}
for _,row in ipairs(tbl.rows) do
  -- find k nearest neighbours
  local neigh = {}
  for _,r2 in ipairs(tbl.rows) do
    if r2 ~= row then
      local d = tbl:distx(row, r2)
      table.insert(neigh, {dist=d, row=r2})
    end
  end
  table.sort(neigh, function(a,b) return a.dist < b.dist end)
  local neigh_k = {unpack(neigh, 1, k)}

  -- expected performance (inverse‑distance weighting)
  local sumw, sump = 0, 0
  for _,n in ipairs(neigh_k) do
    local w = 1/(n.dist + 1e-6)   -- avoid division by zero
    sumw = sumw + w
    sump = sump + w * n.row[perf_col.at]
  end
  local exp_perf = sumw > 0 and sump / sumw or row[perf_col.at]

  -- neighbour performance std
  local neigh_vals = {}
  for _,n in ipairs(neigh_k) do
    table.insert(neigh_vals, n.row[perf_col.at])
  end
  local _, neigh_std = mean_and_std(neigh_vals)

  local diff = row[perf_col.at] - exp_perf
  local absdiff = math.abs(diff)

  if neigh_std > 0 and absdiff > 2 * neigh_std then
    table.insert(flagged_17, {
      idx = idx_of_row[row],
      perf = row[perf_col.at],
      expected = exp_perf,
      diff = diff
    })
  end
end

if #flagged_17 == 0 then
  print("No configurations show unusually large performance deviations from expectation.")
else
  print(string.format("Found %d unusually deviant configurations (|Δ| > 2σ).", #flagged_17))
  print("Index | Performance | Expected | Δ")
  for i,rec in ipairs(flagged_17) do
    print(string.format("%5d | %11s | %8s | %5s",
          rec.idx, display(rec.perf), display(rec.expected), display(rec.diff)))
    if i >= 5 then break end  -- show at most first 5 for brevity
  end
end

-- ------------------------------------------------------------------
-- APACHE-18: Unusually distinctive setting combinations

print("\n== APACHE-18 ==")

local avg_dists = {}
for _,row in ipairs(tbl.rows) do
  -- compute average distance to k nearest neighbours
  local neigh = {}
  for _,r2 in ipairs(tbl.rows) do
    if r2 ~= row then
      table.insert(neigh, tbl:distx(row, r2))
    end
  end
  table.sort(neigh)
  local avg = 0
  for j=1, k do
    avg = avg + neigh[j]
  end
  avg = avg / k
  table.insert(avg_dists, avg)
end
local mean_dist, std_dist = mean_and_std(avg_dists)

local flagged_18 = {}
for idx,row in ipairs(tbl.rows) do
  -- recompute avg distance (reuse pre‑calc if stored, else compute)
  local neigh = {}
  for _,r2 in ipairs(tbl.rows) do
    if r2 ~= row then
      table.insert(neigh, tbl:distx(row, r2))
    end
  end
  table.sort(neigh)
  local avg = 0
  for j=1, k do
    avg = avg + neigh[j]
  end
  avg = avg / k

  local perf = row[perf_col.at]
  if avg > mean_dist + 2 * std_dist and math.abs(perf - mean_perf) <= std_perf then
    table.insert(flagged_18, {
      idx = idx_of_row[row],
      perf = perf,
      avg_dist = avg
    })
  end
end

if #flagged_18 == 0 then
  print("No configurations have unusually distinctive setting combinations with normal performance.")
else
  print(string.format("Found %d configurations with highly distinctive setting patterns (distance > mean+2σ) but normal performance.", #flagged_18))
  print("Index | Performance | Avg Dist")
  for i,rec in ipairs(flagged_18) do
    print(string.format("%5d | %11s | %8s",
          rec.idx, display(rec.perf), display(rec.avg_dist)))
    if i >= 5 then break end  -- show at most first 5
  end
end
