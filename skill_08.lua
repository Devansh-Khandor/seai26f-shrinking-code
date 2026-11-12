#!/usr/bin/env lua
-- skill_08.lua
-- APACHE-15: Suggest a minimal change to a poor configuration that lowers performance.
-- APACHE-16: Find the fewest setting changes needed to reach a target performance, using only valid configurations.
--
-- Usage:  lua skill_08.lua <path-to-Apache_AllMeasurements.csv>
--
-- Requires the ezr library (ezr.lua & ezr-lib.lua) in the same directory.

local ezr = require("ezr")
local csv  = ezr.csv
local sorted = ezr.sorted
local map = ezr.map
local copy = ezr.copy
local keys = ezr.keys
local show = ezr.show
local rand = ezr.rand

-- ------------------------------------------------------------------
-- Load data and basic stats
-- ------------------------------------------------------------------
local function load_data(path)
  local tbl = ezr.Tbl(csv(path))
  local perf_col = tbl.cols.y[1].at
  local perf_vals = {}
  for _, r in ipairs(tbl.rows) do
    local v = r[perf_col]
    if v ~= "?" then
      perf_vals[#perf_vals+1] = v
    end
  end
  local n = #perf_vals
  local sum, sumsq = 0,0
  for _, v in ipairs(perf_vals) do
    sum = sum + v
    sumsq = sumsq + v*v
  end
  local mean = sum / n
  local variance = (sumsq / n) - mean*mean
  local sd = math.sqrt(math.max(variance,0))
  return tbl, {mean=mean,sd=sd, n=n}
end

-- ------------------------------------------------------------------
-- Find the worst (highest performance) configuration
-- ------------------------------------------------------------------
local function find_worst(tbl)
  local worst = nil
  local idx = nil
  local perf_col = tbl.cols.y[1].at
  for i,r in ipairs(tbl.rows) do
    local v = r[perf_col]
    if v ~= "?" and (worst==nil or v>worst) then
      worst, idx = v, i
    end
  end
  return idx, worst
end

-- ------------------------------------------------------------------
-- Build quick lookup from a row key to its performance
-- ------------------------------------------------------------------
local function row_key(row)
  return table.concat(row, ",")
end

local function build_row_map(tbl)
  local map = {}
  local perf_col = tbl.cols.y[1].at
  for _,r in ipairs(tbl.rows) do
    local k = row_key(r)
    map[k] = r[perf_col]
  end
  return map
end

-- ------------------------------------------------------------------
-- kNN prediction (k=5)
-- ------------------------------------------------------------------
local function estimate_knn(tbl, data, row, k)
  k = k or 5
  local dists = {}
  for _, r in ipairs(data) do
    if r ~= row then
      local d = tbl:distx(row, r)
      dists[#dists+1] = {d=d, perf=r[tbl.cols.y[1].at]}
    end
  end
  table.sort(dists, function(a,b) return a.d < b.d end)
  local n = math.min(k, #dists)
  if n == 0 then return nil, nil end
  local sum, sumsq = 0,0
  for i=1,n do
    local v = dists[i].perf
    sum = sum + v
    sumsq = sumsq + v*v
  end
  local mean = sum / n
  local var = (sumsq / n) - mean*mean
  local sd = math.sqrt(math.max(var,0))
  return mean, sd
end

-- ------------------------------------------------------------------
-- APACHE-15: Suggest a single-setting change that improves performance
-- ------------------------------------------------------------------
local function apache15(tbl)
  print("== APACHE-15 ==")
  local idx, worst_perf = find_worst(tbl)
  if not idx then
    print("  No data available.")
    return
  end
  local worst_row = tbl.rows[idx]
  print(string.format("  Poor configuration (row %d, perf=%d):", idx, worst_perf))
  print("  "..table.concat(worst_row, ", "))

  local row_map = build_row_map(tbl)
  local candidates = {}
  -- examine each adjustable setting
  for _,c in ipairs(tbl.cols.x) do
    local alt_vals = {}
    -- gather unique values for this column
    for _,r in ipairs(tbl.rows) do
      local v = r[c.at]
      if v ~= "?" and v ~= worst_row[c.at] then
        alt_vals[v] = true
      end
    end
    for v,_ in pairs(alt_vals) do
      local cand = copy(worst_row)
      cand[c.at] = v
      local k = row_key(cand)
      if row_map[k] ~= nil then
        -- valid configuration exists in dataset
        local perf = row_map[k]
        candidates[#candidates+1] = {row=cand, perf=perf, diff=1}
      else
        -- estimate via kNN
        local pred, sd = estimate_knn(tbl, tbl.rows, cand)
        if pred then
          candidates[#candidates+1] = {row=cand, perf=pred, diff=1}
        end
      end
    end
  end

  if #candidates == 0 then
    print("  No single‑setting alternative found.")
    return
  end

  -- choose candidate with lowest performance
  table.sort(candidates, function(a,b) return a.perf < b.perf end)
  local best = candidates[1]
  print("  Recommended change:")
  for _,c in ipairs(tbl.cols.x) do
    if best.row[c.at] ~= worst_row[c.at] then
      print(string.format("    %s: %s -> %s",
                          c.name, tostring(worst_row[c.at]), tostring(best.row[c.at])))
    end
  end
  print(string.format("  Performance after change (actual/estimated): %s",
                      tostring(best.perf)))
  print()
end

-- ------------------------------------------------------------------
-- APACHE-16: Find minimal changes to reach a performance target
-- ------------------------------------------------------------------
local function apache16(tbl, stats)
  print("== APACHE-16 ==")
  local perf_col = tbl.cols.y[1].at
  local target = stats.mean - 1.5 * stats.sd
  print(string.format("  Target performance: %.0f (mean-1.5*sd)", target))

  local idx, worst_perf = find_worst(tbl)
  if not idx then
    print("  No data available.")
    return
  end
  local poor_row = tbl.rows[idx]
  print(string.format("  Poor configuration (row %d, perf=%d):", idx, worst_perf))
  print("  "..table.concat(poor_row, ", "))

  -- search all rows that meet or beat target
  local best = nil
  for _,r in ipairs(tbl.rows) do
    local v = r[perf_col]
    if v ~= "?" and v <= target then
      -- compute hamming distance
      local diff = 0
      for _,c in ipairs(tbl.cols.x) do
        if r[c.at] ~= poor_row[c.at] then diff = diff + 1 end
      end
      if not best or diff < best.diff or (diff==best.diff and v < best.perf) then
        best = {row=r, diff=diff, perf=v}
      end
    end
  end

  if not best then
    print("  No configuration meets the target performance.")
    return
  end

  print(string.format("  Best alternative (distance=%d, perf=%d):",
                      best.diff, best.perf))
  print("  "..table.concat(best.row, ", "))
  print()
end

-- ------------------------------------------------------------------
-- Main
-- ------------------------------------------------------------------
local function main()
  local path = arg[1]
  if not path then
    io.stderr:write("Error: CSV file path required\n")
    os.exit(1)
  end
  local tbl, stats = load_data(path)

  apache15(tbl)
  apache16(tbl, stats)
end

main()
