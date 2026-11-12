#!/usr/bin/env lua
-- skill_04.lua
-- CAR-07 & CAR-08:
--   CAR-07   – immediately evaluate each incoming vehicle and
--              indicate if it is among the top‑desirable cars.
--   CAR-08   – keep inventory summaries and a short list of
--              recommended vehicles updated incrementally.
--
-- Usage:  lua skill_04.lua <path-to-auto93.csv>
--
-- The script prints a header for each skill followed by at
-- least three lines of numeric results.  The calculations
-- are performed using the ezr library bundled with this
-- repository.

local ezr   = require("ezr")
local csv   = ezr.csv

--------------------------------------------------------------------
-- Helper utilities
--------------------------------------------------------------------

-- Compute median of a numeric list
local function median(vals)
  local n = #vals
  if n == 0 then return nil end
  local sorted = ezr.sorted(vals)
  if n % 2 == 1 then
    return sorted[(n + 1) // 2]
  else
    return (sorted[n // 2] + sorted[n // 2 + 1]) / 2
  end
end

-- Build a list of numeric values for a column (skipping "?" or nil)
local function numeric_values(tbl, col_idx)
  local out = {}
  for _, row in ipairs(tbl.rows) do
    local v = row[col_idx]
    if v ~= "?" and v ~= nil then table.insert(out, v) end
  end
  return out
end

-- Desirability metric – higher is better
-- We use a simple ratio of MPG to weight
local function desirability(row, mpg_idx, lbs_idx)
  local mpg = row[mpg_idx]
  local lbs = row[lbs_idx]
  if mpg == "?" or lbs == "?" or mpg == nil or lbs == nil then
    return nil
  end
  return mpg / lbs
end

-- Find the best (top N) vehicles by desirability
local function top_n(rows, mpg_idx, lbs_idx, n)
  local candidates = {}
  for i, row in ipairs(rows) do
    local d = desirability(row, mpg_idx, lbs_idx)
    if d then
      table.insert(candidates, {idx = i, mpg = row[mpg_idx], lbs = row[lbs_idx], d = d})
    end
  end
  table.sort(candidates, function(a, b) return a.d > b.d end)
  return table.sub(candidates, 1, n)
end

-- Subtable from a list (Lua 5.1 compatible)
local function slice(tbl, lo, hi)
  hi = hi or #tbl
  local res = {}
  for i = lo, hi do res[#res+1] = tbl[i] end
  return res
end

--------------------------------------------------------------------
-- Main
--------------------------------------------------------------------

local function main(csv_path)
  -- Load all rows once – the ezr.Tbl constructor will read them.
  local tbl = ezr.Tbl(csv(csv_path))

  -- Column indices for MPG and weight
  local idx_mpg, idx_lbs
  for _, col in ipairs(tbl.cols.all) do
    if col.name == "Mpg+" then idx_mpg = col.at end
    if col.name == "Lbs-" then idx_lbs = col.at end
  end
  if not idx_mpg or not idx_lbs then
    error("Required columns 'Mpg+' and 'Lbs-' not found")
  end

  ------------------------------------------------------------------
  -- CAR-07 – incremental assessment of each new vehicle
  ------------------------------------------------------------------
  print("== CAR-07 ==")
  local current_metrics = {}
  for i, row in ipairs(tbl.rows) do
    local d = desirability(row, idx_mpg, idx_lbs)
    if d then
      table.insert(current_metrics, d)
    end
    -- Compute rank among current set
    local rank = 1
    for _, x in ipairs(current_metrics) do
      if x > d then rank = rank + 1 end
    end
    local top_n = 5
    local is_top = rank <= top_n
    print(string.format("Row %d: desirability=%.4f, rank=%d, top %d? %s",
      i, d, rank, top_n, is_top and "YES" or "NO"))
  end

  ------------------------------------------------------------------
  -- CAR-08 – final inventory summary and recommended shortlist
  ------------------------------------------------------------------
  print("\n== CAR-08 ==")

  -- Summary statistics
  local mpg_vals   = numeric_values(tbl, idx_mpg)
  local lbs_vals   = numeric_values(tbl, idx_lbs)

  print(string.format("Total vehicles: %d", #tbl.rows))
  print(string.format("Mean MPG: %.2f", tbl.cols.all[idx_mpg]:mid()))
  print(string.format("Mean Lbs: %.2f", tbl.cols.all[idx_lbs]:mid()))
  print(string.format("Median MPG: %.2f", median(mpg_vals)))
  print(string.format("Median Lbs: %.2f", median(lbs_vals)))

  -- Recommended shortlist – top 5 by desirability
  local shortlist = top_n(tbl.rows, idx_mpg, idx_lbs, 5)
  print("\nTop 5 vehicles (most desirable):")
  for rank, rec in ipairs(shortlist) do
    print(string.format(
      "  #%d: Row %d | MPG=%.1f | Lbs=%.0f | Desirability=%.4f",
      rank, rec.idx, rec.mpg, rec.lbs, rec.d))
  end
end

-- Run if script is executed directly
if arg[1] then
  main(arg[1])
else
  io.stderr:write("Usage: lua skill_04.lua <csv_path>\n")
end
