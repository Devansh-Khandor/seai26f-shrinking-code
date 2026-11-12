#!/usr/bin/env lua
-- skill_04.lua
-- Implements the requested APACHE-07 and APACHE-08 analyses.
--
-- Usage: lua skill_04.lua <csv path>
--
-- Requires the ezr library from the provided codebase.

local ezr = require"ezr"

-- Helper: median of a numeric list (assumes list is non‑empty).
local function median(list)
  local s = ezr.sorted(list)
  local n = #s
  if n % 2 == 1 then
    return s[(n + 1) // 2]
  else
    return (s[n // 2] + s[n // 2 + 1]) / 2
  end
end

-- Helper: quartiles (returns Q1, Q3)
local function quartiles(list)
  local s = ezr.sorted(list)
  local n = #s
  local q1 = s[(n + 3) // 4]   -- floor((n+3)/4)
  local q3 = s[(3 * n + 1) // 4]  -- floor((3n+1)/4)
  return q1, q3
end

-- Helper: display numbers using ezr's rounding
local function display(v)
  if type(v) == "number" then
    return ezr.round(v)
  else
    return tostring(v)
  end
end

-- Helper: convert a row table to CSV string.
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

-- Load the data file.
local csvfile = arg[1]
assert(csvfile, "Usage: lua skill_04.lua <csv path>")

-- Build the table from the CSV file.
local tbl = ezr.Tbl(ezr.csv(csvfile))
local perf_col = tbl.cols.y[1]   -- the single goal column

-- Gather all performance values.
local perf_vals = ezr.map(tbl.rows, function(r) return r[perf_col.at] end)

-- Basic statistics.
local perf_mean = perf_col:mid()
local perf_min  = ezr.min(perf_vals)
local perf_max  = ezr.max(perf_vals)
local perf_med  = median(perf_vals)

-- ------------------- APACHE-07 -------------------
-- Unusually poor configuration alert (high performance).
print("== APACHE-07 ==")
print(string.format("Performance summary: min=%s, max=%s, mean=%s, median=%s",
      display(perf_min), display(perf_max), display(perf_mean), display(perf_med)))

-- Determine the outlier threshold (high side: Q3 + 1.5*IQR).
local q1, q3 = quartiles(perf_vals)
local iqr = q3 - q1
local high_threshold = q3 + 1.5 * iqr

-- Scan for unusually poor rows.
local any_alert = false
for idx, r in ipairs(tbl.rows) do
  local perf = r[perf_col.at]
  if perf > high_threshold then
    any_alert = true
    print(string.format(
      "Alert: configuration #%d is unusually poor (performance=%s)",
      idx, display(perf)))
  end
end
if not any_alert then
  print("No unusually poor configurations detected.")
end

-- ------------------- APACHE-08 -------------------
print("\n== APACHE-08 ==")

-- Compute recommended shortlist: top 5 best configurations (lowest performance).
local best_rows = ezr.keysort(tbl.rows, function(r) return r[perf_col.at] end)
local shortlist = {}
for i = 1, math.min(5, #best_rows) do
  local r = best_rows[i]
  table.insert(shortlist, r)
end

print("Recommended best configurations (by lowest performance):")
for rank, r in ipairs(shortlist) do
  print(string.format("  #%d: performance=%s, config=%s",
        rank, display(r[perf_col.at]), row_to_csv(r)))
end
