#!/usr/bin/env lua
--  skill_07.lua
--
--  Implements the following Apache diagnostics:
--
--  APACHE-13 – Discover a configuration with a low
--              performance metric while evaluating as few
--              candidate configurations as reasonably
--              possible.
--
--  APACHE-14 – When a limited benchmarking budget is
--              available, recommend which configurations
--              should be evaluated first so that promising
--              server settings are discovered early.
--
--  Usage:
--      lua skill_07.lua <csv_path>
--
--  The script prints a line with the exact heading
--  "== APACHE-13 ==" and "== APACHE-14 ==" followed by
--  at least three lines of results each.

local ezr = require"ezr"

-- ------------------------------------------------------------------
-- Utility: load CSV into a table and return the table object
-- ------------------------------------------------------------------
local function load_csv(path)
  local src = ezr.csv(path)
  return ezr.Tbl(src)
end

-- ------------------------------------------------------------------
-- Utility: find the index of a row in tbl.rows (by reference)
-- ------------------------------------------------------------------
local function find_row_index(tbl, row)
  for i, r in ipairs(tbl.rows) do
    if r == row then return i end
  end
  return nil
end

-- ------------------------------------------------------------------
-- Main
-- ------------------------------------------------------------------
local csv_path = arg[1]
if not csv_path then
  io.stderr:write("usage: lua skill_07.lua <csv_path>\n")
  os.exit(1)
end

local tbl = load_csv(csv_path)
local perf_col = tbl.cols.y[1]
local n = #tbl.rows

-- Define a small but meaningful budget: at most 20 configs or 10% of the data
local budget = math.min(20, math.floor(n * 0.10))
if budget < 1 then budget = 1 end

-- Acquire the best configurations according to the current model
-- (the model is built on the full set of rows, but the acquirer
-- only selects a subset to “label”)
local acquired = tbl:acquirer(budget)

-- ------------------------------------------------------------------
-- APACHE-13 : discover a low‑performance configuration
-- ------------------------------------------------------------------
print("== APACHE-13 ==")

-- Find the configuration with the lowest actual performance
local best_row, best_perf
for _, r in ipairs(acquired) do
  local p = r[perf_col.at]
  if best_perf == nil or p < best_perf then
    best_perf = p
    best_row = r
  end
end

local best_idx = find_row_index(tbl, best_row)

print(string.format(
  "Discovered a low‑performance configuration: Config %d (performance=%.2f)",
  best_idx, best_perf))

print(string.format("Number of configurations evaluated: %d", #acquired))

print("\nSample of evaluated configurations:")
for i=1, math.min(3, #acquired) do
  local r = acquired[i]
  local idx = find_row_index(tbl, r)
  print(string.format("  Config %d: actual=%.2f", idx, r[perf_col.at]))
end

-- ------------------------------------------------------------------
-- APACHE-14 : recommend the first configurations to evaluate
-- ------------------------------------------------------------------
print("\n== APACHE-14 ==")

-- Build a quick regression tree on the acquired set to obtain
-- predictions for the same rows (this mimics an active learner)
local tree = ezr.Tree(tbl, acquired)
local preds = {}
for i, r in ipairs(acquired) do
  preds[i] = tree:leaf(tbl, r)
end

print("Recommended configurations to evaluate first (best predictions first):")
for i=1, math.min(5, #acquired) do
  local r = acquired[i]
  local idx = find_row_index(tbl, r)
  print(string.format(
    "  %d) Config %d: predicted=%.2f actual=%.2f",
    i, idx, preds[i], r[perf_col.at]))
end
