#!/usr/bin/env lua
-- skill_07.lua
-- Implements the requested APACHE-13 and APACHE-14 analyses.
--
-- Usage: lua skill_07.lua <csv path>
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
assert(csvfile, "Usage: lua skill_07.lua <csv path>")

-- Build table
local tbl = ezr.Tbl(ezr.csv(csvfile))
local perf_col = tbl.cols.y[1]   -- only goal column

-- Helper: map rows to their original indices
local idx_of_row = {}
for i,r in ipairs(tbl.rows) do
  idx_of_row[r] = i
end

-- ---------- APACHE-13 ----------
print("== APACHE-13 ==")

-- Choose a very small evaluation budget
local small_cap = math.min(20, #tbl.rows)   -- at least 5 rows if available
local small_lab = tbl:acquirer(small_cap)

if #small_lab == 0 then
  print("No configurations were evaluated.")
else
  local best_row = small_lab[1]   -- best by performance (disty smallest)
  print(string.format("Evaluated %d configurations (budget=%d).", #small_lab, small_cap))
  print(string.format("Selected configuration (lowest performance):"))
  print(string.format("  Performance = %s", display(best_row[perf_col.at])))
  print(string.format("  Configuration = %s", row_to_csv(best_row)))
end

-- ---------- APACHE-14 ----------
print("\n== APACHE-14 ==")

-- Larger but still limited budget for recommendation
local big_cap = math.min(50, #tbl.rows)   -- e.g., 50 or all rows if fewer
local big_lab = tbl:acquirer(big_cap)

if #big_lab == 0 then
  print("No configurations could be recommended.")
else
  local limit = math.min(20, #big_lab)   -- show up to 20 recommendations
  print(string.format("Recommended evaluation order (top %d rows sorted by promising performance):", limit))
  for i=1,limit do
    local r = big_lab[i]
    print(string.format("  #%d: performance=%s, config=%s",
          idx_of_row[r], display(r[perf_col.at]), row_to_csv(r)))
  end
  if #big_lab > limit then
    print(string.format("  ... (%d more configurations not shown)", #big_lab - limit))
  end
end
