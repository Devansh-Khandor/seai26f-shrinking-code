#!/usr/bin/env lua
-- Skill 07 – Apache configuration recommendation with limited budget
-- Requires the ezr library (ezr.lua & ezr-lib.lua in the same dir)

local ezr = require"ezr"

-- ------------------------------------------------------------------
-- Get the CSV file path from the command line
-- ------------------------------------------------------------------
local csv_path = arg[1]
assert(csv_path, "Usage: lua skill_07.lua <csv path>")

-- ------------------------------------------------------------------
-- Load the data into an ezr table
-- ------------------------------------------------------------------
local tbl = ezr.Tbl(ezr.csv(csv_path))

-- ------------------------------------------------------------------
-- Helper: pretty‑print a row as "col=value" pairs
-- ------------------------------------------------------------------
local function row_to_str(r)
  local parts = {}
  for _, col in ipairs(tbl.cols.x) do
    parts[#parts+1] = col.name .. "=" .. tostring(r[col.at])
  end
  parts[#parts+1] = tbl.cols.klass.name .. "=" .. tostring(r[tbl.cols.klass.at])
  return table.concat(parts, ", ")
end

-- ------------------------------------------------------------------
-- APACHE-13 – Discover a low‑performance configuration with few evaluations
-- ------------------------------------------------------------------
print("== APACHE-13 ==")
local perf_at = tbl.cols.klass.at

-- Use the configured budget from ezr settings
local budget = the.budget or 50

-- Acquire a set of configurations to evaluate
local lab = tbl:acquirer(budget)

-- Find the best (lowest performance) among those evaluated
local best = lab[1]

print(string.format("Evaluated %d configurations (budget=%d).", #lab, budget))
print(string.format("Best configuration found: %s", row_to_str(best)))
print(string.format("Performance value: %s",
                    ezr.round(best[perf_at])))

-- ------------------------------------------------------------------
-- APACHE-14 – Recommend evaluation order for a limited budget
-- ------------------------------------------------------------------
print("\n== APACHE-14 ==")
print("Recommended order to evaluate configurations first:")

-- We recommend the first few from the acquirer order
local recommend_count = math.min(5, #lab)
for i=1,recommend_count do
  local r = lab[i]
  print(string.format("%d. %s (Performance=%s)",
                      i, row_to_str(r), ezr.round(r[perf_at])))
end
