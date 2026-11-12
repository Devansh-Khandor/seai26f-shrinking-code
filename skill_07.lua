#!/usr/bin/env lua
--  skill_07.lua – inventory prioritisation (CAR‑13 & CAR‑14)
--
--  Depends on the ezr library (ezr.lua & ezr-lib.lua) in the
--  same directory or in the Lua package path.
--
--  CAR‑13: Find one vehicle with the best balance of fuel
--          efficiency, weight, and acceleration – evaluating as
--          few candidates as possible.
--
--  CAR‑14: With a limited inspection budget, recommend the
--          vehicles that should be evaluated first so that we
--          discover promising inventory options early.

local ezr = require("ezr")

-- ------------------------------------------------------------------
-- Helpers
-- ------------------------------------------------------------------
-- Normalised score for a column.
local function norm_score(col, r, maximize)
  local v = r[col.at]
  if v == "?" then return 0.5 end        -- neutral if unknown
  local n = col:norm(v)
  return maximize and n or (1 - n)
end

-- Composite desirability score: higher is better.
local function composite_score(r, cols)
  return (norm_score(cols.Lbs, r, false) +
          norm_score(cols.Acc, r, true) +
          norm_score(cols.Mpg, r, true)) / 3
end

-- Pretty‑print a row of the table.
local function print_row(r)
  return string.format("%-5s %-6s %-7s %-4s %-4s %-4s %-4s %-4s",
    ezr.show(r[1]), ezr.show(r[2]), ezr.show(r[3]), ezr.show(r[4]),
    ezr.show(r[5]), ezr.show(r[6]), ezr.show(r[7]), ezr.show(r[8]))
end

-- ------------------------------------------------------------------
-- Load data
-- ------------------------------------------------------------------
local csv_path = arg[1]
if not csv_path then
  io.stderr:write("usage: lua skill_07.lua <csv_path>\n")
  os.exit(1)
end

local tbl = ezr.Tbl(ezr.csv(csv_path))

-- Reference to the columns we care about.
local cols = {
  Lbs = tbl.cols.all[5],   -- "Lbs-"
  Acc = tbl.cols.all[7],   -- "Acc+"
  Mpg = tbl.cols.all[8]    -- "Mpg+"
}

-- ------------------------------------------------------------------
-- CAR‑13: Find best balanced vehicle
-- ------------------------------------------------------------------
print("== CAR-13 ==")

-- Compute composite score for all rows
local best_row, best_score = nil, -1
for _, r in ipairs(tbl.rows) do
  local s = composite_score(r, cols)
  if s > best_score then
    best_score = s
    best_row = r
  end
end

print("Best balanced vehicle (highest composite score):")
print(print_row(best_row))
print(string.format("Composite score: %.3f", ezr.round(best_score, 3)))
print()

-- ------------------------------------------------------------------
-- CAR‑14: Recommend vehicles to evaluate first given a budget
-- ------------------------------------------------------------------
print("== CAR-14 ==")

-- Budget comes from the global settings; fall back to 10 if unset.
local budget = ezr.the.budget or 10
budget = math.min(budget, #tbl.rows)

-- Acquire rows sorted by distance to heaven (d2h)
local ranked = tbl:acquirer(budget)

print(string.format("Top %d vehicles to inspect first (sorted by d2h):", #ranked))
for i, r in ipairs(ranked) do
  local d2h = tbl:disty(r)
  print(string.format("%2d) %s  d2h=%.3f", i, print_row(r), ezr.round(d2h, 3)))
end
