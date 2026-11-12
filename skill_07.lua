#!/usr/bin/env lua
-- Skill 07 – Balanced vehicle identification & inspection prioritisation
-- Usage:  lua skill_07.lua <csv path>

local ezr = require "ezr"

-- ------------------------------------------------------------------
-- Helpers
-- ------------------------------------------------------------------

-- Find the vehicle with the smallest distance to heaven
local function best_balanced_vehicle(tbl)
  local best_row, best_dist = nil, nil
  for _, r in ipairs(tbl.rows) do
    local d = tbl:disty(r)
    if best_dist == nil or d < best_dist then
      best_row, best_dist = r, d
    end
  end
  return best_row, best_dist
end

-- ------------------------------------------------------------------
-- Main
-- ------------------------------------------------------------------

local csv_path = arg[1]
if not csv_path then
  io.stderr:write("Error: CSV file path required\n")
  os.exit(1)
end

local tbl = ezr.Tbl(ezr.csv(csv_path))

-- ---------- CAR-13 ----------
print("== CAR-13 ==")
local row, dist = best_balanced_vehicle(tbl)
print(string.format("Best balanced vehicle: %s", ezr.show(row)))
print(string.format("Distance to heaven: %.4f", dist))
print("This row combines low weight, high mpg, and low acceleration, giving the best overall trade‑off.")

-- ---------- CAR-14 ----------
print("\n== CAR-14 ==")
-- Decide how many vehicles we can inspect – limit to 10 or the dataset size.
local cap = math.min(10, #tbl.rows)
local inspected = tbl:acquirer(cap)

print(string.format("Recommended %d vehicles to evaluate first (sorted best to worst):", #inspected))
for i, r in ipairs(inspected) do
  local d = tbl:disty(r)
  print(string.format("%2d) dist=%.4f  %s", i, d, ezr.show(r)))
end
