#!/usr/bin/env lua
-- Skill 08: CAR-15 & CAR-16
-- Usage: lua skill_08.lua <csv path>

local ezr = require("ezr")
local csv  = ezr.csv
local round = ezr.round
local keysort = ezr.keysort
local sorted = ezr.sorted
local math = math
local sqrt, floor = math.sqrt, math.floor

-- ------------------------------------------------------------------
-- Normalisation helpers
local function norm(v, mn, mx)
  if mx == mn then return 0 end
  return (v - mn) / (mx - mn)
end

-- ------------------------------------------------------------------
-- Composite score: lower weight better, higher mpg/acc better
local function composite_score(r, mins, maxs)
  local w = r["Lbs-"]
  local m = r["Mpg+"]
  local a = r["Acc+"]
  local w_score = (maxs["Lbs-"] - w) / (maxs["Lbs-"] - mins["Lbs-"])
  local m_score = (m - mins["Mpg+"]) / (maxs["Mpg+"] - mins["Mpg+"])
  local a_score = (a - mins["Acc+"]) / (maxs["Acc+"] - mins["Acc+"])
  return w_score + m_score + a_score
end

-- ------------------------------------------------------------------
-- Distance in normalised space
local function dist(r1, r2, mins, maxs)
  local d2 = 0
  for _,c in ipairs{"Lbs-", "Mpg+", "Acc+"} do
    local v1 = norm(r1[c], mins[c], maxs[c])
    local v2 = norm(r2[c], mins[c], maxs[c])
    local diff = v1 - v2
    d2 = d2 + diff * diff
  end
  return sqrt(d2)
end

-- ------------------------------------------------------------------
-- Pretty printing of a vehicle row
local function fmt_row(r)
  return string.format(
    "%-7s %-7s %-5s %-7s %-7s %-7s %-7s %-7s",
    r["Clndrs"], r["Volume"], r["HpX"], r["Model"],
    r["origin"], r["Lbs-"], r["Acc+"], r["Mpg+"])
end

-- ------------------------------------------------------------------
-- Load data
local path = arg[1]
if not path then
  io.stderr:write("Usage: lua skill_08.lua <csv path>\n")
  os.exit(1)
end

local rows_iter = csv(path)
local headers = rows_iter()
if not headers then
  io.stderr:write("Empty CSV\n")
  os.exit(1)
end

local rows = {}
for r in rows_iter do
  table.insert(rows, r)
end

-- ------------------------------------------------------------------
-- Compute min/max for key metrics
local mins, maxs = {}, {}
for _,c in ipairs{"Lbs-", "Mpg+", "Acc+"} do
  local mn, mx = math.huge, -math.huge
  for _,r in ipairs(rows) do
    local v = r[c]
    if v < mn then mn = v end
    if v > mx then mx = v end
  end
  mins[c], maxs[c] = mn, mx
end

-- Compute scores
for _,r in ipairs(rows) do
  r._score = composite_score(r, mins, maxs)
end

-- ------------------------------------------------------------------
-- Find worst scored vehicle
local worst = rows[1]
for _,r in ipairs(rows) do
  if r._score < worst._score then worst = r end
end

-- ------------------------------------------------------------------
-- CAR-15: Alternatives for a poorly rated vehicle
print("== CAR-15 ==")
print("Poorly rated vehicle:")
print(string.format("  %s  Score: %.3f",
      fmt_row(worst), round(worst._score,3)))
print("Top alternative vehicles:")

-- Find alternatives with higher score
local alternatives = {}
for _,r in ipairs(rows) do
  if r ~= worst and r._score > worst._score then
    local d = dist(r, worst, mins, maxs)
    table.insert(alternatives, {row=r, dist=d})
  end
end

-- Sort by distance
alternatives = sorted(alternatives, function(a,b) return a.dist < b.dist end)

-- Pick up to 3
for i=1, math.min(3, #alternatives) do
  local alt = alternatives[i]
  local r = alt.row
  print(string.format("  %d. %s  Score: %.3f  Dist: %.3f",
        i, fmt_row(r), round(r._score,3), round(alt.dist,3)))
  -- Differences
  local dw = r["Lbs-"] - worst["Lbs-"]
  local dm = r["Mpg+"] - worst["Mpg+"]
  local da = r["Acc+"] - worst["Acc+"]
  print(string.format("     Weight diff: %.1f  Mpg diff: %.1f  Acc diff: %.1f",
        dw, dm, da))
end

print()

-- ------------------------------------------------------------------
-- CAR-16: Find a vehicle not meeting preferred targets
-- Define preferred target as median score or higher
local sorted_rows = sorted(rows, function(a,b) return a._score < b._score end)
local median_idx = floor(#sorted_rows/2) + 1
local median_score = sorted_rows[median_idx]._score

-- Choose a vehicle below median but not the absolute worst
local target = nil
for _,r in ipairs(sorted_rows) do
  if r._score < median_score and r ~= worst then
    target = r
    break
  end
end
if not target then target = worst end

-- Find best alternative with higher score and minimal attribute change
local best_alt = nil
local best_dist = math.huge
for _,r in ipairs(rows) do
  if r ~= target and r._score > target._score then
    local d = dist(r, target, mins, maxs)
    if d < best_dist then
      best_dist = d
      best_alt = r
    end
  end
end

print("== CAR-16 ==")
print("Vehicle not meeting preferred targets:")
print(string.format("  %s  Score: %.3f",
      fmt_row(target), round(target._score,3)))
print("Closest better vehicle with minimal characteristic change:")
print(string.format("  %s  Score: %.3f  Dist: %.3f",
      fmt_row(best_alt), round(best_alt._score,3), round(best_dist,3)))
-- Differences
local dw = best_alt["Lbs-"] - target["Lbs-"]
local dm = best_alt["Mpg+"] - target["Mpg+"]
local da = best_alt["Acc+"] - target["Acc+"]
print(string.format("     Weight diff: %.1f  Mpg diff: %.1f  Acc diff: %.1f",
      dw, dm, da))
