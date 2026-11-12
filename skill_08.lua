#!/usr/bin/env lua
--  skill_08.lua – alternative vehicle recommendation (CAR‑15 & CAR‑16)
--  Usage: lua skill_08.lua <csv_path>
--
--  The script first loads the existing CSV, then prompts the
--  user for two row numbers:
--     1. A poorly rated vehicle for which we will show better
--        alternatives with similar characteristics (CAR‑15).
--     2. A vehicle that does not meet preferred performance
--        targets for which we will suggest the closest alternative
--        that meets more targets while changing few characteristics
--        (CAR‑16).
--
--  All calculations use the existing ezr library functions.

local ezr = require("ezr")

-- ------------------------------------------------------------------
-- Helper functions
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

-- Pretty‑print a row.
local function print_row(r)
  return string.format("%-5s %-6s %-7s %-4s %-4s %-4s %-4s %-4s",
    ezr.show(r[1]), ezr.show(r[2]), ezr.show(r[3]), ezr.show(r[4]),
    ezr.show(r[5]), ezr.show(r[6]), ezr.show(r[7]), ezr.show(r[8]))
end

-- Difference description between two rows over independent columns.
local function diff_desc(row1, row2, xcols)
  local parts = {}
  for _, c in ipairs(xcols) do
    local v1, v2 = row1[c.at], row2[c.at]
    if v1 ~= v2 then
      if type(v1) == "number" and type(v2) == "number" then
        table.insert(parts,
          string.format("%s: %.1f→%.1f", c.name, v1, v2))
      else
        table.insert(parts, string.format("%s: %s→%s", c.name, v1, v2))
      end
    end
  end
  return table.concat(parts, "; ")
end

-- Compute median of a numeric list.
local function median(nums)
  local sorted = ezr.sorted(nums)
  local n = #sorted
  if n == 0 then return nil end
  if n % 2 == 1 then return sorted[(n+1)/2]
  else return (sorted[n/2] + sorted[n/2 + 1]) / 2 end
end

-- Count how many performance targets a row meets.
-- Targets: weight <= median, acceleration >= median, mpg >= median
local function target_count(row, thresholds)
  local cnt = 0
  if row[thresholds.Lbs.at] <= thresholds.Lbs then cnt = cnt + 1 end
  if row[thresholds.Acc.at] >= thresholds.Acc then cnt = cnt + 1 end
  if row[thresholds.Mpg.at] >= thresholds.Mpg then cnt = cnt + 1 end
  return cnt
end

-- ------------------------------------------------------------------
-- Load data
-- ------------------------------------------------------------------
local csv_path = arg[1]
if not csv_path then
  io.stderr:write("usage: lua skill_08.lua <csv_path>\n")
  os.exit(1)
end

local tbl = ezr.Tbl(ezr.csv(csv_path))
local rows = tbl.rows
local xcols = tbl.cols.x

-- Reference to goal columns
local cols = {
  Lbs = tbl.cols.all[5],   -- "Lbs-"
  Acc = tbl.cols.all[7],   -- "Acc+"
  Mpg = tbl.cols.all[8]    -- "Mpg+"
}

-- ------------------------------------------------------------------
-- Prompt for the first row (poorly rated vehicle)
-- ------------------------------------------------------------------
print("Enter the index of a poorly rated vehicle (1‑based):")
local line = io.read()
local idx = tonumber(line)
if not idx or idx < 1 or idx > #rows then
  print("Invalid index.")
  os.exit(1)
end
local poor_row = rows[idx]

-- ------------------------------------------------------------------
-- CAR‑15: Alternatives with similar characteristics but better performance
-- ------------------------------------------------------------------
print("\n== CAR-15 ==")

-- Compute composite score of the poorly rated vehicle
local poor_score = composite_score(poor_row, cols)

-- Find nearest neighbors (k=10) by distx
local neighbor_candidates = {}
for _, r in ipairs(rows) do
  if r ~= poor_row then
    local d = tbl:distx(poor_row, r)
    table.insert(neighbor_candidates, {row=r, dist=d})
  end
end
table.sort(neighbor_candidates, function(a,b) return a.dist < b.dist end)

-- Take top 10 nearest neighbors
local k = 10
local nearest = {}
for i=1, math.min(k, #neighbor_candidates) do
  nearest[i] = neighbor_candidates[i]
end

-- Filter those with higher composite score
local better = {}
for _, cand in ipairs(nearest) do
  local s = composite_score(cand.row, cols)
  if s > poor_score then
    table.insert(better, {row=cand.row, dist=cand.dist, score=s})
  end
end

if #better == 0 then
  print("No better alternatives found among the nearest neighbors.")
else
  -- Show top 3 better alternatives
  table.sort(better, function(a,b) return a.score > b.score end)
  for i=1, math.min(3, #better) do
    local alt = better[i]
    print(string.format("%d) Vehicle %s  Distance=%.3f  Composite Score=%.3f",
      i, print_row(alt.row), alt.dist, alt.score))
    print("   Differences: " .. diff_desc(poor_row, alt.row, xcols))
    print()
  end
end

-- ------------------------------------------------------------------
-- Prompt for the second row (vehicle not meeting preferred targets)
-- ------------------------------------------------------------------
print("\nEnter the index of a vehicle that does not meet preferred performance targets (1‑based):")
line = io.read()
local idx2 = tonumber(line)
if not idx2 or idx2 < 1 or idx2 > #rows then
  print("Invalid index.")
  os.exit(1)
end
local target_row = rows[idx2]

-- ------------------------------------------------------------------
-- CAR‑16: Closest alternative meeting more targets
-- ------------------------------------------------------------------
print("\n== CAR-16 ==")

-- Compute median thresholds for targets
local med_Lbs = median(ezr.sorted(ezr.map(rows, function(r) return r[cols.Lbs.at] end), function(a,b) return a<b end))
local med_Acc = median(ezr.sorted(ezr.map(rows, function(r) return r[cols.Acc.at] end), function(a,b) return a<b end))
local med_Mpg = median(ezr.sorted(ezr.map(rows, function(r) return r[cols.Mpg.at] end), function(a,b) return a<b end))

local thresholds = { Lbs=med_Lbs, Acc=med_Acc, Mpg=med_Mpg }

local cur_count = target_count(target_row, thresholds)

-- Search all rows for ones meeting more targets
local best = nil
for _, r in ipairs(rows) do
  if r ~= target_row then
    local cnt = target_count(r, thresholds)
    if cnt > cur_count then
      local d = tbl:distx(target_row, r)
      if not best or cnt > best.cnt or (cnt == best.cnt and d < best.dist) then
        best = {row=r, cnt=cnt, dist=d}
      end
    end
  end
end

if not best then
  print("No alternative found that meets more performance targets.")
else
  print(string.format("Best alternative (row %d):", idx2))
  print(print_row(best.row))
  print(string.format("Distance=%.3f  Meets %d of 3 targets (vs %d for original)",
    best.dist, best.cnt, cur_count))
  print("Differences: " .. diff_desc(target_row, best.row, xcols))
end
