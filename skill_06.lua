#!/usr/bin/env lua
-- Skill 06: Recommendation explanations (CAR-11 & CAR-12)
-- Usage: lua skill_06.lua <csv path>

local ezr = require("ezr")
local csv  = ezr.csv
local round = ezr.round
local sorted = ezr.sorted
local keysort = ezr.keysort

-- ------------------------------------------------------------------
-- Read the CSV file
local path = arg[1]
if not path then
  io.stderr:write("Usage: lua skill_06.lua <csv path>\n")
  os.exit(1)
end

local rows_iter = csv(path)
local headers = rows_iter()
if not headers then
  io.stderr:write("Empty CSV\n")
  os.exit(1)
end

-- Load all rows into a plain array
local rows = {}
for r in rows_iter do
  table.insert(rows, r)
end

-- ------------------------------------------------------------------
-- Identify numeric columns (any field whose first value is a number)
local numeric_cols = {}
for _, col in ipairs(headers) do
  local v = rows[1][col]
  if type(v) == "number" then
    table.insert(numeric_cols, col)
  end
end

-- Compute min/max for each numeric column
local col_min, col_max = {}, {}
for _, col in ipairs(numeric_cols) do
  local mn, mx = math.huge, -math.huge
  for _, r in ipairs(rows) do
    local v = r[col]
    if v ~= nil then
      if v < mn then mn = v end
      if v > mx then mx = v end
    end
  end
  col_min[col] = mn
  col_max[col] = mx
end

-- ------------------------------------------------------------------
-- Normalisation helper
local function norm(v, mn, mx)
  if mx == mn then return 0 end
  return (v - mn) / (mx - mn)
end

-- ------------------------------------------------------------------
-- Scoring: lower weight better, higher mpg/acc better
local function score_row(r)
  local w = r["Lbs-"]
  local m = r["Mpg+"]
  local a = r["Acc+"]
  local w_score = (col_max["Lbs-"] - w) / (col_max["Lbs-"] - col_min["Lbs-"])
  local m_score = (m - col_min["Mpg+"]) / (col_max["Mpg+"] - col_min["Mpg+"])
  local a_score = (a - col_min["Acc+"]) / (col_max["Acc+"] - col_min["Acc+"])
  return w_score + m_score + a_score
end

-- Compute scores and cache them
for _, r in ipairs(rows) do
  r._score = score_row(r)
end

-- Sort by score descending
local sorted_rows = sorted(rows, function(a,b) return a._score > b._score end)

-- ------------------------------------------------------------------
-- Distance between two rows in normalized space
local function dist(r1, r2)
  local d2 = 0
  for _, col in ipairs(numeric_cols) do
    local v1 = norm(r1[col], col_min[col], col_max[col])
    local v2 = norm(r2[col], col_min[col], col_max[col])
    local diff = v1 - v2
    d2 = d2 + diff * diff
  end
  return math.sqrt(d2)
end

-- ------------------------------------------------------------------
-- Utility: format a row for printing
local function fmt_row(r)
  return string.format(
    "%-7s %-7s %-5s %-7s %-7s %-7s %-7s %-7s",
    r["Clndrs"], r["Volume"], r["HpX"], r["Model"],
    r["origin"], r["Lbs-"], r["Acc+"], r["Mpg+"]
  )
end

-- ------------------------------------------------------------------
-- CAR-11: Explain recommendation of the best vehicle
print("== CAR-11 ==")
local best = sorted_rows[1]
print("\nRecommended vehicle (highest score):")
print(fmt_row(best))
print(string.format("Score: %.3f\n", best._score))

-- Find three nearest neighbours (excluding itself)
local neighbours = {}
for _, r in ipairs(rows) do
  if r ~= best then
    push(neighbours, {row=r, dist=dist(best, r)})
  end
end
neighbours = sorted(neighbours, function(a,b) return a.dist < b.dist end)
local top3 = {neighbours[1], neighbours[2], neighbours[3]}

print("Top 3 similar vehicles:")
for i, n in ipairs(top3) do
  local r = n.row
  print(string.format("\n%d. %s", i, fmt_row(r)))
  print(string.format("Distance: %.3f", n.dist))
  print("Metric differences (best - neighbour):")
  for _, col in ipairs(numeric_cols) do
    local diff = best[col] - r[col]
    print(string.format("  %s: %.3f", col, diff))
  end
end

-- ------------------------------------------------------------------
-- CAR-12: Compare the top two vehicles
print("\n== CAR-12 ==")
local second = sorted_rows[2]
if not second then
  print("Only one vehicle in the data.")
else
  print("\nComparison of top two vehicles:")
  print(fmt_row(best))
  print(fmt_row(second))
  print(string.format("\nScores: Best=%.3f, Second=%.3f (diff=%.3f)",
        best._score, second._score, best._score - second._score))

  print("\nMetric trade‑offs:")
  for _, col in ipairs(numeric_cols) do
    local bval = best[col]
    local sval = second[col]
    local higher = (bval > sval) and "best" or "second"
    local diff = bval - sval
    print(string.format("  %s: best=%.3f, second=%.3f (higher=%s, diff=%.3f)",
          col, bval, sval, higher, diff))
  end
end
