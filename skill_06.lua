#!/usr/bin/env lua
-- Skill 06 – Explain recommendations and trade‑offs
-- Usage:  lua skill_06.lua <csv_path>

local ezr = require("ezr")
local round = ezr.round
local sorted = ezr.sorted
local keysort = ezr.keysort

-- ------------------------------------------------------------------
-- Utility: find a column object by name
local function get_col(tbl, name)
  for _, c in pairs(tbl.cols.all) do
    if c.name == name then return c end
  end
  error("column not found: " .. tostring(name))
end

-- ------------------------------------------------------------------
-- Read CSV
local csv_path = arg[1]
if not csv_path then
  io.stderr:write("Error: missing CSV path\n")
  os.exit(1)
end

local tbl = ezr.Tbl(ezr.csv(csv_path))

-- ------------------------------------------------------------------
-- Columns we need
local weight_col = get_col(tbl, "Lbs-")
local mpg_col    = get_col(tbl, "Mpg+")
local acc_col    = get_col(tbl, "Acc+")
local model_col  = get_col(tbl, "Model")

-- ------------------------------------------------------------------
-- Normalisation helpers (0..1)
local function norm(x, xmin, xmax)
  if xmax == xmin then return 0.5 end
  return (x - xmin) / (xmax - xmin)
end

-- ------------------------------------------------------------------
-- Gather min/max for the three metrics
local minW, maxW = math.huge, -math.huge
local minM, maxM = math.huge, -math.huge
local minA, maxA = math.huge, -math.huge

for _, row in ipairs(tbl.rows) do
  local w = row[weight_col.at]
  local m = row[mpg_col.at]
  local a = row[acc_col.at]
  if w < minW then minW = w end
  if w > maxW then maxW = w end
  if m < minM then minM = m end
  if m > maxM then maxM = m end
  if a < minA then minA = a end
  if a > maxA then maxA = a end
end

-- ------------------------------------------------------------------
-- Compute composite score for each vehicle
local vehicle_scores = {}
for _, row in ipairs(tbl.rows) do
  local w = row[weight_col.at]
  local m = row[mpg_col.at]
  local a = row[acc_col.at]
  local score = (1 - norm(w, minW, maxW))          -- lower weight
               + (1 - norm(a, minA, maxA))          -- lower acceleration
               +  norm(m, minM, maxM)               -- higher mpg
  vehicle_scores[#vehicle_scores+1] = {row=row, score=score}
end

-- ------------------------------------------------------------------
-- Sort by score descending
table.sort(vehicle_scores, function(a,b) return a.score > b.score end)

-- ------------------------------------------------------------------
-- CAR-11: Explain best recommendation
print("== CAR-11 ==")
local best = vehicle_scores[1]
local best_row = best.row
print(string.format(
  "Best vehicle: Model=%s, Score=%.2f, Weight=%d, Acc=%.2f, MPG=%.2f",
  best_row[model_col.at], round(best.score,2),
  best_row[weight_col.at], best_row[acc_col.at], best_row[mpg_col.at]))

-- Find 3 nearest neighbours using normalized distance over X columns
local neighbors = {}
for i=2, #vehicle_scores do
  local r = vehicle_scores[i].row
  local d = tbl:distx(best_row, r)
  neighbors[#neighbors+1] = {row=r, score=vehicle_scores[i].score, dist=d}
end
table.sort(neighbors, function(a,b) return a.dist < b.dist end)
local topN = math.min(3, #neighbors)

print("\nNearest similar vehicles:")
for i=1,topN do
  local n = neighbors[i]
  print(string.format(
    "%d) Model=%s | Score=%.2f | Dist=%.3f",
    i, n.row[model_col.at], round(n.score,2), round(n.dist,3)))
  print(string.format(
    "   Weight diff = %+.0f | Acc diff = %+.1f | MPG diff = %+.1f",
    n.row[weight_col.at] - best_row[weight_col.at],
    n.row[acc_col.at] - best_row[acc_col.at],
    n.row[mpg_col.at] - best_row[mpg_col.at]))
end

-- ------------------------------------------------------------------
-- CAR-12: Compare two vehicles with different strengths
print("\n== CAR-12 ==")
-- Take the best and worst
local worst = vehicle_scores[#vehicle_scores]
local worst_row = worst.row

print(string.format(
  "Top vehicle:  Model=%s | Score=%.2f | Weight=%d | Acc=%.2f | MPG=%.2f",
  best_row[model_col.at], round(best.score,2),
  best_row[weight_col.at], best_row[acc_col.at], best_row[mpg_col.at]))

print(string.format(
  "Bottom vehicle: Model=%s | Score=%.2f | Weight=%d | Acc=%.2f | MPG=%.2f",
  worst_row[model_col.at], round(worst.score,2),
  worst_row[weight_col.at], worst_row[acc_col.at], worst_row[mpg_col.at]))

-- Show trade‑offs
print("\nTrade‑offs:")
print(string.format(
  "Weight difference: %+.0f kg",
  worst_row[weight_col.at] - best_row[weight_col.at]))

print(string.format(
  "Acceleration difference: %+.1f m/s²",
  worst_row[acc_col.at] - best_row[acc_col.at]))

print(string.format(
  "MPG difference: %+.1f mpg",
  worst_row[mpg_col.at] - best_row[mpg_col.at]))

print("\nInterpretation:")
print("The top vehicle is lighter, accelerates faster, and achieves higher fuel efficiency, which together raise its composite score.")
print("The bottom vehicle suffers from heavier weight, slower acceleration, and lower MPG, resulting in a lower overall rating.")
print("If your priority is fuel economy and performance, the top vehicle is preferable; if you value other aspects (e.g., cargo space, not in this data), you might consider the bottom vehicle.")
