#!/usr/bin/env lua
-- Skill 04: Real‑time assessment and incremental inventory
-- Usage: lua skill_04.lua <csv path>

local ezr = require("ezr")
local csv = ezr.csv
local sorted = ezr.sorted
local round = ezr.round
local keysort = ezr.keysort

-- Helper to compute mean, std, min, max for a numeric list
local function stats(vals)
  local n, sum, sumsq = 0, 0, 0
  local minv, maxv = math.huge, -math.huge
  for _, v in ipairs(vals) do
    n = n + 1
    sum = sum + v
    sumsq = sumsq + v * v
    if v < minv then minv = v end
    if v > maxv then maxv = v end
  end
  local mean = sum / n
  local var = n > 1 and (sumsq - sum * sum / n) / (n - 1) or 0
  local std = math.sqrt(var)
  return mean, std, minv, maxv
end

-- Normalization helper
local function norm(v, minv, maxv, reverse)
  if maxv == minv then return 0 end
  local n = (v - minv) / (maxv - minv)
  return reverse and (1 - n) or n
end

-- ------------------------------------------------------------------
-- Read CSV
local path = arg[1]
if not path then
  io.stderr:write("Usage: lua skill_04.lua <csv path>\n")
  os.exit(1)
end

local rows_iter = csv(path)
local headers = rows_iter()
if not headers then
  io.stderr:write("Empty CSV\n")
  os.exit(1)
end

-- Containers for all rows and column values
local rows = {}
local weight_vals, mpg_vals, acc_vals = {}, {}, {}

-- Read rows
local idx = 0
for r in rows_iter do
  idx = idx + 1
  table.insert(rows, r)
  table.insert(weight_vals, r["Lbs-"])
  table.insert(mpg_vals, r["Mpg+"])
  table.insert(acc_vals, r["Acc+"])
end

-- ------------------------------------------------------------------
-- Summary statistics
local wmean, wstd, wmin, wmax = stats(weight_vals)
local mmean, mstd, mmin, mmax = stats(mpg_vals)
local amean, astd, amin, amax = stats(acc_vals)

-- ------------------------------------------------------------------
-- Compute score for each row
for _, r in ipairs(rows) do
  local w = r["Lbs-"]
  local m = r["Mpg+"]
  local a = r["Acc+"]
  r.score = norm(w, wmin, wmax, true) +
            norm(m, mmin, mmax, false) +
            norm(a, amin, amax, false)
end

-- Sort rows by score descending to get top cars
local sorted_by_score = sorted(rows, function(a,b) return a.score > b.score end)
local topN = math.min(5, #rows)
local top_scores = {}
for i = 1, topN do
  top_scores[sorted_by_score[i].score] = true
end

-- ------------------------------------------------------------------
-- CAR-07: real‑time assessment of each arriving car
print("== CAR-07 ==")
print("Index  Score  Desirable")
for i, r in ipairs(rows) do
  local des = top_scores[r.score] and "yes" or "no"
  print(string.format("%5d  %6.3f  %s", i, r.score, des))
end
print()

-- ------------------------------------------------------------------
-- CAR-08: inventory summary & recommended shortlist
print("== CAR-08 ==")
print(string.format(
  "Weight: mean=%.2f, std=%.2f, min=%.2f, max=%.2f",
  wmean, wstd, wmin, wmax))
print(string.format(
  "Fuel efficiency: mean=%.2f, std=%.2f, min=%.2f, max=%.2f",
  mmean, mstd, mmin, mmax))
print(string.format(
  "Acceleration: mean=%.2f, std=%.2f, min=%.2f, max=%.2f",
  amean, astd, amin, amax))
print()

print(string.format(
  "%-7s %-7s %-5s %-7s %-7s %-7s %-7s %-7s %-7s",
  "Clndrs", "Volume", "HpX", "Model", "origin",
  "Weight", "Acc", "Mpg", "Score"))

for i = 1, topN do
  local r = sorted_by_score[i]
  print(string.format(
    "%-7s %-7s %-5s %-7s %-7s %-7s %-7s %-7s %.3f",
    r["Clndrs"], r["Volume"], r["HpX"], r["Model"],
    r["origin"], r["Lbs-"], r["Acc+"], r["Mpg+"], r.score))
end
