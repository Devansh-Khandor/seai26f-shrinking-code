#!/usr/bin/env lua
-- Skill 01: Inventory summary and best cars
-- Usage: lua skill_01.lua <csv path>

local ezr = require("ezr")
local round = ezr.round
local csv = ezr.csv
local shuffle = ezr.shuffle
local keysort = ezr.keysort

-- Helper: compute mean, std, min, max for a list of numbers
local function stats(values)
  local n, sum, sumsq = 0, 0, 0
  local minv, maxv = math.huge, -math.huge
  for _, v in ipairs(values) do
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

-- Load CSV
local path = arg[1]
if not path then
  io.stderr:write("Usage: lua skill_01.lua <csv path>\n")
  os.exit(1)
end

local rows_iter = csv(path)
-- Read header
local headers = rows_iter()
if not headers then
  io.stderr:write("Empty CSV\n")
  os.exit(1)
end

-- Read all rows and convert to table keyed by header names
local rows = {}
local weight_vals, mpg_vals, acc_vals = {}, {}, {}

for r in rows_iter do
  local t = {}
  for i, v in ipairs(r) do
    local key = headers[i]
    t[key] = v
    t[i] = v
  end
  table.insert(rows, t)
  table.insert(weight_vals, t["Lbs-"])
  table.insert(mpg_vals, t["Mpg+"])
  table.insert(acc_vals, t["Acc+"])
end

-- CAR-01: summary of inventory
print("== CAR-01 ==")
local wmean, wstd, wmin, wmax = stats(weight_vals)
local mmean, mstd, mmin, mmax = stats(mpg_vals)
local anmean, astd, amin, amax = stats(acc_vals)

print(string.format(
  "Weight: mean=%.2f, std=%.2f, min=%.2f, max=%.2f",
  wmean, wstd, wmin, wmax))
print(string.format(
  "Fuel efficiency: mean=%.2f, std=%.2f, min=%.2f, max=%.2f",
  mmean, mstd, mmin, mmax))
print(string.format(
  "Acceleration: mean=%.2f, std=%.2f, min=%.2f, max=%.2f",
  anmean, astd, amin, amax))

-- CAR-02: best overall combination
print("\n== CAR-02 ==")

-- Normalized ranges for scoring
local function norm(v, minv, maxv, reverse)
  if maxv == minv then return 0 end
  local n = (v - minv) / (maxv - minv)
  return reverse and (1 - n) or n
end

for _, r in ipairs(rows) do
  local w = r["Lbs-"]
  local m = r["Mpg+"]
  local a = r["Acc+"]
  r.score = norm(w, wmin, wmax, true) +
            norm(m, mmin, mmax, false) +
            norm(a, amin, amax, false)
end

-- Sort descending by score
table.sort(rows, function(a, b) return a.score > b.score end)

-- Header for detailed list
print(string.format(
  "%-7s %-7s %-5s %-7s %-7s %-7s %-7s %-7s %-7s",
  "Clndrs", "Volume", "HpX", "Model", "origin",
  "Weight", "Acc", "Mpg", "Score"))

local top = math.min(5, #rows)
for i = 1, top do
  local r = rows[i]
  print(string.format(
    "%-7s %-7s %-5s %-7s %-7s %-7s %-7s %-7s %.2f",
    r["Clndrs"], r["Volume"], r["HpX"], r["Model"],
    r["origin"], r["Lbs-"], r["Acc+"], r["Mpg+"], r.score))
end
