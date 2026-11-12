#!/usr/bin/env lua
-- Skill 07: CAR-13 & CAR-14
-- Usage: lua skill_07.lua <csv path>

local ezr = require("ezr")
local csv  = ezr.csv
local round = ezr.round
local sorted = ezr.sorted

-- ------------------------------------------------------------
-- Helper: normalise a value between min and max
local function norm(v, mn, mx)
  if mx == mn then return 0 end
  return (v - mn) / (mx - mn)
end

-- ------------------------------------------------------------
-- Load data
local path = arg[1]
if not path then
  io.stderr:write("Usage: lua skill_07.lua <csv path>\n")
  os.exit(1)
end

local rows_iter = csv(path)
local headers = rows_iter()
if not headers then
  io.stderr:write("Empty CSV\n")
  os.exit(1)
end

local rows = {}
while true do
  local r = rows_iter()
  if not r then break end
  table.insert(rows, r)
end

-- ------------------------------------------------------------
-- Find min/max of the three metrics we care about
local w_min, w_max = math.huge, -math.huge
local mpg_min, mpg_max = math.huge, -math.huge
local acc_min, acc_max = math.huge, -math.huge

for _, r in ipairs(rows) do
  local w = r["Lbs-"]
  local m = r["Mpg+"]
  local a = r["Acc+"]
  if w and w < w_min then w_min = w end
  if w and w > w_max then w_max = w end
  if m and m < mpg_min then mpg_min = m end
  if m and m > mpg_max then mpg_max = m end
  if a and a < acc_min then acc_min = a end
  if a and a > acc_max then acc_max = a end
end

-- ------------------------------------------------------------
-- Compute composite score for each row
for _, r in ipairs(rows) do
  local w = r["Lbs-"]
  local m = r["Mpg+"]
  local a = r["Acc+"]
  -- weight is better when lower → reverse
  local w_score = 1 - norm(w, w_min, w_max)
  local m_score = norm(m, mpg_min, mpg_max)
  local a_score = norm(a, acc_min, acc_max)
  r._score = w_score + m_score + a_score   -- 0 .. 3
end

-- ------------------------------------------------------------
-- CAR-13: Vehicle with best balance
print("== CAR-13 ==")
-- Find best row
local best = rows[1]
for _, r in ipairs(rows) do
  if r._score > best._score then best = r end
end

print(string.format("Best balanced vehicle:"))
print(string.format("  Clndrs = %s", tostring(best["Clndrs"])))
print(string.format("  Volume = %s", tostring(best["Volume"])))
print(string.format("  HpX    = %s", tostring(best["HpX"])))
print(string.format("  Model  = %s", tostring(best["Model"])))
print(string.format("  Origin = %s", tostring(best["origin"])))
print(string.format("  Weight = %s", tostring(best["Lbs-"])))
print(string.format("  Acc    = %s", tostring(best["Acc+"])))
print(string.format("  MPG    = %s", tostring(best["Mpg+"])))
print(string.format("  Score  = %.3f", best._score))

-- ------------------------------------------------------------
-- CAR-14: Recommendation order with limited inspection budget
print("\n== CAR-14 ==")
-- default budget: 5 or all rows if fewer
local budget = math.min(#rows, 5)

-- Sort rows by score descending
local ordered = sorted(rows, function(a,b) return a._score > b._score end)

print(string.format("Top %d vehicles to inspect first (sorted by score):", budget))
print(string.format("%-5s %-5s %-5s %-7s %-7s %-7s %-7s %-7s %6s",
  "Idx","Clndrs","Volume","HpX","Model","Origin","Wgt","Acc","Score"))
for i = 1, budget do
  local r = ordered[i]
  print(string.format("%-5d %-5s %-5s %-7s %-7s %-7s %-7s %-7s %6.3f",
    i,
    tostring(r["Clndrs"]),
    tostring(r["Volume"]),
    tostring(r["HpX"]),
    tostring(r["Model"]),
    tostring(r["origin"]),
    tostring(r["Lbs-"]),
    tostring(r["Acc+"]),
    r._score))
end
