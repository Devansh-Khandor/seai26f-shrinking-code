#!/usr/bin/env lua
-- Skill 01 – Inventory summary and best overall vehicles
-- Usage:  lua skill_01.lua <csv_path>

local ezr = require("ezr")
local round = ezr.round
local show  = ezr.show

-- --------------------------------------------------------------
-- Utility: find a column object by name
local function get_col(tbl, name)
  for _, c in pairs(tbl.cols.all) do
    if c.name == name then return c end
  end
  error("column not found: " .. tostring(name))
end

-- --------------------------------------------------------------
-- Read data
local csv_path = arg[1]
if not csv_path then
  io.stderr:write("Error: missing CSV path\n")
  os.exit(1)
end

local tbl = ezr.Tbl(ezr.csv(csv_path))

-- --------------------------------------------------------------
-- CAR-01: Summary of weight, mpg and acceleration
local weight_col = get_col(tbl, "Lbs-")
local mpg_col    = get_col(tbl, "Mpg+")
local acc_col    = get_col(tbl, "Acc+")

-- Collect min/max for each metric
local wmin, wmax = nil, nil
local mmin, mmax = nil, nil
local amin, amax = nil, nil

for _, row in ipairs(tbl.rows) do
  local w = row[weight_col.at]
  local m = row[mpg_col.at]
  local a = row[acc_col.at]

  if not wmin or w < wmin then wmin = w end
  if not wmax or w > wmax then wmax = w end
  if not mmin or m < mmin then mmin = m end
  if not mmax or m > mmax then mmax = m end
  if not amin or a < amin then amin = a end
  if not amax or a > amax then amax = a end
end

print("== CAR-01 ==")
print(string.format("Weight:   mean=%.1f, best=%.1f, worst=%.1f",
      round(weight_col:mid()), wmax, wmin))
print(string.format("Mpg:      mean=%.2f, best=%.2f, worst=%.2f",
      round(mpg_col:mid()), mmax, mmin))
print(string.format("Accel:    mean=%.2f, best=%.2f, worst=%.2f",
      round(acc_col:mid()), amax, amin))

-- --------------------------------------------------------------
-- CAR-02: Vehicles with best overall combination
-- Compute a composite score: high mpg, low weight, low accel
local function norm(x, xmin, xmax)
  if xmax == xmin then return 0.5 end
  return (x - xmin) / (xmax - xmin)
end

local vehicles = {}
for _, row in ipairs(tbl.rows) do
  local w = row[weight_col.at]
  local m = row[mpg_col.at]
  local a = row[acc_col.at]
  local w_score = 1 - norm(w, wmin, wmax)       -- lower weight better
  local a_score = 1 - norm(a, amin, amax)       -- lower accel better
  local m_score = norm(m, mmin, mmax)           -- higher mpg better
  local score = w_score + a_score + m_score
  table.insert(vehicles, {row=row, score=score})
end

-- Sort descending by score
table.sort(vehicles, function(a,b) return a.score > b.score end)

print("\n== CAR-02 ==")
print("Top vehicles for best combination (Weight low, Mpg high, Acc low):")
local top_n = 5
for i=1, math.min(top_n, #vehicles) do
  local v = vehicles[i]
  -- Show key fields and score
  local row = v.row
  print(string.format(
        "%d) Lbs-=%d, Acc+=%.1f, Mpg+=%.2f  Score=%.2f",
        i, row[weight_col.at], row[acc_col.at], row[mpg_col.at], round(v.score,2)))
end
