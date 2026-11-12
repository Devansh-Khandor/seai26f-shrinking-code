#!/usr/bin/env lua
--  skill_01.lua – inventory summary & multi‑criteria ranking
--  Usage: lua skill_01.lua <csv_path>
--
--  Depends on the ezr library (ezr.lua & ezr-lib.lua) in the same
--  directory or in the Lua package path.

local ezr = require("ezr")

-- ------------------------------------------------------------------
-- Helpers
-- ------------------------------------------------------------------
local function find_best_worst(col, rows, comp)
  local best, worst
  for _, r in ipairs(rows) do
    local v = r[col.at]
    if v == "?" then goto continue end
    if not best or comp(v, best) then best = v end
    if not worst or comp(best, v) then worst = v end
    ::continue::
  end
  return best, worst
end

local function norm_score(col, r, maximize)
  local v = r[col.at]
  if v == "?" then return 0.5 end        -- neutral if unknown
  local n = col:norm(v)
  return maximize and n or (1 - n)
end

-- ------------------------------------------------------------------
-- Load data
-- ------------------------------------------------------------------
local csv_path = arg[1]
if not csv_path then
  io.stderr:write("usage: lua skill_01.lua <csv_path>\n")
  os.exit(1)
end

local tbl = ezr.Tbl(ezr.csv(csv_path))

-- ------------------------------------------------------------------
-- Identify the columns of interest
-- ------------------------------------------------------------------
local Lbs_col  = tbl.cols.all[5]  -- "Lbs-"
local Acc_col  = tbl.cols.all[7]  -- "Acc+"
local Mpg_col  = tbl.cols.all[8]  -- "Mpg+"

-- ------------------------------------------------------------------
-- CAR-01: Summary statistics
-- ------------------------------------------------------------------
print("== CAR-01 ==")
local rows = tbl.rows

-- Weight (minimise)
local w_best, w_worst = find_best_worst(Lbs_col, rows, function(a,b) return a < b end)
print(string.format("Weight  : avg = %s, best = %s, worst = %s",
  ezr.show(Lbs_col:mid()), ezr.show(w_best), ezr.show(w_worst)))

-- Fuel efficiency (maximise)
local mpg_best, mpg_worst = find_best_worst(Mpg_col, rows, function(a,b) return a > b end)
print(string.format("Fuel    : avg = %s, best = %s, worst = %s",
  ezr.show(Mpg_col:mid()), ezr.show(mpg_best), ezr.show(mpg_worst)))

-- Acceleration (maximise)
local acc_best, acc_worst = find_best_worst(Acc_col, rows, function(a,b) return a > b end)
print(string.format("Accel   : avg = %s, best = %s, worst = %s",
  ezr.show(Acc_col:mid()), ezr.show(acc_best), ezr.show(acc_worst)))

-- ------------------------------------------------------------------
-- CAR-02: Multi‑criteria ranking
-- ------------------------------------------------------------------
print("\n== CAR-02 ==")
-- Compute composite score: (1‑norm(weight) + norm(acc) + norm(mpg)) / 3
local scored = {}
for _, r in ipairs(rows) do
  local score = (norm_score(Lbs_col, r, false) +
                 norm_score(Acc_col, r, true) +
                 norm_score(Mpg_col, r, true)) / 3
  table.insert(scored, {row = r, score = score})
end

-- Sort descending by score
table.sort(scored, function(a,b) return a.score > b.score end)

-- Show top 5 (or fewer)
local top_n = math.min(5, #scored)
print(string.format("Top %d vehicles by combined score:", top_n))
print(string.format("%-5s %-6s %-7s %-4s %-4s %-4s %-4s %-4s",
  "Clndrs","Vol","HpX","Model","Origin","Lbs-","Acc+","Mpg+"))
for i = 1, top_n do
  local r = scored[i].row
  print(string.format("%-5s %-6s %-7s %-4s %-4s %-4s %-4s %-4s",
    ezr.show(r[1]), ezr.show(r[2]), ezr.show(r[3]), ezr.show(r[4]),
    ezr.show(r[5]), ezr.show(r[6]), ezr.show(r[7]), ezr.show(r[8])))
end
