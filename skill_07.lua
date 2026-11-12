#!/usr/bin/env lua
-- Skill 07: Balanced vehicle selection & inspection prioritisation
-- Usage: lua skill_07.lua <csv path>

local ezr = require("ezr")
local show, round, keysort = ezr.show, ezr.round, ezr.keysort

-- -----------------------------------------------------------------
-- Helper functions (copied from skill_04 for reuse)
-- -----------------------------------------------------------------
local function col_by_name(cols, name)
  for _,c in ipairs(cols.all) do
    if c.name == name then return c end
  end
  error("column " .. name .. " not found")
end

local function minmax(col, rows)
  local minv, maxv = math.huge, -math.huge
  for _,r in ipairs(rows) do
    local v = r[col.at]
    if v ~= "?" then
      if v < minv then minv = v end
      if v > maxv then maxv = v end
    end
  end
  return minv, maxv
end

local function norm(v, minv, maxv, reverse)
  if maxv == minv then return 0.5 end
  local n = (v - minv) / (maxv - minv)
  return reverse and 1 - n or n
end

local function score_for_row(row, wmin, wmax, mmin, mmax, amin, amax, wcol, mcol, acol)
  local w = row[wcol.at]
  local m = row[mcol.at]
  local a = row[acol.at]
  if w == "?" or m == "?" or a == "?" then return nil end
  local ws = 1 - norm(w, wmin, wmax)   -- lower weight better
  local ms = norm(m, mmin, mmax)       -- higher mpg better
  local as = 1 - norm(a, amin, amax)   -- lower acceleration better
  return ws + ms + as -- 0 .. 3
end

local function top_indices(rows, scores, n)
  local idx = {}
  for i = 1, #rows do idx[i] = i end
  table.sort(idx, function(i,j)
    local si, sj = scores[i] or -math.huge, scores[j] or -math.huge
    return si > sj
  end)
  local top = {}
  for i = 1, math.min(n, #idx) do
    top[i] = idx[i]
  end
  return top
end

-- -----------------------------------------------------------------
-- Main program
-- -----------------------------------------------------------------
local path = arg[1]
if not path then
  io.stderr:write("Usage: lua skill_07.lua <csv path>\n")
  os.exit(1)
end

-- Load CSV and build table
local tbl = ezr.Tbl(ezr.csv(path))
local cols = tbl.cols

-- Columns used for scoring
local wcol = col_by_name(cols, "Lbs-")
local mcol = col_by_name(cols, "Mpg+")
local acol = col_by_name(cols, "Acc+")
local model_col = col_by_name(cols, "Model")

-- Compute global min/max for normalisation
local wmin, wmax = minmax(wcol, tbl.rows)
local mmin, mmax = minmax(mcol, tbl.rows)
local amin, amax = minmax(acol, tbl.rows)

-- Compute scores for all rows
local scores = {}
for i,r in ipairs(tbl.rows) do
  scores[i] = score_for_row(r, wmin, wmax, mmin, mmax, amin, amax, wcol, mcol, acol)
end

-- -----------------------------------------------------------------
-- CAR-13 : Balanced vehicle selection
-- -----------------------------------------------------------------
print("== CAR-13 ==")

-- Find the row with the highest score
local best_i = 1
for i=2,#tbl.rows do
  if scores[i] and scores[i] > (scores[best_i] or -math.huge) then
    best_i = i
  end
end
local best_row = tbl.rows[best_i]
local best_score = scores[best_i] or 0

print(string.format(
  "Best balanced vehicle: Model=%s, Weight=%.0f, MPG=%.1f, Acc=%.1f, Score=%.2f",
  tostring(best_row[model_col.at]),
  best_row[wcol.at], best_row[mcol.at], best_row[acol.at], round(best_score,2)
))
print(string.format(
  "Min/Max ranges: Weight[%.0f..%.0f], MPG[%.1f..%.1f], Acc[%.1f..%.1f]",
  wmin, wmax, mmin, mmax, amin, amax
))

-- -----------------------------------------------------------------
-- CAR-14 : Inspection priority list
-- -----------------------------------------------------------------
print("\n== CAR-14 ==")

-- Decide how many vehicles to recommend (default 5)
local budget = 5
local top_ids = top_indices(tbl.rows, scores, budget)

print(string.format("%-10s %-6s %-6s %-6s %-6s",
  "Model","Weight","MPG","Acc","Score"))
for _,i in ipairs(top_ids) do
  local r = tbl.rows[i]
  local s = scores[i] or 0
  print(string.format("%-10s %-6s %-6s %-6s %-6.2f",
        tostring(r[model_col.at]), tostring(r[wcol.at]),
        tostring(r[mcol.at]), tostring(r[acol.at]), round(s,2)))
end
