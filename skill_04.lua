#!/usr/bin/env lua
-- Skill 04 – Streaming inventory assessment and live summaries
-- Usage:  lua skill_04.lua <csv_path>

local ezr = require("ezr")
local round = ezr.round

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
local model_col  = get_col(tbl, "Model")   -- for printing

-- ------------------------------------------------------------------
-- Normalisation helper (0..1)
local function norm(x, xmin, xmax)
  if xmax == xmin then return 0.5 end
  return (x - xmin) / (xmax - xmin)
end

-- ------------------------------------------------------------------
-- State that evolves as rows arrive
local minW, maxW = math.huge, -math.huge
local minM, maxM = math.huge, -math.huge
local minA, maxA = math.huge, -math.huge

local scores = {}          -- list of all scores computed so far
local rows   = {}          -- list of {row, score}

-- ------------------------------------------------------------------
-- Process rows one by one, as if they arrive
for _, r in ipairs(tbl.rows) do
  local w = r[weight_col.at]
  local m = r[mpg_col.at]
  local a = r[acc_col.at]

  -- Update min/max
  if w < minW then minW = w end
  if w > maxW then maxW = w end
  if m < minM then minM = m end
  if m > maxM then maxM = m end
  if a < minA then minA = a end
  if a > maxA then maxA = a end

  -- Compute composite score for this vehicle
  local score = (1 - norm(w, minW, maxW))          -- lower weight
               + (1 - norm(a, minA, maxA))          -- lower acceleration
               +  norm(m, minM, maxM)               -- higher mpg

  -- Store
  table.insert(scores, score)
  table.insert(rows, {row = r, score = score})

  -- Determine desirability relative to the median so far
  local sorted = ezr.sorted(scores)
  local median = sorted[math.floor(#sorted/2) + 1]
  local desirable = score >= median
end

-- ------------------------------------------------------------------
-- CAR-07: assessment of each arriving vehicle
print("== CAR-07 ==")
for i, rec in ipairs(rows) do
  local r = rec.row
  local des = rec.score >= ezr.sorted(rec.score) and "Yes" or "No"  -- placeholder
  -- We actually want median at the time of arrival; compute again
  local sorted = ezr.sorted(scores[1], #scores[i])  -- not possible, we recompute each time
end

-- Since we already processed all rows, recompute median per row
local medians = {}
local tmp = {}
for i=1,#scores do
  tmp[i] = scores[i]
  table.sort(tmp, function(a,b) return a<b end)
  medians[i] = tmp[math.floor(#tmp/2)+1]
end

for i, rec in ipairs(rows) do
  local r = rec.row
  local des = rec.score >= medians[i] and "Yes" or "No"
  print(string.format("%d) Model=%s, Score=%.2f, Desirable=%s",
          i, r[model_col.at], round(rec.score,2), des))
end

-- ------------------------------------------------------------------
-- CAR-08: live inventory summaries and top vehicles
print("\n== CAR-08 ==")

-- Summaries
print("Inventory summaries:")
print(string.format("Weight: mean=%.1f, best=%.1f, worst=%.1f",
          round(weight_col:mid()), minW, maxW))
print(string.format("Mpg:    mean=%.2f, best=%.2f, worst=%.2f",
          round(mpg_col:mid()), maxM, minM))
print(string.format("Accel:  mean=%.2f, best=%.2f, worst=%.2f",
          round(acc_col:mid()), minA, maxA))

-- Top vehicles by score
table.sort(rows, function(a,b) return a.score > b.score end)
local topN = math.min(5, #rows)
print("\nTop vehicles by composite score:")
for i=1,topN do
  local rec = rows[i]
  local r = rec.row
  print(string.format("%d) Model=%s, Score=%.2f, Weight=%d, Acc=%.2f, Mpg=%.2f",
          i, r[model_col.at], round(rec.score,2), r[weight_col.at],
          r[acc_col.at], r[mpg_col.at]))
end
