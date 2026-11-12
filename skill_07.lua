#!/usr/bin/env lua
-- Skill 07 – Balancing fuel efficiency, weight, acceleration with limited inspection
-- and early discovery of promising vehicles.
-- Usage:  lua skill_07.lua <csv_path>

local ezr   = require("ezr")
local round = ezr.round
local sorted= ezr.sorted

-- ------------------------------------------------------------------
-- Helper: find column object by name
local function get_col(tbl, name)
  for _, c in pairs(tbl.cols.all) do
    if c.name == name then return c end
  end
  error("column not found: " .. tostring(name))
end

-- ------------------------------------------------------------------
-- Normalisation helper (0..1)
local function norm(x, xmin, xmax)
  if xmax == xmin then return 0.5 end
  return (x - xmin) / (xmax - xmin)
end

-- ------------------------------------------------------------------
-- Composite score: lower weight, lower acceleration, higher mpg
local function score(row, minW, maxW, minA, maxA, minM, maxM)
  local w = row[1]
  local a = row[2]
  local m = row[3]
  return (1 - norm(w, minW, maxW)) +
         (1 - norm(a, minA, maxA)) +
         norm(m, minM, maxM)
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
local acc_col    = get_col(tbl, "Acc+")
local mpg_col    = get_col(tbl, "Mpg+")
local model_col  = get_col(tbl, "Model")

-- ------------------------------------------------------------------
-- Determine global min/max for normalisation
local minW, maxW = math.huge, -math.huge
local minA, maxA = math.huge, -math.huge
local minM, maxM = math.huge, -math.huge

for _, row in ipairs(tbl.rows) do
  local w = row[weight_col.at]
  local a = row[acc_col.at]
  local m = row[mpg_col.at]
  if w < minW then minW = w end
  if w > maxW then maxW = w end
  if a < minA then minA = a end
  if a > maxA then maxA = a end
  if m < minM then minM = m end
  if m > maxM then maxM = m end
end

-- ------------------------------------------------------------------
-- Budget for active learning (10% of data or at least 1)
local n_rows = #tbl.rows
local cap = math.max(1, math.floor(n_rows * 0.10))
if cap > n_rows then cap = n_rows end

-- ------------------------------------------------------------------
-- CAR-13: Balanced vehicle with few evaluations
print("== CAR-13 ==")

-- Acquire a small, representative subset
local sampled_rows = tbl:acquirer(cap)   -- sorted by dY (best first)
-- Compute scores only for sampled rows
for _, r in ipairs(sampled_rows) do
  r._bal_score = score(r,
        minW, maxW, minA, maxA, minM, maxM)
end
-- Sort by score descending
table.sort(sampled_rows, function(a,b) return a._bal_score > b._bal_score end)

local top_n = math.min(3, #sampled_rows)
for i=1,top_n do
  local r = sampled_rows[i]
  print(string.format(
    "%d) Model=%s | Score=%.2f | Weight=%d | Acc=%.2f | MPG=%.2f",
    i,
    r[model_col.at],
    round(r._bal_score,2),
    r[weight_col.at],
    r[acc_col.at],
    r[mpg_col.at]))
end

-- ------------------------------------------------------------------
-- CAR-14: Prioritise vehicles for inspection
print("\n== CAR-14 ==")

-- Use the same acquisition order as before
for i, r in ipairs(sampled_rows) do
  local dY = tbl:disty(r)
  print(string.format(
    "%d) Model=%s | dY=%.3f | Weight=%d | Acc=%.2f | MPG=%.2f",
    i,
    r[model_col.at],
    round(dY,3),
    r[weight_col.at],
    r[acc_col.at],
    r[mpg_col.at]))
  if i >= 5 then break end   -- show first 5 recommendations
end
