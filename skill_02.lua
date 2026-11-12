#!/usr/bin/env lua
--  skill_02.lua
--
--  Implements the two Apache request types:
--
--  APACHE-03 – Predict the performance of every
--  configuration in the CSV.
--
--  APACHE-04 – Find the first pair of historical
--  configurations that differ by a single setting,
--  predict their performance, and provide an
--  uncertainty estimate for the predicted difference.
--
--  Usage:
--      lua skill_02.lua <csv_path>
--
--  Output:
--      Lines prefixed by "== APACHE-03 ==" and
--      "== APACHE-04 ==" followed by numeric results.

local ezr = require"ezr"

local csvpath = arg[1]
if not csvpath then
  io.stderr:write("usage: lua skill_02.lua <csv_path>\n")
  os.exit(1)
end

-- Load the data
local src = ezr.csv(csvpath)
local tbl = ezr.Tbl(src)

-- Train a tree regressor
local tree = ezr.Tree(tbl, tbl.rows)

-- Find the performance column (the goal)
local perf_col = tbl.cols.y[1]   -- first goal column
local n = #tbl.rows

-- Compute predictions for every row and residuals
local preds = {}
local residuals = {}
for i, row in ipairs(tbl.rows) do
  local pred = tree:leaf(tbl, row)
  local actual = row[perf_col.at]
  preds[i] = pred
  residuals[i] = actual - pred
end

-- Standard error of the model (root‑mean‑square residual)
local sumsq = 0
for _, r in ipairs(residuals) do
  sumsq = sumsq + r * r
end
local sigma = math.sqrt(sumsq / n)

-- ---------- APACHE-03 ----------
print("== APACHE-03 ==")
for i = 1, 3 do
  local row = tbl.rows[i]
  local actual = row[perf_col.at]
  print(string.format("Config %d: actual=%.2f predicted=%.2f",
          i, actual, preds[i]))
end

-- ---------- APACHE-04 ----------
-- Helper: two rows differ by exactly one setting
local function differ_by_one(r1, r2)
  local diff = 0
  for _, c in ipairs(tbl.cols.x) do
    if r1[c.at] ~= r2[c.at] then
      diff = diff + 1
      if diff > 1 then return false end
    end
  end
  return diff == 1
end

local i1, i2
for a = 1, n - 1 do
  for b = a + 1, n do
    if differ_by_one(tbl.rows[a], tbl.rows[b]) then
      i1, i2 = a, b
      break
    end
  end
  if i1 then break end
end

print("\n== APACHE-04 ==")
if i1 then
  local row1, row2 = tbl.rows[i1], tbl.rows[i2]
  local pred1, pred2 = preds[i1], preds[i2]
  local diff = pred2 - pred1
  local uncert = math.sqrt(2) * sigma
  print(string.format(
      "Config %d vs %d: predicted difference=%.2f ± %.2f",
      i1, i2, diff, uncert))
  print(string.format("  Config %d predicted=%.2f actual=%.2f",
      i1, pred1, row1[perf_col.at]))
  print(string.format("  Config %d predicted=%.2f actual=%.2f",
      i2, pred2, row2[perf_col.at]))
else
  print("No pair differing by one setting found.")
end
