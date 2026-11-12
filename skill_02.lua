#!/usr/bin/env lua
-- Skill 02 – Estimating missing fuel‑efficiency and weight
-- Usage:  lua skill_02.lua <csv_path>

local ezr   = require("ezr")
local round = ezr.round

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
-- Column objects
local weight_col = get_col(tbl, "Lbs-")
local mpg_col    = get_col(tbl, "Mpg+")
local model_col  = get_col(tbl, "Model")

-- --------------------------------------------------------------
-- CAR-03: Predict missing MPG
local missing_mpg_rows = {}
local candidate_mpg_rows = {}
for _, row in ipairs(tbl.rows) do
  if row[mpg_col.at] == "?" then
    table.insert(missing_mpg_rows, row)
  else
    table.insert(candidate_mpg_rows, row)
  end
end

local function predict(row, target_col, candidates, k)
  local dlist = {}
  for _, cand in ipairs(candidates) do
    local d = tbl:distx(row, cand)
    table.insert(dlist, {d = d, row = cand})
  end
  table.sort(dlist, function(a, b) return a.d < b.d end)
  local sum, wsum = 0, 0
  for i = 1, math.min(k, #dlist) do
    local d = dlist[i].d
    if d == 0 then d = 1e-9 end
    local weight = 1 / d
    local val = candidate_mpg_rows[i][target_col.at]
    sum  = sum + weight * val
    wsum = wsum + weight
  end
  return sum / wsum
end

local mpg_predictions = {}
for _, row in ipairs(missing_mpg_rows) do
  table.insert(mpg_predictions, {row = row,
                                 pred = predict(row, mpg_col, candidate_mpg_rows, 5)})
end

print("== CAR-03 ==")
print(string.format("Missing MPG rows: %d", #missing_mpg_rows))
for i, rec in ipairs(mpg_predictions) do
  if i > 3 then break end
  local r = rec.row
  print(string.format("%d) Model=%s, Predicted MPG=%.2f",
        i, r[model_col.at], round(rec.pred, 2)))
end

-- --------------------------------------------------------------
-- CAR-04: Predict missing Weight with uncertainty
local missing_weight_rows = {}
local candidate_weight_rows = {}
for _, row in ipairs(tbl.rows) do
  if row[weight_col.at] == "?" then
    table.insert(missing_weight_rows, row)
  else
    table.insert(candidate_weight_rows, row)
  end
end

local function predict_with_uncertainty(row, target_col, candidates, k)
  local dlist = {}
  for _, cand in ipairs(candidates) do
    local d = tbl:distx(row, cand)
    table.insert(dlist, {d = d, row = cand})
  end
  table.sort(dlist, function(a, b) return a.d < b.d end)
  local neighbors = {}
  for i = 1, math.min(k, #dlist) do
    table.insert(neighbors, dlist[i].row)
  end
  local sum, sum2, n = 0, 0, 0
  for _, cand in ipairs(neighbors) do
    local val = cand[target_col.at]
    sum  = sum + val
    sum2 = sum2 + val * val
    n = n + 1
  end
  local mean = sum / n
  local variance = (sum2 - sum * sum / n) / (n - 1)
  local stddev = math.sqrt(math.max(variance, 0))
  return mean, stddev
end

local weight_predictions = {}
for _, row in ipairs(missing_weight_rows) do
  local mean, std = predict_with_uncertainty(row, weight_col, candidate_weight_rows, 5)
  table.insert(weight_predictions, {row = row, pred = mean, unc = std})
end

print("\n== CAR-04 ==")
print(string.format("Missing Weight rows: %d", #missing_weight_rows))
for i, rec in ipairs(weight_predictions) do
  if i > 3 then break end
  local r = rec.row
  print(string.format("%d) Model=%s, Predicted Weight=%d ± %.1f",
        i, r[model_col.at], round(rec.pred), round(rec.unc, 1)))
end
