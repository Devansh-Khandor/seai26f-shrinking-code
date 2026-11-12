#!/usr/bin/env lua
-- Skill 10 – Evaluation of active‑learning selection and prediction
-- Usage:  lua skill_10.lua <csv_path>

local ezr   = require("ezr")
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
-- Normalisation helper (0..1)
local function norm(x, xmin, xmax)
  if xmax == xmin then return 0.5 end
  return (x - xmin) / (xmax - xmin)
end

-- ------------------------------------------------------------------
-- Composite performance score: lower weight, lower acceleration,
-- higher mpg
local function comp_score(row, minW, maxW, minA, maxA, minM, maxM)
  local w = row[weight_col.at]
  local a = row[acc_col.at]
  local m = row[mpg_col.at]
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
-- Global min/max for normalisation (used for composite score)
local minW, maxW = math.huge, -math.huge
local minA, maxA = math.huge, -math.huge
local minM, maxM = math.huge, -math.huge

for _, row in ipairs(tbl.rows) do
  local w = row[weight_col.at]
  local a = row[acc_col.at]
  local m = row[mpg_col.at]
  if type(w) == "number" then
    minW = math.min(minW, w)
    maxW = math.max(maxW, w)
  end
  if type(a) == "number" then
    minA = math.min(minA, a)
    maxA = math.max(maxA, a)
  end
  if type(m) == "number" then
    minM = math.min(minM, m)
    maxM = math.max(maxM, m)
  end
end

-- ------------------------------------------------------------------
-- Budget for active learning (10% of data or at least 1)
local n_rows = #tbl.rows
local cap = math.max(1, math.floor(n_rows * 0.10))
if cap > n_rows then cap = n_rows end

-----------------------------------------------------------------------
-- CAR-19: Compare active‑learning selection vs random
print("== CAR-19 ==")

-- Acquire a small, representative subset (sorted by dY)
local active_rows = tbl:acquirer(cap)   -- sorted best-first by dY

-- Filter to rows that have a known MPG (used for comparison)
local function keep_known_mpg(rows)
  local out = {}
  for _, r in ipairs(rows) do
    if r[mpg_col.at] ~= "?" then
      out[#out + 1] = r
    end
  end
  return out
end

local active_known = keep_known_mpg(active_rows)
local n_active = #active_known

-- Random selection with the same number of evaluations
local shuffled = ezr.shuffle(tbl.rows)
local random_known = {}
for _, r in ipairs(shuffled) do
  if r[mpg_col.at] ~= "?" then
    random_known[#random_known + 1] = r
    if #random_known >= n_active then break end
  end
end
local n_random = #random_known

-- Helper to compute mean of a numeric field
local function mean(field, rows)
  local s, n = 0, 0
  for _, r in ipairs(rows) do
    local v = r[field.at]
    if type(v) == "number" then
      s = s + v
      n = n + 1
    end
  end
  return n > 0 and s / n or nil
end

local function mean_score(rows)
  local s, n = 0, 0
  for _, r in ipairs(rows) do
    local sc = comp_score(r, minW, maxW, minA, maxA, minM, maxM)
    s = s + sc
    n = n + 1
  end
  return n > 0 and s / n or nil
end

-- Means
local active_mpg = mean(mpg_col, active_known)
local random_mpg = mean(mpg_col, random_known)
local active_sc  = mean_score(active_known)
local random_sc  = mean_score(random_known)

-- Print results
print(string.format(
  "Active‑learning selection (n=%d) mean MPG=%.2f, mean score=%.3f",
  n_active, round(active_mpg,2), round(active_sc,3)))
print(string.format(
  "Random selection (n=%d)   mean MPG=%.2f, mean score=%.3f",
  n_random, round(random_mpg,2), round(random_sc,3)))

-- Percent improvement in MPG
if active_mpg and random_mpg and random_mpg ~= 0 then
  local mpg_impr = (active_mpg - random_mpg) / random_mpg * 100
  print(string.format("MPG improvement: %.1f%%", round(mpg_impr,1)))
end
-- Percent improvement in composite score
if active_sc and random_sc and random_sc ~= 0 then
  local sc_impr = (active_sc - random_sc) / random_sc * 100
  print(string.format("Composite‑score improvement: %.1f%%", round(sc_impr,1)))
end

-----------------------------------------------------------------------
-- CAR-20: Prediction quality vs historical average
print("\n== CAR-20 ==")

-- Training set: active rows with known MPG
local training_rows = keep_known_mpg(active_rows)
if #training_rows == 0 then
  print("No training data with known MPG; cannot build model.")
  os.exit(0)
end

-- Build a regression tree for MPG
local train_tbl = tbl:clone(training_rows)
local t = ezr.Tree(train_tbl, training_rows,
                  function(r) return r[mpg_col.at] end)

-- Set of training row references for quick lookup
local train_set = {}
for _, r in ipairs(training_rows) do train_set[r] = true end

-- Predict on unseen rows with known MPG
local sum_abs_err, sum_sq_err, n_pred = 0, 0, 0
for _, r in ipairs(tbl.rows) do
  if r[mpg_col.at] ~= "?" and not train_set[r] then
    local pred = t:leaf(tbl, r)
    local act  = r[mpg_col.at]
    local err  = act - pred
    sum_abs_err = sum_abs_err + math.abs(err)
    sum_sq_err  = sum_sq_err + err * err
    n_pred = n_pred + 1
  end
end

if n_pred == 0 then
  print("No unseen MPG records to evaluate.")
  os.exit(0)
end

local mae_tree = sum_abs_err / n_pred
local rmse_tree = math.sqrt(sum_sq_err / n_pred)

-- Baseline: historical average from training data
local sum_train = 0
for _, r in ipairs(training_rows) do sum_train = sum_train + r[mpg_col.at] end
local mean_train = sum_train / #training_rows
local mae_base = 0
for _, r in ipairs(tbl.rows) do
  if r[mpg_col.at] ~= "?" and not train_set[r] then
    mae_base = mae_base + math.abs(r[mpg_col.at] - mean_train)
  end
end
mae_base = mae_base / n_pred

-- Percent improvement of MAE
local mae_impr = (mae_base - mae_tree) / mae_base * 100

-- Print results
print(string.format(
  "Number of predictions: %d",
  n_pred))
print(string.format(
  "Model MAE: %.2f, RMSE: %.2f",
  round(mae_tree,2), round(rmse_tree,2)))
print(string.format(
  "Baseline MAE: %.2f",
  round(mae_base,2)))
print(string.format(
  "MAE improvement over baseline: %.1f%%",
  round(mae_impr,1)))
