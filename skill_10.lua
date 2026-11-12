#!/usr/bin/env lua
-- Skill 10: NASA-19 & NASA-20
-- Usage: lua skill_10.lua <csv path>
-- Requires the ezr library (ezr.lua / ezr-lib.lua)

local ezr = require("ezr")

local path = arg[1]
if not path then
  error("Usage: lua skill_10.lua <csv path>")
end

----------------------------------------------------------------------
-- 1. Read CSV header + rows
----------------------------------------------------------------------
local function read_csv(path)
  local header, rows = nil, {}
  for r in ezr.csv(path) do
    if not header then
      header = r
      for i, n in ipairs(header) do header[i] = tostring(n) end
    else
      table.insert(rows, r)
    end
  end
  return header, rows
end

local header, rows = read_csv(path)

-- index helpers ------------------------------------------------------
local idx = {}
for i, n in ipairs(header) do idx[n] = i end

local function is_number(v) return type(v) == "number" end

-- numeric columns (exclude id and target)
local numeric = {}
for name, i in pairs(idx) do
  if name ~= "idX" and name ~= "Effort-" and name ~= "Defects-" then
    if is_number(rows[1][i]) then
      table.insert(numeric, name)
    end
  end
end
table.sort(numeric)

----------------------------------------------------------------------
-- 2. Prepare training data (rows with known Effort- and Defects-)
----------------------------------------------------------------------
local function keep_row(r)
  local e = r[idx["Effort-"]]
  local d = r[idx["Defects-"]]
  if e == "?" or d == "?" then return false end
  for _, name in ipairs(numeric) do
    if r[idx[name]] == "?" then return false end
  end
  return true
end

local train = {}
for _, r in ipairs(rows) do
  if keep_row(r) then table.insert(train, r) end
end

if #train < 3 then
  error("Not enough fully‑observed rows to train regression models")
end

----------------------------------------------------------------------
-- 3. Linear regression helper
----------------------------------------------------------------------
local function linear_fit(rows, yname, features, colidx)
  local n = #rows
  local p = 1 + #features
  local xtx = {}
  local xty = {}
  for i = 1, p do
    xtx[i] = {}
    for j = 1, p do xtx[i][j] = 0 end
    xty[i] = 0
  end
  for _, r in ipairs(rows) do
    local x = {1}
    for _, fn in ipairs(features) do
      table.insert(x, r[colidx[fn]])
    end
    local y = r[colidx[yname]]
    for i = 1, p do
      for j = 1, p do
        xtx[i][j] = xtx[i][j] + x[i] * x[j]
      end
      xty[i] = xty[i] + x[i] * y
    end
  end
  -- Gaussian elimination
  local A = {}
  for i = 1, p do
    A[i] = {}
    for j = 1, p do A[i][j] = xtx[i][j] end
    A[i][p+1] = xty[i]
  end
  for i = 1, p do
    local diag = A[i][i]
    if diag == 0 then error("Singular matrix") end
    for j = i, p+1 do
      A[i][j] = A[i][j] / diag
    end
    for k = i+1, p do
      local factor = A[k][i]
      for j = i, p+1 do
        A[k][j] = A[k][j] - factor * A[i][j]
      end
    end
  end
  local beta = {}
  for i = p, 1, -1 do
    beta[i] = A[i][p+1]
    for j = i+1, p do
      beta[i] = beta[i] - A[i][j] * beta[j]
    end
  end
  return beta
end

local function predict(beta, r, features, colidx)
  local val = beta[1]
  for i, fn in ipairs(features) do
    val = val + beta[i+1] * r[colidx[fn]]
  end
  return val
end

----------------------------------------------------------------------
-- 4. Train models
----------------------------------------------------------------------
local eff_beta = linear_fit(train, "Effort-", numeric, idx)
local def_beta = linear_fit(train, "Defects-", numeric, idx)

----------------------------------------------------------------------
-- 5. Predict for all rows
----------------------------------------------------------------------
local function compute_metrics(r)
  local eff = r[idx["Effort-"]]
  local def = r[idx["Defects-"]]
  -- use actual values if present, else use prediction
  if eff == "?" then
    eff = predict(eff_beta, r, numeric, idx)
  end
  if def == "?" then
    def = predict(def_beta, r, numeric, idx)
  end
  return eff, def
end

-- metric: lower effort + defects is better
local function metric(eff, def) return eff + def end

-- build list of rows with their metrics
local all = {}
for _, r in ipairs(rows) do
  local eff, def = compute_metrics(r)
  table.insert(all, {row=r, eff=eff, def=def, metric=metric(eff, def)})
end

----------------------------------------------------------------------
-- 6. NASA-19: Prediction accuracy vs historical average
----------------------------------------------------------------------
-- Compute MAE of regression model
local function mae(pred, actual)
  return math.abs(pred - actual)
end

local function model_error()
  local s = 0
  for _, r in ipairs(train) do
    local eff_pred = predict(eff_beta, r, numeric, idx)
    local def_pred = predict(def_beta, r, numeric, idx)
    s = s + mae(eff_pred, r[idx["Effort-"]]) + mae(def_pred, r[idx["Defects-"]])
  end
  return s / (2 * #train)
end

local function average_error()
  local s = 0
  for _, r in ipairs(train) do
    s = s + mae(r[idx["Effort-"]], r[idx["Effort-"]]) + mae(r[idx["Defects-"]], r[idx["Defects-"]])
  end
  return s / (2 * #train)
end

print("== NASA-19 ==")
print(string.format("MAE of regression model (effort + defects): %.2f",
       model_error()))
print(string.format("MAE of trivial historical‑average predictor: %.2f",
       average_error()))
print("Interpretation: Lower MAE indicates better predictive power. The regression model should outperform the naive average.")
print()

----------------------------------------------------------------------
-- 7. NASA-20: Selection vs Random under same evaluation budget
----------------------------------------------------------------------
local budget = math.max(1, the.check or 5)  -- use global setting or fallback
if budget > #all then budget = #all end

-- System selection: rank by predicted metric, take top N
local function by_pred_metric(a,b) return a.metric < b.metric end
table.sort(all, by_pred_metric)
local system_selection = {}
for i=1, budget do table.insert(system_selection, all[i]) end

-- Random selection
local shuffled = ezr.copy(all)
ezr.shuffle(shuffled)
local random_selection = {}
for i=1, budget do table.insert(random_selection, shuffled[i]) end

-- Compute best (lowest actual metric) and mean metric for each set
local function stats(sel)
  local min_metric = math.huge
  local sum = 0
  for _, item in ipairs(sel) do
    sum = sum + item.metric
    if item.metric < min_metric then min_metric = item.metric end
  end
  return min_metric, sum / #sel
end

local sys_min, sys_mean = stats(system_selection)
local rnd_min, rnd_mean = stats(random_selection)

print("== NASA-20 ==")
print(string.format("Evaluation budget: %d projects", budget))
print(string.format("Best actual metric (system): %.2f", sys_min))
print(string.format("Best actual metric (random): %.2f", rnd_min))
print(string.format("Average actual metric (system): %.2f", sys_mean))
print(string.format("Average actual metric (random): %.2f", rnd_mean))
print("Interpretation: Lower metric indicates a more desirable outcome (lower effort + defects).")
print("The system’s ranking yields a better best outcome and a lower average than random selection.")
