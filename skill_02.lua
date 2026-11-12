#!/usr/bin/env lua
-- Skill 02: NASA-03 & NASA-04
-- Usage: lua skill_02.lua <csv path>
-- Requires the ezr library (ezr.lua / ezr-lib.lua)

local ezr = require("ezr")

local path = arg[1]
if not path then
  error("Usage: lua skill_02.lua <csv path>")
end

-- ------------------------------------------------------------------
-- Helper: read CSV header and data rows
-- ------------------------------------------------------------------
local function read_data(path)
  local rows = {}
  local header = nil
  for r in ezr.csv(path) do
    if not header then
      header = r
      for i, name in ipairs(header) do
        header[i] = tostring(name)
      end
    else
      table.insert(rows, r)
    end
  end
  return header, rows
end

-- ------------------------------------------------------------------
-- Helper: get numeric columns (by checking first data row)
-- ------------------------------------------------------------------
local function numeric_columns(header, rows)
  local nums = {}
  local first = rows[1]
  for i, name in ipairs(header) do
    local val = first[i]
    if type(val) == "number" then
      nums[name] = i
    end
  end
  return nums
end

-- ------------------------------------------------------------------
-- Linear regression: X matrix (with intercept), y vector
-- ------------------------------------------------------------------
local function linear_fit(rows, yname, features, colidx)
  local n = #rows
  local p = 1 + #features          -- intercept + predictors
  -- Build X'X matrix and X'y vector
  local xtx = {}
  local xty = {}
  for i = 1, p do
    xtx[i] = {}
    for j = 1, p do xtx[i][j] = 0 end
    xty[i] = 0
  end
  for _, row in ipairs(rows) do
    local x = {1}                -- intercept
    for _, fn in ipairs(features) do
      x[#x+1] = row[colidx[fn]]
    end
    local y = row[colidx[yname]]
    for i = 1, p do
      for j = 1, p do
        xtx[i][j] = xtx[i][j] + x[i] * x[j]
      end
      xty[i] = xty[i] + x[i] * y
    end
  end
  -- Solve (X'X) * beta = X'y  via Gaussian elimination
  local beta = {}
  -- Augmented matrix
  local A = {}
  for i = 1, p do
    A[i] = {}
    for j = 1, p do A[i][j] = xtx[i][j] end
    A[i][p+1] = xty[i]
  end
  -- Forward elimination
  for i = 1, p do
    local diag = A[i][i]
    if diag == 0 then
      error("Singular matrix")
    end
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
  -- Back substitution
  for i = p, 1, -1 do
    beta[i] = A[i][p+1]
    for j = i+1, p do
      beta[i] = beta[i] - A[i][j] * beta[j]
    end
  end
  return beta
end

-- ------------------------------------------------------------------
-- Predict using coefficients
-- ------------------------------------------------------------------
local function predict(beta, row, features, colidx)
  local val = beta[1]          -- intercept
  for i = 1, #features do
    val = val + beta[i+1] * row[colidx[features[i]]]
  end
  return val
end

-- ------------------------------------------------------------------
-- Main analysis
-- ------------------------------------------------------------------
local header, rows = read_data(path)
local colidx = {}
for i, name in ipairs(header) do colidx[name] = i end

-- Identify numeric columns
local numeric = numeric_columns(header, rows)

-- Remove target columns from feature set
local effort_features = {}
for name, idx in pairs(numeric) do
  if name ~= "Effort-" then table.insert(effort_features, name) end
end
table.sort(effort_features)

local defect_features = {}
for name, idx in pairs(numeric) do
  if name ~= "Defects-" then table.insert(defect_features, name) end
end
table.sort(defect_features)

-- Fit models
local effort_beta = linear_fit(rows, "Effort-", effort_features, colidx)
local defect_beta = linear_fit(rows, "Defects-", defect_features, colidx)

-- Compute predictions and residuals
local effort_preds = {}
local defect_preds = {}
local effort_resid = {}
local defect_resid = {}

for _, row in ipairs(rows) do
  local e_pred = predict(effort_beta, row, effort_features, colidx)
  local d_pred = predict(defect_beta, row, defect_features, colidx)
  table.insert(effort_preds, e_pred)
  table.insert(defect_preds, d_pred)
  table.insert(effort_resid, row[colidx["Effort-"]] - e_pred)
  table.insert(defect_resid, row[colidx["Defects-"]] - d_pred)
end

-- Helper stats
local function avg(t)
  local s = 0
  for _, v in ipairs(t) do s = s + v end
  return s / #t
end
local function std(t, mean)
  local ss = 0
  for _, v in ipairs(t) do ss = ss + (v - mean)^2 end
  return math.sqrt(ss / (#t - 1))
end
local function minmax(t)
  local mi, ma = t[1], t[1]
  for _, v in ipairs(t) do
    if v < mi then mi = v end
    if v > ma then ma = v end
  end
  return mi, ma
end

-- Effort stats
local avgEffPred = avg(effort_preds)
local sigmaEff = std(effort_resid, 0)
local minEffPred, maxEffPred = minmax(effort_preds)

-- Defect stats
local avgDefPred = avg(defect_preds)
local sigmaDef = std(defect_resid, 0)
local minDefPred, maxDefPred = minmax(defect_preds)

-- ------------------------------------------------------------------
-- Output
-- ------------------------------------------------------------------
print("== NASA-03 ==")
print(string.format("Average predicted effort: %s",
      ezr.round(avgEffPred)))
print(string.format("Prediction standard error: %s",
      ezr.round(sigmaEff)))
print(string.format("Min / Max predicted effort: %s / %s",
      ezr.round(minEffPred), ezr.round(maxEffPred)))

print("\n== NASA-04 ==")
print(string.format("Average predicted defects: %s",
      ezr.round(avgDefPred)))
print(string.format("Prediction interval (±2σ): %s to %s",
      ezr.round(avgDefPred - 2*sigmaDef),
      ezr.round(avgDefPred + 2*sigmaDef)))
print(string.format("Min / Max predicted defects: %s / %s",
      ezr.round(minDefPred), ezr.round(maxDefPred)))
