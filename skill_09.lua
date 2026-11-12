#!/usr/bin/env lua
-- Skill 09: NASA-17 & NASA-18
-- Usage: lua skill_09.lua <csv path>
-- Requires the ezr library (ezr.lua / ezr-lib.lua)

local ezr = require("ezr")

local path = arg[1]
if not path then
  error("Usage: lua skill_09.lua <csv path>")
end

-- ------------------------------------------------------------------
-- Load CSV header and all rows
-- ------------------------------------------------------------------
local header, rows = {}, nil
for r in ezr.csv(path) do
  if not header then
    header = r
    for i, n in ipairs(header) do header[i] = tostring(n) end
  else
    table.insert(rows, r)
  end
end
if #rows == 0 then
  error("No data rows in CSV")
end

-- ------------------------------------------------------------------
-- Helpers: identify numeric and symbolic columns
-- ------------------------------------------------------------------
local function get_numeric_cols(header, rows)
  local nums = {}
  local first = rows[1]
  for i, name in ipairs(header) do
    if type(first[i]) == "number" then
      -- exclude idX and outcome columns
      if name ~= "idX" and name ~= "Effort-" and name ~= "Defects-" then
        table.insert(nums, name)
      end
    end
  end
  return nums
end

local function get_symbolic_cols(header, rows)
  local syms = {}
  local first = rows[1]
  for i, name in ipairs(header) do
    if type(first[i]) ~= "number" then
      if name ~= "idX" and name ~= "Effort-" and name ~= "Defects-" then
        table.insert(syms, name)
      end
    end
  end
  return syms
end

local numeric_cols = get_numeric_cols(header, rows)
local symbolic_cols = get_symbolic_cols(header, rows)

-- ------------------------------------------------------------------
-- Compute statistics for numeric columns
-- ------------------------------------------------------------------
local numeric_stats = {}
for _, col in ipairs(numeric_cols) do
  local sum, sumsq = 0, 0
  local n = 0
  for _, r in ipairs(rows) do
    local v = r[ezr.colindex(header, col)]
    if type(v) == "number" then
      sum = sum + v
      sumsq = sumsq + v * v
      n = n + 1
    end
  end
  local mu = sum / n
  local sigma = math.sqrt(math.max(sumsq / n - mu * mu, 0))
  numeric_stats[col] = {mu=mu, sigma=sigma}
end

-- Helper to get column index by name
function ezr.colindex(header, name)
  for i, n in ipairs(header) do
    if n == name then return i end
  end
  return nil
end

-- ------------------------------------------------------------------
-- Compute mode for symbolic columns
-- ------------------------------------------------------------------
local symbolic_modes = {}
for _, col in ipairs(symbolic_cols) do
  local counts = {}
  local best, bestcnt = nil, 0
  for _, r in ipairs(rows) do
    local v = r[ezr.colindex(header, col)]
    if v ~= "?" then
      counts[v] = (counts[v] or 0) + 1
      if counts[v] > bestcnt then
        bestcnt = counts[v]
        best = v
      end
    end
  end
  symbolic_modes[col] = best or ""
end

-- ------------------------------------------------------------------
-- Compute distance of each row to the centroid of characteristics
-- ------------------------------------------------------------------
local function compute_row_distance(r)
  local sumsq = 0
  -- numeric part
  for _, col in ipairs(numeric_cols) do
    local v = r[ezr.colindex(header, col)]
    local s = numeric_stats[col]
    if type(v) == "number" and s.sigma > 0 then
      local z = (v - s.mu) / s.sigma
      sumsq = sumsq + z * z
    end
  end
  -- symbolic part
  for _, col in ipairs(symbolic_cols) do
    local v = r[ezr.colindex(header, col)]
    local mode = symbolic_modes[col]
    local d = (v == mode or v == "?") and 0 or 1
    sumsq = sumsq + d * d
  end
  return math.sqrt(sumsq)
end

local distances = {}
for _, r in ipairs(rows) do
  local d = compute_row_distance(r)
  table.insert(distances, {row=r, dist=d})
end

-- Compute mean and sd of distances
local sumd, sumdsq, nd = 0, 0, 0
for _, d in ipairs(distances) do
  sumd = sumd + d.dist
  sumdsq = sumdsq + d.dist * d.dist
  nd = nd + 1
end
local meanDist = sumd / nd
local sdDist = math.sqrt(math.max(sumdsq / nd - meanDist * meanDist, 0))

-- ------------------------------------------------------------------
-- Regression helper (multiple linear regression via normal equations)
-- ------------------------------------------------------------------
local function build_regression(rows, target, predictors)
  local X = {}
  local y = {}
  for _, r in ipairs(rows) do
    local yval = r[ezr.colindex(header, target)]
    if type(yval) == "number" then
      local rowX = {1} -- intercept
      local ok = true
      for _, p in ipairs(predictors) do
        local pv = r[ezr.colindex(header, p)]
        if type(pv) ~= "number" then ok = false; break end
        table.insert(rowX, pv)
      end
      if ok then
        table.insert(X, rowX)
        table.insert(y, yval)
      end
    end
  end

  local p = #X[1]
  -- Compute XtX and Xty
  local XtX = {}
  local Xty = {}
  for i = 1, p do
    XtX[i] = {}
    for j = 1, p do XtX[i][j] = 0 end
    Xty[i] = 0
  end
  for k = 1, #X do
    local rowX = X[k]
    local yk = y[k]
    for i = 1, p do
      for j = 1, p do
        XtX[i][j] = XtX[i][j] + rowX[i] * rowX[j]
      end
      Xty[i] = Xty[i] + rowX[i] * yk
    end
  end

  -- Solve linear system XtX * beta = Xty
  -- Augmented matrix
  local A = {}
  for i = 1, p do
    A[i] = {}
    for j = 1, p do A[i][j] = XtX[i][j] end
    A[i][p+1] = Xty[i]
  end
  -- Gaussian elimination
  for i = 1, p do
    local diag = A[i][i]
    if diag == 0 then error("Singular matrix in regression") end
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

local function predict(beta, r, predictors)
  local val = beta[1]
  for i = 1, #predictors do
    val = val + beta[i+1] * r[ezr.colindex(header, predictors[i])]
  end
  return val
end

-- ------------------------------------------------------------------
-- Prepare data for regression (exclude rows with missing values)
-- ------------------------------------------------------------------
local all_numeric_predictors = {}
for _, col in ipairs(numeric_cols) do
  table.insert(all_numeric_predictors, col)
end

local effort_beta = build_regression(rows, "Effort-", all_numeric_predictors)
local defects_beta = build_regression(rows, "Defects-", all_numeric_predictors)

-- ------------------------------------------------------------------
-- Compute residuals and z-scores
-- ------------------------------------------------------------------
local effort_res = {}
local defects_res = {}
for _, r in ipairs(rows) do
  local actual_eff = r[ezr.colindex(header, "Effort-")]
  local actual_def = r[ezr.colindex(header, "Defects-")]
  if type(actual_eff) == "number" and type(actual_def) == "number" then
    local pred_eff = predict(effort_beta, r, all_numeric_predictors)
    local pred_def = predict(defects_beta, r, all_numeric_predictors)
    local reff = actual_eff - pred_eff
    local rdef = actual_def - pred_def
    table.insert(effort_res, reff)
    table.insert(defects_res, rdef)
    table.insert(rows, {row=r, reff=reff, rdef=rdef})
  end
end

local function mean_and_sd(arr)
  local n = #arr
  local sum = 0
  for _, v in ipairs(arr) do sum = sum + v end
  local mean = sum / n
  local ss = 0
  for _, v in ipairs(arr) do ss = ss + (v - mean) * (v - mean) end
  local sd = math.sqrt(math.max(ss / n, 0))
  return mean, sd
end

local eff_mu, eff_sd = mean_and_sd(effort_res)
local def_mu, def_sd = mean_and_sd(defects_res)

-- Add z-scores to rows table
for _, item in ipairs(rows) do
  local r = item.row
  if r then
    local reff = item.reff
    local rdef = item.rdef
    item.reff_z = reff / (eff_sd + 1e-12)
    item.rdef_z = rdef / (def_sd + 1e-12)
  end
end

-- ------------------------------------------------------------------
-- NASA-17: Flag projects with unusual characteristic combinations
-- ------------------------------------------------------------------
print("== NASA-17 ==")
-- Sort distances descending
table.sort(distances, function(a, b) return a.dist > b.dist end)
local topN = math.min(3, #distances)
for i = 1, topN do
  local r = distances[i].row
  local d = distances[i].dist
  local id = r[ezr.colindex(header, "idX")]
  print(string.format("ID=%s  Distance=%.3f  (μ=%.3f σ=%.3f)", id, d, meanDist, sdDist))
end

-- ------------------------------------------------------------------
-- NASA-18: Flag projects with effort/defect outliers
-- ------------------------------------------------------------------
print("\n== NASA-18 ==")
-- Compute a combined outlier metric (max absolute z)
for _, item in ipairs(rows) do
  local r = item.row
  if r then
    item.outlier_metric = math.max(math.abs(item.reff_z), math.abs(item.rdef_z))
  end
end
table.sort(rows, function(a, b) return a.outlier_metric > b.outlier_metric end)
local topN2 = math.min(3, #rows)
for i = 1, topN2 do
  local item = rows[i]
  local r = item.row
  local id = r[ezr.colindex(header, "idX")]
  local actual_eff = r[ezr.colindex(header, "Effort-")]
  local actual_def = r[ezr.colindex(header, "Defects-")]
  local pred_eff = predict(effort_beta, r, all_numeric_predictors)
  local pred_def = predict(defects_beta, r, all_numeric_predictors)
  print(string.format(
    "ID=%s  Effort=%s  Pred=%s  z=%.2f  Defects=%s  Pred=%s  z=%.2f",
    id,
    ezr.round(actual_eff), ezr.round(pred_eff), ezr.round(item.reff_z),
    ezr.round(actual_def), ezr.round(pred_def), ezr.round(item.rdef_z)))
end
