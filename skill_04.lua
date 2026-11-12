#!/usr/bin/env lua
-- Skill 04: NASA-07 & NASA-08
-- Usage: lua skill_04.lua <csv path>
-- Requires the ezr library (ezr.lua / ezr-lib.lua)

local ezr = require("ezr")

local path = arg[1]
if not path then
  error("Usage: lua skill_04.lua <csv path>")
end

-- ------------------------------------------------------------------
-- Read CSV – header + rows
-- ------------------------------------------------------------------
local function read_csv(path)
  local header, rows = nil, {}
  for r in ezr.csv(path) do
    if not header then
      header = r
      for i, name in ipairs(header) do header[i] = tostring(name) end
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
-- Simple statistics helpers
-- ------------------------------------------------------------------
local function mean(v)
  local s = 0
  for _, x in ipairs(v) do s = s + x end
  return s / #v end

local function sd(v, m)
  local ss = 0
  for _, x in ipairs(v) do ss = ss + (x - m) * (x - m) end
  return math.sqrt(ss / (#v - 1)) end

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
  local A = {}
  for i = 1, p do
    A[i] = {}
    for j = 1, p do A[i][j] = xtx[i][j] end
    A[i][p+1] = xty[i]
  end

  -- Forward elimination
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
-- Main
-- ------------------------------------------------------------------
local header, rows = read_csv(path)
local colidx = {}
for i, name in ipairs(header) do colidx[name] = i end

-- Identify numeric columns
local numeric = numeric_columns(header, rows)

-- Separate known (completed) and unknown (proposals) data
local known = {}
local proposals = {}
for _, row in ipairs(rows) do
  local eff = row[colidx["Effort-"]]
  local def = row[colidx["Defects-"]]
  if eff ~= "?" and def ~= "?" then
    table.insert(known, row)
  else
    table.insert(proposals, row)
  end
end

if #known == 0 then
  error("No completed-project records found for training")
end

-- Feature sets: all numeric except idX and target
local effort_features = {}
for name,_ in pairs(numeric) do
  if name ~= "idX" and name ~= "Effort-" then
    table.insert(effort_features, name)
  end
end
table.sort(effort_features)

local defect_features = {}
for name,_ in pairs(numeric) do
  if name ~= "idX" and name ~= "Defects-" then
    table.insert(defect_features, name)
  end
end
table.sort(defect_features)

-- Fit models
local effort_beta = linear_fit(known, "Effort-", effort_features, colidx)
local defect_beta = linear_fit(known, "Defects-", defect_features, colidx)

-- Gather actual historical values for thresholds
local eff_actual = {}
local def_actual = {}
for _, row in ipairs(known) do
  table.insert(eff_actual, row[colidx["Effort-"]])
  table.insert(def_actual, row[colidx["Defects-"]])
end

local eff_mean, eff_sd = mean(eff_actual), sd(eff_actual, mean(eff_actual))
local def_mean, def_sd = mean(def_actual), sd(def_actual, mean(def_actual))

-- ------------------------------------------------------------------
-- NASA-07: Evaluate proposals
-- ------------------------------------------------------------------
print("== NASA-07 ==")
if #proposals == 0 then
  print("No new proposals to evaluate.")
else
  for _, row in ipairs(proposals) do
    local id = row[colidx["idX"]]
    local pred_eff = predict(effort_beta, row, effort_features, colidx)
    local pred_def = predict(defect_beta, row, defect_features, colidx)
    local eff_z = eff_sd > 0 and (pred_eff - eff_mean) / eff_sd or 0
    local def_z = def_sd > 0 and (pred_def - def_mean) / def_sd or 0
    print(string.format("Proposal %s: Effort=%.1f (%+.1fσ), Defects=%.1f (%+.1fσ)",
      id, pred_eff, eff_z, pred_def, def_z))
    if eff_z > 2 then
      print(string.format("  *Alert*: Predicted effort unusually high for ID %s", id))
    end
    if def_z > 2 then
      print(string.format("  *Alert*: Predicted defects unusually high for ID %s", id))
    end
  end
end

-- ------------------------------------------------------------------
-- NASA-08: Update historical summaries
-- ------------------------------------------------------------------
print("\n== NASA-08 ==")
print(string.format("Updated effort summary: mean=%.1f, sd=%.1f (n=%d)",
  eff_mean, eff_sd, #eff_actual))
print(string.format("Updated defects summary: mean=%.1f, sd=%.1f (n=%d)",
  def_mean, def_sd, #def_actual))
