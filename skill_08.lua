#!/usr/bin/env lua
-- Skill 08: NASA-15 & NASA-16
-- Usage: lua skill_08.lua <csv path>
-- Requires the ezr library (ezr.lua / ezr-lib.lua)

local ezr = require("ezr")

local path = arg[1]
if not path then
  error("Usage: lua skill_08.lua <csv path>")
end

-- ------------------------------------------------------------------
-- Helper: read CSV – header + rows
-- ------------------------------------------------------------------
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

-- ------------------------------------------------------------------
-- Linear regression helpers (copied from earlier skills)
-- ------------------------------------------------------------------
local function mean(v)
  local s = 0
  for _, x in ipairs(v) do s = s + x end
  return s / #v end

local function sd(v, m)
  local ss = 0
  for _, x in ipairs(v) do ss = ss + (x - m) * (x - m) end
  return math.sqrt(ss / (#v - 1)) end

local function linear_fit(rows, yname, features, colidx)
  local n = #rows
  local p = 1 + #features          -- intercept + predictors
  local xtx = {}
  local xty = {}
  for i = 1, p do
    xtx[i] = {}
    for j = 1, p do xtx[i][j] = 0 end
    xty[i] = 0
  end
  for _, row in ipairs(rows) do
    local x = {1}
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

local function predict(beta, row, features, colidx)
  local val = beta[1]
  for i = 1, #features do
    val = val + beta[i+1] * row[colidx[features[i]]]
  end
  return val
end

-- ------------------------------------------------------------------
-- Load data
-- ------------------------------------------------------------------
local header, rows = read_csv(path)

-- Build column index
local idx = {}
for i, n in ipairs(header) do idx[n] = i end

-- Separate known and proposal rows
local known, proposals = {}, {}
for _, r in ipairs(rows) do
  local eff = r[idx["Effort-"]]
  local def = r[idx["Defects-"]]
  if eff ~= "?" and def ~= "?" then
    table.insert(known, r)
  else
    table.insert(proposals, r)
  end
end

if #known == 0 then
  error("No completed projects to train on")
end

-- ------------------------------------------------------------------
-- Determine numeric features for regression
-- ------------------------------------------------------------------
local numeric = {}
for i, n in ipairs(header) do
  local val = rows[1][i]
  if type(val) == "number" then
    if n ~= "idX" and n ~= "Effort-" and n ~= "Defects-" then
      table.insert(numeric, n)
    end
  end
end

-- Build models
local effort_beta = linear_fit(known, "Effort-", numeric, idx)
local defect_beta = linear_fit(known, "Defects-", numeric, idx)

-- ------------------------------------------------------------------
-- Compute group statistics for categorical (symbolic) columns
-- ------------------------------------------------------------------
local categorical = {}
for i, n in ipairs(header) do
  if n ~= "idX" and n ~= "Effort-" and n ~= "Defects-" then
    local val = rows[1][i]
    if type(val) ~= "number" then
      table.insert(categorical, n)
    end
  end
end

local stats = {} -- stats[col][value] = {eff=sum, def=sum, n=count}
for _, col in ipairs(categorical) do
  stats[col] = {}
  for _, r in ipairs(known) do
    local v = r[idx[col]]
    if v and v ~= "?" then
      local g = stats[col][v]
      if not g then
        g = {eff=0, def=0, n=0}
        stats[col][v] = g
      end
      g.eff = g.eff + r[idx["Effort-"]]
      g.def = g.def + r[idx["Defects-"]]
      g.n   = g.n + 1
    end
  end
  -- convert sums to means
  for _, g in pairs(stats[col]) do
    g.eff = g.eff / g.n
    g.def = g.def / g.n
  end
end

-- ------------------------------------------------------------------
-- Predict for proposals
-- ------------------------------------------------------------------
for _, r in ipairs(proposals) do
  r.__predEffort = predict(effort_beta, r, numeric, idx)
  r.__predDefects = predict(defect_beta, r, numeric, idx)
end

-- ------------------------------------------------------------------
-- Helper to find best categorical change for effort
-- ------------------------------------------------------------------
local function best_effort_change(r)
  local best = nil
  for _, col in ipairs(categorical) do
    local cur = r[idx[col]]
    if cur and cur ~= "?" then
      local curStat = stats[col][cur]
      if not curStat then goto continue end
      local bestVal, bestEff = nil, nil
      for val, g in pairs(stats[col]) do
        if not bestEff or g.eff < bestEff then
          bestEff = g.eff
          bestVal = val
        end
      end
      if bestEff < curStat.eff then
        local delta = curStat.eff - bestEff
        if not best or delta > best.delta then
          best = {
            col=col, from=cur, to=bestVal,
            curEff=curStat.eff, newEff=bestEff,
            delta=delta
          }
        end
      end
    end
    ::continue::
  end
  return best
end

-- ------------------------------------------------------------------
-- Helper to find best categorical change for defects
-- ------------------------------------------------------------------
local function best_defect_change(r)
  local best = nil
  for _, col in ipairs(categorical) do
    local cur = r[idx[col]]
    if cur and cur ~= "?" then
      local curStat = stats[col][cur]
      if not curStat then goto continue end
      local bestVal, bestDef = nil, nil
      for val, g in pairs(stats[col]) do
        if not bestDef or g.def < bestDef then
          bestDef = g.def
          bestVal = val
        end
      end
      if bestDef < curStat.def then
        local delta = curStat.def - bestDef
        if not best or delta > best.delta then
          best = {
            col=col, from=cur, to=bestVal,
            curDef=curStat.def, newDef=bestDef,
            delta=delta
          }
        end
      end
    end
    ::continue::
  end
  return best
end

-- ------------------------------------------------------------------
-- Find top proposals for NASA-15 and NASA-16
-- ------------------------------------------------------------------
local function top_by(field, count)
  table.sort(proposals, function(a,b) return a[field] > b[field] end)
  local top = {}
  for i=1, math.min(count, #proposals) do
    top[i] = proposals[i]
  end
  return top
end

local topEffort = top_by("__predEffort", 3)
local topDefects = top_by("__predDefects", 3)

-- ------------------------------------------------------------------
-- Output NASA-15
-- ------------------------------------------------------------------
print("== NASA-15 ==")
for _, r in ipairs(topEffort) do
  local id = r[idx["idX"]]
  local estEff = r.__predEffort
  local change = best_effort_change(r)
  print(string.format("Project %s: Predicted effort = %.1f LOC", id, estEff))
  if change then
    print(string.format("  Suggestion: change %s from '%s' to '%s' →",
          change.col, change.from, change.to))
    print(string.format("  New effort estimate: %.1f LOC (reduction %.1f LOC)",
          change.newEff, change.delta))
  else
    print("  No single categorical adjustment can reduce effort based on historical data.")
  end
end

-- ------------------------------------------------------------------
-- Output NASA-16
-- ------------------------------------------------------------------
print("\n== NASA-16 ==")
for _, r in ipairs(topDefects) do
  local id = r[idx["idX"]]
  local estDef = r.__predDefects
  local change = best_defect_change(r)
  print(string.format("Project %s: Predicted defects = %.1f", id, estDef))
  if change then
    print(string.format("  Suggestion: change %s from '%s' to '%s' →",
          change.col, change.from, change.to))
    print(string.format("  New defect estimate: %.1f (reduction %.1f)",
          change.newDef, change.delta))
  else
    print("  No single categorical adjustment can reduce defects based on historical data.")
  end
end
