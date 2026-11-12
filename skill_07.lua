#!/usr/bin/env lua
-- Skill 07: NASA-13 & NASA-14
-- Usage: lua skill_07.lua <csv path>
-- Requires the ezr library (ezr.lua / ezr-lib.lua)

local ezr = require("ezr")

local path = arg[1]
if not path then
  error("Usage: lua skill_07.lua <csv path>")
end

-- ------------------------------------------------------------------
-- Helpers for reading CSV
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

local header, rows = read_csv(path)
if #rows == 0 then error("No data rows") end

local idx = {}
for i, n in ipairs(header) do idx[n] = i end

-- ------------------------------------------------------------------
-- Numeric columns (first row decides type)
-- ------------------------------------------------------------------
local function numeric_columns(header, rows)
  local nums = {}
  local first = rows[1]
  for i, name in ipairs(header) do
    if type(first[i]) == "number" then
      nums[name] = i
    end
  end
  return nums
end

local numeric = numeric_columns(header, rows)

-- ------------------------------------------------------------------
-- Separate known and proposal rows
-- ------------------------------------------------------------------
local known = {}
local proposals = {}
for _, r in ipairs(rows) do
  local eff = r[idx["Effort-"]]
  local def = r[idx["Defects-"]]
  if eff ~= "?" and def ~= "?" then
    table.insert(known, r)
  else
    table.insert(proposals, r)
  end
end

if #known == 0 then error("No completed projects to learn from") end

-- ------------------------------------------------------------------
-- Simple linear regression
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
-- Build feature lists
-- ------------------------------------------------------------------
local effort_features = {}
local defect_features = {}
for name,_ in pairs(numeric) do
  if name ~= "idX" and name ~= "Effort-" then
    table.insert(effort_features, name)
  end
  if name ~= "idX" and name ~= "Defects-" then
    table.insert(defect_features, name)
  end
end
table.sort(effort_features)
table.sort(defect_features)

-- ------------------------------------------------------------------
-- Train models
-- ------------------------------------------------------------------
local effort_beta = linear_fit(known, "Effort-", effort_features, idx)
local defect_beta = linear_fit(known, "Defects-", defect_features, idx)

-- ------------------------------------------------------------------
-- Historical summary for z‑scoring
-- ------------------------------------------------------------------
local eff_hist = {}
local def_hist = {}
for _, r in ipairs(known) do
  table.insert(eff_hist, r[idx["Effort-"]])
  table.insert(def_hist, r[idx["Defects-"]])
end
local eff_mean, eff_sd = mean(eff_hist), sd(eff_hist, mean(eff_hist))
local def_mean, def_sd = mean(def_hist), sd(def_hist, mean(def_hist))

-- ------------------------------------------------------------------
-- NASA-13: Historical best projects and group characteristics
-- ------------------------------------------------------------------
-- 1. Compute score for each known project
local function efficiency_score(loc, eff, def)
  local eff_part = loc / eff
  local def_part = def > 0 and 1/(def+0.001) or 1
  return eff_part * def_part
end

local project_scores = {}
for _, r in ipairs(known) do
  local loc = r[idx["Kloc+"]]
  local eff = r[idx["Effort-"]]
  local def = r[idx["Defects-"]]
  local score = efficiency_score(loc, eff, def)
  table.insert(project_scores, {
    id = r[idx["idX"]],
    loc = loc,
    effort = eff,
    defects = def,
    score = score,
    features = {
      acap = r[idx["acap"]],
      cplx = r[idx["cplx"]],
      team = r[idx["team"]]
    }
  })
end

-- Top 3 projects
table.sort(project_scores, function(a,b) return a.score > b.score end)
local topProjects = {}
for i=1, math.min(3, #project_scores) do
  topProjects[i] = project_scores[i]
end

-- 2. Group by a few key characteristics and find best group
local group_map = {}
for _, p in ipairs(project_scores) do
  local key = (p.features.acap or "") .. "|" .. (p.features.cplx or "") .. "|" .. (p.features.team or "")
  local g = group_map[key]
  if not g then
    g = {acap=p.features.acap, cplx=p.features.cplx, team=p.features.team,
         total=0, loc=0, effort=0, defects=0}
    group_map[key] = g
  end
  g.total = g.total + 1
  g.loc = g.loc + p.loc
  g.effort = g.effort + p.effort
  g.defects = g.defects + p.defects
end

-- Compute averages and choose group with highest efficiency
local best_group = nil
local best_group_score = -1
for _, g in pairs(group_map) do
  local avg_loc = g.loc / g.total
  local avg_eff = g.effort / g.total
  local avg_def = g.defects / g.total
  local score = efficiency_score(avg_loc, avg_eff, avg_def)
  if score > best_group_score then
    best_group_score = score
    best_group = g
  end
end

-- ------------------------------------------------------------------
-- NASA-14: Recommend proposals to evaluate first
-- ------------------------------------------------------------------
local proposal_scores = {}
for _, r in ipairs(proposals) do
  local loc = r[idx["Kloc+"]]
  if loc and loc ~= "?" then
    local pred_eff = predict(effort_beta, r, effort_features, idx)
    local pred_def = predict(defect_beta, r, defect_features, idx)
    local score = efficiency_score(loc, pred_eff, pred_def)
    table.insert(proposal_scores, {
      id = r[idx["idX"]],
      pred_eff = pred_eff,
      pred_def = pred_def,
      score = score,
      features = {
        acap = r[idx["acap"]],
        cplx = r[idx["cplx"]],
        team = r[idx["team"]]
      }
    })
  end
end

-- Top 3 proposals by predicted score
table.sort(proposal_scores, function(a,b) return a.score > b.score end)
local topProposals = {}
for i=1, math.min(3, #proposal_scores) do
  topProposals[i] = proposal_scores[i]
end

-- ------------------------------------------------------------------
-- Output
-- ------------------------------------------------------------------
print("== NASA-13 ==")
print("Top 3 historically efficient projects:")
for i,p in ipairs(topProjects) do
  print(string.format("  %d. ID=%s, LOC=%.1f, Effort=%.1f, Defects=%d, Score=%.3f",
        i, p.id, p.loc, p.effort, p.defects, p.score))
end
print("\nBest group by acap, cplx, team:")
if best_group then
  print(string.format("  acap=%s, cplx=%s, team=%s -> Avg LOC=%.1f, Avg Effort=%.1f, Avg Defects=%.1f",
        best_group.acap, best_group.cplx, best_group.team,
        best_group.loc / best_group.total,
        best_group.effort / best_group.total,
        best_group.defects / best_group.total))
else
  print("  No group data available")
end

print("\n== NASA-14 ==")
print("Top 3 proposals to evaluate first:")
for i,p in ipairs(topProposals) do
  print(string.format("  %d. ID=%s, PredEffort=%.1f, PredDefects=%.1f, PredScore=%.3f",
        i, p.id, p.pred_eff, p.pred_def, p.score))
end
