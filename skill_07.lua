#!/usr/bin/env lua
-- skill_07.lua
-- NASA-13: Identify characteristics that historically deliver more software
--           with lower effort and fewer defects.
-- NASA-14: Recommend which proposals should be evaluated first
--           given a limited evaluation budget.
--
-- Run as:  lua skill_07.lua <csv path>

local ezr = require("ezr")

local csv = ezr.csv
local Tbl = ezr.Tbl

-- ----------------------------------------------------------------------
-- Helpers
-- ----------------------------------------------------------------------
local function colIndex(tbl, name)
  for at, c in ipairs(tbl.cols.all) do
    if c.name == name then return at end
  end
  error("Column '" .. name .. "' not found")
end

-- Simple linear regression (x -> y)
local function linear_regression(x, y)
  local n = #x
  local sumx, sumy, sumxy, sumx2, sumy2 = 0, 0, 0, 0, 0
  for i = 1, n do
    local xi, yi = x[i], y[i]
    sumx  = sumx  + xi
    sumy  = sumy  + yi
    sumxy = sumxy + xi*yi
    sumx2 = sumx2 + xi*xi
    sumy2 = sumy2 + yi*yi
  end
  local meanx = sumx / n
  local meany = sumy / n
  local covxy = sumxy/n - meanx*meany
  local varx  = sumx2/n - meanx*meanx
  local vary  = sumy2/n - meany*meany

  local slope     = covxy / varx
  local intercept = meany - slope * meanx

  return {slope = slope, intercept = intercept}
end

-- Statistics: mean, median, min, max
local function stats(vals)
  local n = #vals
  if n == 0 then return {mean = 0, median = 0, min = 0, max = 0} end
  local sum = 0
  local minv, maxv = vals[1], vals[1]
  for _, v in ipairs(vals) do
    sum = sum + v
    if v < minv then minv = v end
    if v > maxv then maxv = v end
  end
  local mean = sum / n
  local sorted = {}
  for i = 1, n do sorted[i] = vals[i] end
  table.sort(sorted)
  local median = sorted[(n + 1) // 2]
  return {mean = mean, median = median, min = minv, max = maxv}
end

-- ----------------------------------------------------------------------
-- Load data
-- ----------------------------------------------------------------------
local path = arg[1]
if not path then
  error("Usage: lua skill_07.lua <csv path>")
end

local tbl = Tbl(csv(path))

-- Column indices
local idxKloc   = colIndex(tbl, "Kloc+")
local idxEffort = colIndex(tbl, "Effort-")
local idxDefect = colIndex(tbl, "Defects-")
local idxId     = colIndex(tbl, "idX")

-- ----------------------------------------------------------------------
-- Extract numeric vectors for regression
-- ----------------------------------------------------------------------
local kloc = {}
local effort = {}
local defects = {}
for _, r in ipairs(tbl.rows) do
  local k = r[idxKloc]
  local e = r[idxEffort]
  local d = r[idxDefect]
  if type(k) == "number" and type(e) == "number" and type(d) == "number" then
    table.insert(kloc, k)
    table.insert(effort, e)
    table.insert(defects, d)
  end
end

-- ----------------------------------------------------------------------
-- Build models (used for NASA-14)
-- ----------------------------------------------------------------------
local effort_mod = linear_regression(kloc, effort)
local defect_mod = linear_regression(kloc, defects)

-- ----------------------------------------------------------------------
-- Compute overall means for effort and defects
-- ----------------------------------------------------------------------
local overall_eff = stats(effort).mean
local overall_def = stats(defects).mean

-- ----------------------------------------------------------------------
-- NASA-13: Find promising characteristics
-- ----------------------------------------------------------------------
local function get_q1_q3(values)
  local n = #values
  local sorted = {}
  for i = 1, n do sorted[i] = values[i] end
  table.sort(sorted)
  local q1 = sorted[(n + 3) // 4]
  local q3 = sorted[(3*n + 1) // 4]
  return q1, q3
end

local function analyze_numeric(tbl, colIdx, effMean, defMean, idxEff, idxDef)
  local values = {}
  for _, r in ipairs(tbl.rows) do
    table.insert(values, r[colIdx])
  end
  local q1, q3 = get_q1_q3(values)

  local low_eff_sum, low_eff_cnt = 0, 0
  local low_def_sum, low_def_cnt = 0, 0
  local high_eff_sum, high_eff_cnt = 0, 0
  local high_def_sum, high_def_cnt = 0, 0

  for _, r in ipairs(tbl.rows) do
    local v = r[colIdx]
    if v <= q1 then
      low_eff_sum = low_eff_sum + r[idxEff]
      low_eff_cnt = low_eff_cnt + 1
      low_def_sum = low_def_sum + r[idxDef]
      low_def_cnt = low_def_cnt + 1
    elseif v > q3 then
      high_eff_sum = high_eff_sum + r[idxEff]
      high_eff_cnt = high_eff_cnt + 1
      high_def_sum = high_def_sum + r[idxDef]
      high_def_cnt = high_def_cnt + 1
    end
  end

  local low_eff_mean = low_eff_cnt > 0 and low_eff_sum / low_eff_cnt or effMean
  local low_def_mean = low_def_cnt > 0 and low_def_sum / low_def_cnt or defMean

  local diff_eff = effMean - low_eff_mean
  local diff_def = defMean - low_def_mean

  return {
    name = tbl.cols.all[colIdx].name,
    diff_eff = diff_eff,
    diff_def = diff_def
  }
end

local function analyze_symbolic(tbl, colIdx, effMean, defMean, idxEff, idxDef)
  local stats_map = {}
  for _, r in ipairs(tbl.rows) do
    local v = r[colIdx]
    local s = stats_map[v]
    if not s then
      s = {eff_sum = 0, eff_cnt = 0, def_sum = 0, def_cnt = 0}
      stats_map[v] = s
    end
    s.eff_sum = s.eff_sum + r[idxEff]
    s.eff_cnt = s.eff_cnt + 1
    s.def_sum = s.def_sum + r[idxDef]
    s.def_cnt = s.def_cnt + 1
  end

  local best = nil
  local best_score = -1

  for v, s in pairs(stats_map) do
    local mean_eff = s.eff_cnt > 0 and s.eff_sum / s.eff_cnt or effMean
    local mean_def = s.def_cnt > 0 and s.def_sum / s.def_cnt or defMean
    local diff_eff = effMean - mean_eff
    local diff_def = defMean - mean_def
    local score = diff_eff + diff_def
    if score > best_score then
      best_score = score
      best = {value = v, diff_eff = diff_eff, diff_def = diff_def}
    end
  end

  if best then
    best.name = tbl.cols.all[colIdx].name
    return best
  end
end

-- Gather numeric statistics
local numeric_stats = {}
for idx, col in ipairs(tbl.cols.all) do
  if idx ~= idxEffort and idx ~= idxDefect and idx ~= idxId then
    if type(tbl.rows[1][idx]) == "number" then
      table.insert(numeric_stats,
        analyze_numeric(tbl, idx, overall_eff, overall_def, idxEffort, idxDefect))
    end
  end
end
table.sort(numeric_stats,
  function(a,b) return (a.diff_eff + a.diff_def) > (b.diff_eff + b.diff_def) end)

-- Gather symbolic statistics
local symbolic_stats = {}
for idx, col in ipairs(tbl.cols.all) do
  if idx ~= idxEffort and idx ~= idxDefect and idx ~= idxId then
    if type(tbl.rows[1][idx]) ~= "number" then
      local rec = analyze_symbolic(tbl, idx, overall_eff, overall_def, idxEffort, idxDefect)
      if rec then table.insert(symbolic_stats, rec) end
    end
  end
end
table.sort(symbolic_stats,
  function(a,b) return (a.diff_eff + a.diff_def) > (b.diff_eff + b.diff_def) end)

-- ----------------------------------------------------------------------
-- Output NASA-13
-- ----------------------------------------------------------------------
print("== NASA-13 ==")
print("Top 3 numeric characteristics associated with lower effort and fewer defects:")
for i, rec in ipairs(numeric_stats) do
  if i > 3 then break end
  print(string.format("%d. %-10s: effort diff %.2f, defects diff %.2f",
        i, rec.name, rec.diff_eff, rec.diff_def))
end

print("\nTop 3 symbolic characteristics associated with lower effort and fewer defects:")
for i, rec in ipairs(symbolic_stats) do
  if i > 3 then break end
  print(string.format("%d. %-10s=%-5s: effort diff %.2f, defects diff %.2f",
        i, rec.name, tostring(rec.value), rec.diff_eff, rec.diff_def))
end

-- ----------------------------------------------------------------------
-- NASA-14: Recommend proposals to evaluate first
-- ----------------------------------------------------------------------
local all_rows = tbl.rows
local scored = {}
for _, r in ipairs(all_rows) do
  local k = r[idxKloc]
  local pred_eff = effort_mod.intercept + effort_mod.slope * k
  local pred_def = defect_mod.intercept + defect_mod.slope * k
  local score = pred_eff + pred_def
  table.insert(scored, {id = r[idxId], pred_eff = pred_eff,
        pred_def = pred_def, score = score})
end
table.sort(scored, function(a,b) return a.score < b.score end)

local budget = the.budget or 10
local to_eval = {}
for i = 1, math.min(budget, #scored) do
  table.insert(to_eval, scored[i])
end

print("\n== NASA-14 ==")
print(string.format("Recommended proposals to evaluate first (budget=%d):",
      #to_eval))
for i, r in ipairs(to_eval) do
  print(string.format("%d. id=%s: predicted effort %.1f, predicted defects %.1f, score %.2f",
        i, r.id, r.pred_eff, r.pred_def, r.score))
end
