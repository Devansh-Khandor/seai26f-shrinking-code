#!/usr/bin/env lua
-- skill_05.lua
-- NASA-09: Find characteristics most strongly associated with high effort
-- NASA-10: Find characteristics distinguishing high defect projects
--
-- Run as:  lua skill_05.lua <csv path>

local ezr = require("ezr")
local csv = ezr.csv
local Tbl = ezr.Tbl

-- ----------------------------------------------------------------------
-- Helpers
-- ----------------------------------------------------------------------
-- Find column index by name
local function colIndex(tbl, name)
  for at, c in ipairs(tbl.cols.all) do
    if c.name == name then
      return at
    end
  end
  error("Column '" .. name .. "' not found")
end

-- Median + 1.5*IQR threshold
local function threshold(vals)
  local n = #vals
  if n == 0 then return 0 end
  table.sort(vals)
  local q1 = vals[(n + 3) // 4]
  local q3 = vals[(3*n + 1) // 4]
  local iqr = q3 - q1
  return vals[(n + 1) // 2] + 1.5 * iqr
end

-- Compute mean of numeric values (skipping "?" or non-numbers)
local function mean(vals)
  local sum, cnt = 0, 0
  for _, v in ipairs(vals) do
    if type(v) == "number" then
      sum = sum + v
      cnt = cnt + 1
    end
  end
  return cnt > 0 and sum / cnt or 0
end

-- Mode and its proportion for a list of values
local function mode_and_prop(vals)
  local freq = {}
  local total = 0
  for _, v in ipairs(vals) do
    if v ~= "?" then
      freq[v] = (freq[v] or 0) + 1
      total = total + 1
    end
  end
  local best, bestv, bestcnt = nil, nil, 0
  for k, c in pairs(freq) do
    if c > bestcnt then
      best, bestv, bestcnt = k, k, c
    end
  end
  return bestv, bestcnt / (total > 0 and total or 1)
end

-- ----------------------------------------------------------------------
-- Load data
-- ----------------------------------------------------------------------
local path = arg[1]
if not path then
  error("Usage: lua skill_05.lua <csv path>")
end

local tbl = Tbl(csv(path))

-- ----------------------------------------------------------------------
-- Column indices
-- ----------------------------------------------------------------------
local idxEffort = colIndex(tbl, "Effort-")
local idxDefect = colIndex(tbl, "Defects-")
local idxId     = colIndex(tbl, "idX")
local idxYear   = colIndex(tbl, "YearX")

-- ----------------------------------------------------------------------
-- Gather all numeric values for effort & defects (for thresholds)
-- ----------------------------------------------------------------------
local effort_vals = {}
local defect_vals = {}
for _, r in ipairs(tbl.rows) do
  local e = r[idxEffort]
  local d = r[idxDefect]
  if type(e) == "number" then table.insert(effort_vals, e) end
  if type(d) == "number" then table.insert(defect_vals, d) end
end

-- ----------------------------------------------------------------------
-- Compute thresholds for unusually high outcomes
-- ----------------------------------------------------------------------
local effort_thresh = threshold(effort_vals)
local defect_thresh = threshold(defect_vals)

-- ----------------------------------------------------------------------
-- Split rows into high / low groups
-- ----------------------------------------------------------------------
local high_eff_rows, low_eff_rows = {}, {}
local high_def_rows, low_def_rows = {}, {}

for _, r in ipairs(tbl.rows) do
  local e = r[idxEffort]
  local d = r[idxDefect]
  if type(e) == "number" then
    if e > effort_thresh then table.insert(high_eff_rows, r)
    else table.insert(low_eff_rows, r) end
  end
  if type(d) == "number" then
    if d > defect_thresh then table.insert(high_def_rows, r)
    else table.insert(low_def_rows, r) end
  end
end

-- ----------------------------------------------------------------------
-- Analyze characteristics
-- ----------------------------------------------------------------------
local function analyze(group_high, group_low, outcome_name, outcome_thresh)
  local numeric_stats = {}
  local symbolic_stats = {}

  -- Collect values per column across high/low groups
  for at, c in ipairs(tbl.cols.x) do
    local name = c.name
    if name ~= "Effort-" and name ~= "Defects-" and name ~= "idX" and name ~= "YearX" then
      local high_vals = {}
      local low_vals  = {}
      for _, r in ipairs(group_high) do
        table.insert(high_vals, r[c.at])
      end
      for _, r in ipairs(group_low) do
        table.insert(low_vals, r[c.at])
      end

      -- Numeric columns: mean difference
      local high_mean = mean(high_vals)
      local low_mean  = mean(low_vals)
      local diff = math.abs(high_mean - low_mean)

      if diff > 0 then
        table.insert(numeric_stats, {name=name, diff=diff})
      end

      -- Symbolic columns: mode proportion difference
      local high_mode, high_prop = mode_and_prop(high_vals)
      local low_mode,  low_prop  = mode_and_prop(low_vals)
      local prop_diff = math.abs(high_prop - low_prop)
      if prop_diff > 0 then
        table.insert(symbolic_stats, {name=name, diff=prop_diff})
      end
    end
  end

  -- Sort and keep top 3
  local function sort_desc(a,b) return a.diff > b.diff end
  table.sort(numeric_stats, sort_desc)
  table.sort(symbolic_stats, sort_desc)

  return {
    numeric = numeric_stats[1] and {numeric_stats[1], numeric_stats[2], numeric_stats[3]} or {},
    symbolic = symbolic_stats[1] and {symbolic_stats[1], symbolic_stats[2], symbolic_stats[3]} or {}
  }
end

-- ----------------------------------------------------------------------
-- NASA-09: High effort
-- ----------------------------------------------------------------------
print("== NASA-09 ==")
print(string.format(
  "High effort threshold (median + 1.5 IQR): %s",
  ezr.show(effort_thresh)))

local high_eff_stats = analyze(high_eff_rows, low_eff_rows,
                              "Effort", effort_thresh)

print("Top numeric characteristics (mean difference):")
for i, rec in ipairs(high_eff_stats.numeric) do
  print(string.format("%d. %s : %s", i, rec.name, ezr.show(rec.diff)))
end
print("Top symbolic characteristics (proportion difference):")
for i, rec in ipairs(high_eff_stats.symbolic) do
  print(string.format("%d. %s : %s", i, rec.name, ezr.show(rec.diff)))
end

print("\n")

-- ----------------------------------------------------------------------
-- NASA-10: High defects
-- ----------------------------------------------------------------------
print("== NASA-10 ==")
print(string.format(
  "High defect threshold (median + 1.5 IQR): %s",
  ezr.show(defect_thresh)))

local high_def_stats = analyze(high_def_rows, low_def_rows,
                               "Defects", defect_thresh)

print("Top numeric characteristics (mean difference):")
for i, rec in ipairs(high_def_stats.numeric) do
  print(string.format("%d. %s : %s", i, rec.name, ezr.show(rec.diff)))
end
print("Top symbolic characteristics (proportion difference):")
for i, rec in ipairs(high_def_stats.symbolic) do
  print(string.format("%d. %s : %s", i, rec.name, ezr.show(rec.diff)))
end
