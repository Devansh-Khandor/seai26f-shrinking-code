#!/usr/bin/env lua
--[[
  skill_03.lua
  NASA-93 data analysis – programmer capability and complexity comparisons.
  Usage:  lua skill_03.lua <csv-path>
  Prints:
    NASA-05: statistical comparison of Effort- across programmer capability ratings
    NASA-06: statistical comparison of Defects- across complexity ratings
--]]

local ezr = require "ezr"

-- ---------- helpers ----------------------------------------------------
local function abort(msg)
  io.stderr:write(msg .. "\n")
  os.exit(1)
end

local function round(v)
  return ezr.round(v)
end

-- compute simple statistics (mean, std, n) for a list of numeric values
local function stats(vals)
  local n = #vals
  if n == 0 then
    return {n = 0, mean = 0, std = 0}
  end
  local sum, sumsq = 0, 0
  for _, v in ipairs(vals) do
    sum = sum + v
    sumsq = sumsq + v * v
  end
  local mean = sum / n
  local var = n > 1 and (sumsq - sum * mean) / (n - 1) or 0
  local std = math.sqrt(var)
  return {n = n, mean = mean, std = std}
end

-- normal CDF using Abramowitz & Stegun approximation
local function normal_cdf(z)
  local t = 1 / (1 + 0.3275911 * math.abs(z))
  local a1, a2, a3, a4, a5 = 0.254829592, -0.284496736, 1.421413741,
                             -1.453152027, 1.061405429
  local erf = 1 - (((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t)
                * math.exp(-z * z)
  if z < 0 then erf = -erf end
  return 0.5 * (1 + erf)
end

-- two‑sided p‑value for normal difference
local function normal_pvalue(z)
  local p = 2 * (1 - normal_cdf(math.abs(z)))
  return p
end

-- ---------- main --------------------------------------------------------
local csv_path = arg[1]
if not csv_path then abort("Usage: lua skill_03.lua <csv-path>") end

-- read rows
local rows = {}
for r in ezr.csv(csv_path) do
  table.insert(rows, r)
end
if #rows == 0 then abort("No data found in " .. csv_path) end

-- build table & map
local tbl = ezr.Tbl(rows)
local col_map = {}
for i, c in ipairs(tbl.cols.all) do
  col_map[tbl.cols.names[i]] = c
end

-- indices of interest
local acap_col = col_map["acap"]
local cplx_col = col_map["cplx"]
local effort_col = col_map["Effort-"]
local defect_col = col_map["Defects-"]

-- group data by a capability / complexity rating
local function group_by(col, target_col)
  local groups = {}
  for _, r in ipairs(rows) do
    local g = r[col.at]
    local v = r[target_col.at]
    if g ~= "?" and v ~= "?" then
      groups[g] = groups[g] or {}
      table.insert(groups[g], v)
    end
  end
  return groups
end

local effort_by_acap = group_by(acap_col, effort_col)
local defect_by_cplx = group_by(cplx_col, defect_col)

-- ---------- NASA-05 ----------------------------------------------------
print("== NASA-05 ==")
print("Programmer capability comparison of Effort- (lower is better).")
print()

-- stats per capability
local acap_stats = {}
for g, vals in pairs(effort_by_acap) do
  acap_stats[g] = stats(vals)
end

-- print per group
for g, s in pairs(acap_stats) do
  print(string.format("Group '%s': n=%d, mean=%s, std=%s",
    g, s.n, round(s.mean), round(s.std)))
end
print()

-- pairwise comparisons
local groups = {}
for g in pairs(acap_stats) do table.insert(groups, g) end
table.sort(groups)

for i = 1, #groups-1 do
  for j = i+1, #groups do
    local g1, g2 = groups[i], groups[j]
    local s1, s2 = acap_stats[g1], acap_stats[g2]
    if s1.n >= 2 and s2.n >= 2 then
      local diff = s1.mean - s2.mean
      local se = math.sqrt(s1.std*s1.std / s1.n + s2.std*s2.std / s2.n)
      local z = diff / se
      local p = normal_pvalue(z)
      local low = round(diff - 1.96 * se)
      local high = round(diff + 1.96 * se)
      local meaningful = (low > 0 or high < 0) and p < 0.05
      print(string.format("Comparison: '%s' vs '%s'", g1, g2))
      print(string.format("  Mean Effort- difference: %s",
        round(diff)))
      print(string.format("  95%% CI: [%s, %s]", low, high))
      print(string.format("  p-value: %s", round(p)))
      print(string.format("  Meaningful difference? %s",
        meaningful and "Yes" or "No"))
      print()
    end
  end
end

-- ---------- NASA-06 ----------------------------------------------------
print("== NASA-06 ==")
print("Complexity rating comparison of Defects- (lower is better).")
print()

-- stats per complexity
local cplx_stats = {}
for g, vals in pairs(defect_by_cplx) do
  cplx_stats[g] = stats(vals)
end

-- print per group
for g, s in pairs(cplx_stats) do
  print(string.format("Group '%s': n=%d, mean=%s, std=%s",
    g, s.n, round(s.mean), round(s.std)))
end
print()

-- pairwise comparisons
local groups2 = {}
for g in pairs(cplx_stats) do table.insert(groups2, g) end
table.sort(groups2)

for i = 1, #groups2-1 do
  for j = i+1, #groups2 do
    local g1, g2 = groups2[i], groups2[j]
    local s1, s2 = cplx_stats[g1], cplx_stats[g2]
    if s1.n >= 2 and s2.n >= 2 then
      local diff = s1.mean - s2.mean
      local se = math.sqrt(s1.std*s1.std / s1.n + s2.std*s2.std / s2.n)
      local z = diff / se
      local p = normal_pvalue(z)
      local low = round(diff - 1.96 * se)
      local high = round(diff + 1.96 * se)
      local meaningful = (low > 0 or high < 0) and p < 0.05
      print(string.format("Comparison: '%s' vs '%s'", g1, g2))
      print(string.format("  Mean Defects- difference: %s",
        round(diff)))
      print(string.format("  95%% CI: [%s, %s]", low, high))
      print(string.format("  p-value: %s", round(p)))
      print(string.format("  Meaningful difference? %s",
        meaningful and "Yes" or "No"))
      print()
    end
  end
end
