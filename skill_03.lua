#!/usr/bin/env lua
-- Skill 03: Statistical comparison of fuel efficiency and overall performance
-- Usage: lua skill_03.lua <csv path>
--
--  CAR-05 : Compare fuel efficiency across cylinder counts and assess reliability.
--  CAR-06 : Compare overall performance between two model‑year ranges.

local ezr = require "ezr"
local csv = ezr.csv
local shuffle = ezr.shuffle
local keysort = ezr.keysort
local sorted = ezr.sorted
local round = ezr.round
local same = ezr.same
local cohen = ezr.cohen

-- ------------------------------------------------------------------
-- Utility: calculate mean, std, min, max for a numeric list
local function stats(vals)
  local n, sum, sumsq = 0, 0, 0
  local minv, maxv = math.huge, -math.huge
  for _, v in ipairs(vals) do
    n = n + 1
    sum = sum + v
    sumsq = sumsq + v * v
    if v < minv then minv = v end
    if v > maxv then maxv = v end
  end
  local mean = sum / n
  local var = n > 1 and (sumsq - sum * sum / n) / (n - 1) or 0
  local std = math.sqrt(var)
  return mean, std, minv, maxv
end

-- ------------------------------------------------------------------
-- Read CSV and build a flat list of rows
local path = arg[1]
if not path then
  io.stderr:write("Usage: lua skill_03.lua <csv path>\n")
  os.exit(1)
end

local rows_iter = csv(path)
local headers = rows_iter()
if not headers then
  io.stderr:write("Empty CSV\n")
  os.exit(1)
end

local rows = {}
while true do
  local r = rows_iter()
  if not r then break end
  table.insert(rows, r)
end

-- ------------------------------------------------------------------
--  CAR-05 : Fuel efficiency by cylinder count
print("== CAR-05 ==")

-- Group rows by cylinder count
local cyl_groups = {}
for _, r in ipairs(rows) do
  local k = tostring(r["Clndrs"])   -- cylinder count as key
  cyl_groups[k] = cyl_groups[k] or {}
  table.insert(cyl_groups[k], r["Mpg+"])
end

-- Prepare data for comparison
local cyl_keys = {}
for k in pairs(cyl_groups) do table.insert(cyl_keys, k) end
table.sort(cyl_keys, function(a,b) return tonumber(a) < tonumber(b) end)

-- Print per‑cylinder summary
print(string.format("%-6s %-6s %-6s %-6s %-6s", "Cyl","N","Mean","Std","Range"))
for _,k in ipairs(cyl_keys) do
  local vals = cyl_groups[k]
  local mean, std, minv, maxv = stats(vals)
  print(string.format("%-6s %-6d %-6.2f %-6.2f %-6.2f",
        k, #vals, mean, std, maxv-minv))
end

-- Pairwise comparison of cylinder groups
print("\nPairwise fuel‑efficiency differences (Cyl1 vs Cyl2):")
print(string.format("%-6s %-6s %-10s %-10s %-8s %-8s",
        "Cyl1","Cyl2","Mean Diff","Cohen d","Same?","Significant?"))

for i = 1, #cyl_keys do
  for j = i+1, #cyl_keys do
    local g1 = cyl_groups[cyl_keys[i]]
    local g2 = cyl_groups[cyl_keys[j]]
    local m1 = stats(g1).mean
    local m2 = stats(g2).mean
    local diff = m1 - m2
    local effect = cohen(g1, g2)
    local sorted_g1 = sorted(g1)
    local sorted_g2 = sorted(g2)
    local same_flag = same(sorted_g1, sorted_g2)
    local sig = not same_flag and math.abs(effect) > 0.5
    print(string.format("%-6s %-6s %10.2f %10.2f %-8s %-8s",
          cyl_keys[i], cyl_keys[j], diff, effect, same_flag and "yes" or "no",
          sig and "yes" or "no"))
  end
end

-- ------------------------------------------------------------------
--  CAR-06 : Overall performance across model‑year ranges
print("\n== CAR-06 ==")

-- Determine model years from the "Model" field (assumed to be the year)
local years = {}
for _, r in ipairs(rows) do
  local yr = tonumber(r["Model"])
  if yr then
    years[yr] = true
  end
end
local year_list = {}
for yr in pairs(years) do table.insert(year_list, yr) end
table.sort(year_list)

-- Split into two ranges: lower half and upper half
local mid = math.floor(#year_list / 2)
local range1 = {}
for i=1,mid do
  range1[year_list[i]] = true
end
local range2 = {}
for i=mid+1,#year_list do
  range2[year_list[i]] = true
end

-- Composite performance score: normalize Mpg+, Acc+, HpX, Volume
local function composite_score(r)
  -- Normalization helpers
  local function norm(v, minv, maxv, rev)
    if maxv == minv then return 0 end
    local n = (v - minv) / (maxv - minv)
    return rev and (1-n) or n
  end
  -- Gather min/max across all rows for each metric
  local mpg_vals, acc_vals, hp_vals, vol_vals = {}, {}, {}, {}
  for _, rr in ipairs(rows) do
    table.insert(mpg_vals, rr["Mpg+"])
    table.insert(acc_vals, rr["Acc+"])
    table.insert(hp_vals,  rr["HpX"])
    table.insert(vol_vals, rr["Volume"])
  end
  local _,_,_,mpg_max = stats(mpg_vals)
  local _,_,_,acc_max = stats(acc_vals)
  local _,_,_,hp_max  = stats(hp_vals)
  local _,_,_,vol_max = stats(vol_vals)
  local _,_,_,mpg_min = stats(mpg_vals)
  local _,_,_,acc_min = stats(acc_vals)
  local _,_,_,hp_min  = stats(hp_vals)
  local _,_,_,vol_min = stats(vol_vals)

  local mpg  = norm(r["Mpg+"], mpg_min, mpg_max, false)
  local acc  = norm(r["Acc+"], acc_min, acc_max, false)
  local hp   = norm(r["HpX"] , hp_min , hp_max , false)
  local vol  = norm(r["Volume"], vol_min, vol_max, false)
  return (mpg + acc + hp + vol) / 4
end

-- Build score list for each range
local scores1, scores2 = {}, {}
for _, r in ipairs(rows) do
  local yr = tonumber(r["Model"])
  local sc = composite_score(r)
  if yr then
    if range1[yr] then table.insert(scores1, sc) end
    if range2[yr] then table.insert(scores2, sc) end
  end
end

local mean1, std1, _, _ = stats(scores1)
local mean2, std2, _, _ = stats(scores2)

print(string.format("Range 1: years %d-%d, N=%d, mean score=%.3f",
        year_list[1], year_list[mid], #scores1, mean1))
print(string.format("Range 2: years %d-%d, N=%d, mean score=%.3f",
        year_list[mid+1], year_list[#year_list], #scores2, mean2))

local effect = cohen(scores1, scores2)
local sorted1, sorted2 = sorted(scores1), sorted(scores2)
local same_flag = same(sorted1, sorted2)
local sig = not same_flag and math.abs(effect) > 0.5

print(string.format("\nOverall performance comparison:")
      .."\n  Difference in mean score: %.3f"
      .."\n  Cohen's d: %.3f"
      .."\n  Same distribution? %s"
      .."\n  Significant difference? %s",
      mean1 - mean2, effect, same_flag and "yes" or "no",
      sig and "yes" or "no"))
