#!/usr/bin/env lua
-- Skill 03 – Statistical comparison of fuel‑efficiency and model‑year performance
-- Usage:  lua skill_03.lua <csv path>

local ezr = require "ezr"

-- ------------------------------------------------------------------
-- Helper utilities
-- ------------------------------------------------------------------
-- Simple arithmetic mean
local function mean(vs)
  local sum, n = 0, 0
  for _, v in ipairs(vs) do
    if v ~= "?" then sum, n = sum + v, n + 1 end
  end
  return n > 0 and sum / n or nil
end

-- Return a map from key to list of values
local function group_by(tbl, keycol, valcol)
  local groups = {}
  for _, r in ipairs(tbl.rows) do
    local k = r[keycol]
    local v = r[valcol]
    if k and v ~= "?" then
      groups[k] = groups[k] or {}
      table.insert(groups[k], v)
    end
  end
  return groups
end

-- ------------------------------------------------------------------
-- Main
-- ------------------------------------------------------------------
local csv_path = arg[1]
if not csv_path then
  io.stderr:write("Error: CSV file path required\n")
  os.exit(1)
end

local tbl = ezr.Tbl(ezr.csv(csv_path))

-- ------------------------------------------------------------------
-- CAR-05 : Fuel‑efficiency vs cylinder count
-- ------------------------------------------------------------------
print("== CAR-05 ==")
-- Group mpg by cylinder count
local mpg_by_cyl = group_by(tbl, tbl.cols.all.Clndrs.at, tbl.cols.all["Mpg+"].at)

-- Pre‑compute sorted lists for same test
local sorted_mpg_by_cyl = {}
for cyl, vals in pairs(mpg_by_cyl) do
  sorted_mpg_by_cyl[cyl] = ezr.sorted(vals)
end

-- Collect cylinder keys for pairwise comparison
local cyl_keys = {}
for k in pairs(mpg_by_cyl) do table.insert(cyl_keys, k) end
table.sort(cyl_keys)

for i = 1, #cyl_keys - 1 do
  for j = i + 1, #cyl_keys do
    local cyl1, cyl2 = cyl_keys[i], cyl_keys[j]
    local mpg1, mpg2 = mpg_by_cyl[cyl1], mpg_by_cyl[cyl2]
    local mean1, mean2 = mean(mpg1), mean(mpg2)
    local sd = ezr.cohen(sorted_mpg_by_cyl[cyl1], sorted_mpg_by_cyl[cyl2])
    local reliable = ezr.same(sorted_mpg_by_cyl[cyl1], sorted_mpg_by_cyl[cyl2]) == false
    print(string.format(
      "Cyl %d vs %d: mean mpg %.2f vs %.2f, effect size (Cohen's d)=%.2f, reliable? %s",
      cyl1, cyl2, mean1, mean2, sd,
      reliable and "YES" or "NO"))
  end
end

-- ------------------------------------------------------------------
-- CAR-06 : Performance across model‑year ranges
-- ------------------------------------------------------------------
print("\n== CAR-06 ==")

-- Determine median model year to split groups
local years = {}
for _, r in ipairs(tbl.rows) do
  local y = r[tbl.cols.all.Model.at]
  if y then table.insert(years, y) end
end
table.sort(years)
local median_year = years[math.floor(#years/2)]

-- Simple composite score: higher mpg better, lower weight better
local w_col = tbl.cols.all["Lbs-"]
local mpg_col = tbl.cols.all["Mpg+"]
local w_min, w_max = ezr.min(table.map(tbl.rows, function(r) return r[w_col.at] end)),
                     ezr.max(table.map(tbl.rows, function(r) return r[w_col.at] end))
local mpg_min, mpg_max = ezr.min(table.map(tbl.rows, function(r) return r[mpg_col.at] end)),
                         ezr.max(table.map(tbl.rows, function(r) return r[mpg_col.at] end))

local function score_row(r)
  local w = r[w_col.at]
  local m = r[mpg_col.at]
  local w_s = (w_max - w) / (w_max - w_min)
  local m_s = (m - mpg_min) / (mpg_max - mpg_min)
  return w_s + m_s
end

-- Separate rows into two groups
local low_group, high_group = {}, {}
for _, r in ipairs(tbl.rows) do
  local y = r[tbl.cols.all.Model.at]
  if y then
    if y <= median_year then table.insert(low_group, r)
    else table.insert(high_group, r) end
  end
end

local function scores(group)
  local s = {}
  for _, r in ipairs(group) do
    local v = score_row(r)
    if v then table.insert(s, v) end
  end
  return s
end

local low_scores, high_scores = scores(low_group), scores(high_group)
local sorted_low, sorted_high = ezr.sorted(low_scores), ezr.sorted(high_scores)

local mean_low, mean_high = mean(low_scores), mean(high_scores)
local effect = ezr.cohen(sorted_low, sorted_high)
local reliable = ezr.same(sorted_low, sorted_high) == false

print(string.format(
  "Model years <= %d vs > %d: mean score %.2f vs %.2f, effect size (Cohen's d)=%.2f, reliable? %s",
  median_year, median_year, mean_low, mean_high, effect,
  reliable and "YES" or "NO"))
