#!/usr/bin/env lua
-- skill_05.lua
-- Implements APACHE-09 and APACHE-10 analyses.

local ezr = require"ezr"

-- Helpers ---------------------------------------------------------
local function display(v)
  if type(v) == "number" then return ezr.round(v) else return tostring(v) end
end

local function quartiles(list)
  local s = ezr.sorted(list)
  local n = #s
  local q1 = s[(n + 3) // 4]
  local q3 = s[(3*n + 1) // 4]
  return q1, q3
end

-- Main ------------------------------------------------------------
local csvfile = arg[1]
assert(csvfile, "Usage: lua skill_05.lua <csv path>")

local tbl = ezr.Tbl(ezr.csv(csvfile))
local perf_col = tbl.cols.y[1]

-- Gather performance values
local perf_vals = ezr.map(tbl.rows, function(r) return r[perf_col.at] end)

-- Determine threshold for poorly performing configurations
local q1, q3 = quartiles(perf_vals)
local iqr = q3 - q1
local high_threshold = q3 + 1.5 * iqr

-- Identify poor rows
local poor_rows = {}
for _, r in ipairs(tbl.rows) do
  if r[perf_col.at] > high_threshold then
    table.insert(poor_rows, r)
  end
end

-- ---------- APACHE-09 ----------
print("== APACHE-09 ==")
print(string.format("Poor configuration threshold (high performance): %s", display(high_threshold)))
print(string.format("Number of poor configurations found: %d", #poor_rows))

-- Compute association differences for each setting
local diffs = {}
for _, c in ipairs(tbl.cols.all) do
  if c ~= perf_col then
    local sum1, n1, sum0, n0 = 0,0,0,0
    for _, r in ipairs(tbl.rows) do
      local v = r[c.at]
      if v == "?" then
        -- ignore missing
      elseif v == 1 or v == "True" or v == "true" then
        sum1 = sum1 + r[perf_col.at]; n1 = n1 + 1
      else
        sum0 = sum0 + r[perf_col.at]; n0 = n0 + 1
      end
    end
    local mean1 = n1>0 and sum1/n1 or nil
    local mean0 = n0>0 and sum0/n0 or nil
    if mean1 and mean0 then
      table.insert(diffs, {name=c.name, diff=mean1-mean0})
    end
  end
end

table.sort(diffs, function(a,b) return math.abs(a.diff or 0) > math.abs(b.diff or 0) end)

print("Top settings most strongly associated with poor performance:")
for i=1, math.min(5,#diffs) do
  local d = diffs[i]
  print(string.format("%d. %s: diff=%s", i, d.name, display(d.diff)))
end

-- ---------- APACHE-10 ----------
print("\n== APACHE-10 ==")

-- Helper to build a configuration key string
local function config_key(r)
  local parts = {}
  for _,c in ipairs(tbl.cols.all) do
    if c ~= perf_col then
      table.insert(parts, tostring(r[c.at]))
    end
  end
  return table.concat(parts, ",")
end

-- Compute top combos: by frequency (poor rows) or by average performance (all rows)
local function top_combos(rows, topN, by_avg)
  local freq = {}
  local sum = {}
  for _, r in ipairs(rows) do
    local k = config_key(r)
    freq[k] = (freq[k] or 0) + 1
    sum[k] = (sum[k] or 0) + r[perf_col.at]
  end
  local combos = {}
  for k,cnt in pairs(freq) do
    local avg = sum[k]/cnt
    table.insert(combos, {key=k, count=cnt, avg=avg})
  end
  if by_avg then
    table.sort(combos, function(a,b) return a.avg < b.avg end)
  else
    table.sort(combos, function(a,b) return a.count > b.count end)
  end
  local top = {}
  for i=1, math.min(topN, #combos) do
    table.insert(top, combos[i])
  end
  return top
end

local top_poor = top_combos(poor_rows, 3, false)
local top_good = top_combos(tbl.rows, 3, true)

print("Top 3 poor configuration patterns (most frequent among poor configs):")
for i,comb in ipairs(top_poor) do
  print(string.format("%d. pattern=%s, count=%d, avg_perf=%s",
        i, comb.key, comb.count, display(comb.avg)))
end

print("\nTop 3 best configuration patterns (lowest avg performance):")
for i,comb in ipairs(top_good) do
  print(string.format("%d. pattern=%s, count=%d, avg_perf=%s",
        i, comb.key, comb.count, display(comb.avg)))
end

print("\nDifference between average performance of top poor vs top good patterns:")
for i=1, math.min(#top_poor,#top_good) do
  local diff = top_good[i].avg - top_poor[i].avg
  print(string.format("%d. poor avg=%s, good avg=%s, diff=%s",
        i, display(top_poor[i].avg), display(top_good[i].avg), display(diff)))
end
