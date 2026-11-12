#!/usr/bin/env lua
-- Skill 05 – Apache performance diagnostics (blame & pattern)
-- Requires the ezr library (ezr.lua & ezr-lib.lua in the same dir)

local ezr = require"ezr"

-- ------------------------------------------------------------------
-- Get the CSV file path from the command line
-- ------------------------------------------------------------------
local csv_path = arg[1]
assert(csv_path, "Usage: lua skill_05.lua <csv path>")

-- ------------------------------------------------------------------
-- Load the data into an ezr table
-- ------------------------------------------------------------------
local tbl = ezr.Tbl(ezr.csv(csv_path))

-- ------------------------------------------------------------------
-- Helper: return a set of distinct values for a column
-- ------------------------------------------------------------------
local function distinct_values(col, rows)
  local s = {}
  for _, r in ipairs(rows or tbl.rows) do
    local v = r[col.at]
    if v ~= "?" then s[v] = true end
  end
  local res = {}
  for k in pairs(s) do table.insert(res, k) end
  return res
end

-- ------------------------------------------------------------------
-- Helper: check if a column is binary (<= 2 distinct values)
-- ------------------------------------------------------------------
local function is_binary(col)
  return #distinct_values(col) <= 2
end

-- ------------------------------------------------------------------
-- Find the performance column (goal)
-- ------------------------------------------------------------------
local perf_col = tbl.cols.klass
local perf_at  = perf_col.at

-- ------------------------------------------------------------------
-- Compute overall stats
-- ------------------------------------------------------------------
local perf_vals = {}
for _, r in ipairs(tbl.rows) do
  perf_vals[#perf_vals + 1] = r[perf_at]
end
local overall_mean = perf_col:mid()
local overall_sd   = perf_col:div()
local poor_thresh  = overall_mean + 2 * overall_sd   -- "poor" definition

-- ------------------------------------------------------------------
-- APACHE-09 – Settings most strongly associated with poor performance
-- ------------------------------------------------------------------
print("== APACHE-09 ==")

-- Gather binary columns
local binary_cols = {}
for _, c in ipairs(tbl.cols.all) do
  if c ~= perf_col and is_binary(c) then
    table.insert(binary_cols, c)
  end
end

-- Analyse each binary column
local analyses = {}
for _, c in ipairs(binary_cols) do
  local groups = {}
  local vals = {}
  for _, r in ipairs(tbl.rows) do
    local v = r[c.at]
    if v ~= "?" then
      if not groups[v] then
        groups[v] = {sum=0, cnt=0, perf_list={}}
      end
      groups[v].sum = groups[v].sum + r[perf_at]
      groups[v].cnt = groups[v].cnt + 1
      table.insert(groups[v].perf_list, r[perf_at])
    end
  end
  -- Only analyse columns that actually split rows
  if next(groups) and next(groups) and #next(groups) > 1 then
    local vals = {}
    for k in pairs(groups) do table.insert(vals, k) end
    -- compute means
    local m1 = groups[vals[1]].sum / groups[vals[1]].cnt
    local m2 = groups[vals[2]].sum / groups[vals[2]].cnt
    -- determine which value is worse (higher mean)
    local worse_val, better_val, worse_mean, better_mean
    if m1 > m2 then
      worse_val, better_val = vals[1], vals[2]
      worse_mean, better_mean = m1, m2
    else
      worse_val, better_val = vals[2], vals[1]
      worse_mean, better_mean = m2, m1
    end
    local d = ezr.cohen(groups[vals[1]].perf_list, groups[vals[2]].perf_list)
    table.insert(analyses, {
      col = c.name,
      worse_val = worse_val,
      better_val = better_val,
      worse_mean = worse_mean,
      better_mean = better_mean,
      d = d
    })
  end
end

-- sort by absolute effect size descending
table.sort(analyses, function(a,b) return math.abs(a.d) > math.abs(b.d) end)

-- print top 3
for i=1, math.min(3, #analyses) do
  local a = analyses[i]
  print(string.format("%s: Worse value '%s' (mean=%s) vs better value '%s' (mean=%s), effect size d=%.2f",
    a.col, a.worse_val, ezr.round(a.worse_mean),
    a.better_val, ezr.round(a.better_mean),
    ezr.round(a.d)))
end

-- ------------------------------------------------------------------
-- APACHE-10 – Frequent bad combinations
-- ------------------------------------------------------------------
print("\n== APACHE-10 ==")

-- Identify poor rows
local poor_rows = {}
for _, r in ipairs(tbl.rows) do
  if r[perf_at] > poor_thresh then
    table.insert(poor_rows, r)
  end
end

-- Prepare binary column indices for combos
local bin_indices = {}
for at, c in ipairs(tbl.cols.all) do
  if c ~= perf_col and is_binary(c) then
    bin_indices[#bin_indices+1] = {at=at, name=c.name}
  end
end

-- Collect statistics for each pair (1,1) combo
local combos = {}
for _, a in ipairs(bin_indices) do
  for _, b in ipairs(bin_indices) do
    if a.at < b.at then
      local key = a.name .. "=" .. a.name .. "," .. b.name .. "=" .. b.name
      combos[key] = {total=0, poor=0, sum=0, count=0}
    end
  end
end

-- Iterate rows
for _, r in ipairs(tbl.rows) do
  local perf = r[perf_at]
  local is_poor = perf > poor_thresh
  for _, a in ipairs(bin_indices) do
    for _, b in ipairs(bin_indices) do
      if a.at < b.at and r[a.at]==1 and r[b.at]==1 then
        local key = a.name .. "=" .. a.name .. "," .. b.name .. "=" .. b.name
        local c = combos[key]
        c.total = c.total + 1
        c.sum = c.sum + perf
        c.count = c.count + 1
        if is_poor then c.poor = c.poor + 1 end
      end
    end
  end
end

-- Build list of combos with stats
local combo_list = {}
for k,v in pairs(combos) do
  if v.total > 0 then
    local mean_combo = v.sum / v.total
    local ratio = v.poor / v.total
    table.insert(combo_list, {
      key = k,
      total = v.total,
      poor = v.poor,
      ratio = ratio,
      mean = mean_combo
    })
  end
end

-- Sort by ratio descending
table.sort(combo_list, function(a,b) return a.ratio > b.ratio end)

-- Print top 3 combos (or fewer if not enough)
for i=1, math.min(3, #combo_list) do
  local c = combo_list[i]
  print(string.format("%s: %d poor / %d total (%.1f%%)",
    c.key, c.poor, c.total, 100*c.ratio))
  print(string.format("  Mean performance of combo: %s",
    ezr.round(c.mean)))
  print(string.format("  Overall mean performance: %s",
    ezr.round(overall_mean)))
  print(string.format("  Difference: %s",
    ezr.round(c.mean - overall_mean)))
end
