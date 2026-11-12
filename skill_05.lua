#!/usr/bin/env lua
-- Skill 05: Analysis of poor fuel efficiency (CAR-09) and
-- overall performance (CAR-10)
-- Usage: lua skill_05.lua <csv path>

local ezr = require("ezr")
local csv  = ezr.csv
local round = ezr.round
local keysort = ezr.keysort
local sorted = ezr.sorted

-- ------------------------------------------------------------------
-- Utility: compute basic statistics for a list of numbers
local function stats(vals)
  local n, sum, sumsq = 0, 0, 0
  local minv, maxv = math.huge, -math.huge
  for _, v in ipairs(vals) do
    if v ~= nil then
      n = n + 1
      sum = sum + v
      sumsq = sumsq + v * v
      if v < minv then minv = v end
      if v > maxv then maxv = v end
    end
  end
  local mean = n > 0 and sum / n or nil
  local var = n > 1 and (sumsq - sum * sum / n) / (n - 1) or 0
  local std = math.sqrt(var)
  return mean, std, minv, maxv, n
end

-- ------------------------------------------------------------------
-- Read CSV
local path = arg[1]
if not path then
  io.stderr:write("Usage: lua skill_05.lua <csv path>\n")
  os.exit(1)
end

local rows_iter = csv(path)
local headers = rows_iter()
if not headers then
  io.stderr:write("Empty CSV\n")
  os.exit(1)
end

-- Load all rows
local rows = {}
for r in rows_iter do
  table.insert(rows, r)
end

-- ------------------------------------------------------------------
-- Identify numeric and symbolic columns
local num_cols = {}
local sym_cols = {}
local first = rows[1]
for _, col in ipairs(headers) do
  local val = first[col]
  if type(val) == "number" then
    table.insert(num_cols, col)
  else
    table.insert(sym_cols, col)
  end
end

-- ------------------------------------------------------------------
-- Compute min/max for numeric columns
local col_min = {}
local col_max = {}
for _, col in ipairs(num_cols) do
  local vals = {}
  for _, r in ipairs(rows) do
    local v = r[col]
    if v ~= nil then
      table.insert(vals, v)
    end
  end
  _, _, mn, mx, _ = stats(vals)
  col_min[col] = mn
  col_max[col] = mx
end

-- ------------------------------------------------------------------
-- Overall performance score (same as earlier skills)
local function norm(v, minv, maxv, reverse)
  if maxv == minv then return 0 end
  local n = (v - minv) / (maxv - minv)
  return reverse and (1 - n) or n
end

-- Pre‑compute min/max for the three metrics used in the score
local weight_vals, mpg_vals, acc_vals = {}, {}, {}
for _, r in ipairs(rows) do
  table.insert(weight_vals, r["Lbs-"])
  table.insert(mpg_vals, r["Mpg+"])
  table.insert(acc_vals, r["Acc+"])
end
local _, _, wmin, wmax, _ = stats(weight_vals)
local _, _, mmin, mmax, _ = stats(mpg_vals)
local _, _, amin, amax, _ = stats(acc_vals)

-- Assign score to each row
for _, r in ipairs(rows) do
  local w = r["Lbs-"]
  local m = r["Mpg+"]
  local a = r["Acc+"]
  r.score = norm(w, wmin, wmax, true) +
            norm(m, mmin, mmax, false) +
            norm(a, amin, amax, false)
end

-- ------------------------------------------------------------------
-- CAR-09: Correlation of numeric features with fuel efficiency (Mpg+)
print("== CAR-09 ==")
local correlations = {}
for _, col in ipairs(num_cols) do
  if col ~= "Mpg+" then
    local sumx, sumy, sumx2, sumy2, sumxy, n = 0, 0, 0, 0, 0, 0
    for _, r in ipairs(rows) do
      local x = r[col]
      local y = r["Mpg+"]
      if x ~= nil and y ~= nil then
        sumx = sumx + x
        sumy = sumy + y
        sumx2 = sumx2 + x * x
        sumy2 = sumy2 + y * y
        sumxy = sumxy + x * y
        n = n + 1
      end
    end
    if n > 1 then
      local numerator   = n * sumxy - sumx * sumy
      local denominator = math.sqrt((n * sumx2 - sumx * sumx) *
                                   (n * sumy2 - sumy * sumy))
      if denominator ~= 0 then
        local corr = numerator / denominator
        table.insert(correlations, {col=col, corr=corr})
      end
    end
  end
end

-- Sort by absolute correlation descending
table.sort(correlations, function(a,b)
  return math.abs(a.corr) > math.abs(b.corr)
end)

-- Print top 3
for i = 1, math.min(3, #correlations) do
  local c = correlations[i]
  print(string.format("%-15s  corr = %s",
        c.col, round(c.corr)))
end

-- ------------------------------------------------------------------
-- CAR-10: Characteristics of unfavorable vs favorable overall performance
print("\n== CAR-10 ==")

-- Find median score
local sorted_rows = sorted(rows, function(a,b) return a.score < b.score end)
local mid_idx = math.floor(#sorted_rows / 2) + 1
local median = sorted_rows[mid_idx].score

-- Split into groups
local unfav, fav = {}, {}
for _, r in ipairs(rows) do
  if r.score <= median then
    table.insert(unfav, r)
  else
    table.insert(fav, r)
  end
end

-- Helper to compute mean of a column within a group
local function group_mean(group, col)
  local s, n = 0, 0
  for _, r in ipairs(group) do
    local v = r[col]
    if v ~= nil then
      s = s + v
      n = n + 1
    end
  end
  return n > 0 and s / n or nil
end

-- Compute numeric differences
local diffs = {}
for _, col in ipairs(num_cols) do
  local m1 = group_mean(unfav, col)
  local m2 = group_mean(fav, col)
  if m1 ~= nil and m2 ~= nil then
    local diff = m1 - m2
    table.insert(diffs, {col=col, unfav=m1, fav=m2, diff=diff})
  end
end

-- Sort by absolute difference
table.sort(diffs, function(a,b)
  return math.abs(a.diff) > math.abs(b.diff)
end)

-- Print top 3 numeric contrasts
for i = 1, math.min(3, #diffs) do
  local d = diffs[i]
  print(string.format("%-15s  unfav = %s  fav = %s  diff = %s",
        d.col,
        round(d.unfav),
        round(d.fav),
        round(d.diff)))
end

-- Symbolic columns: most common value in each group
if #sym_cols > 0 then
  print("\nSymbolic feature modes:")
  for _, col in ipairs(sym_cols) do
    local mode_unfav = nil
    local mode_fav = nil
    local freq = {}
    for _, r in ipairs(unfav) do
      local v = r[col] or "?"
      freq[v] = (freq[v] or 0) + 1
    end
    if next(freq) then
      local maxv, maxc = nil, 0
      for v,c in pairs(freq) do
        if c > maxc then maxc, maxv = c, v end
      end
      mode_unfav = maxv
    end
    freq = {}
    for _, r in ipairs(fav) do
      local v = r[col] or "?"
      freq[v] = (freq[v] or 0) + 1
    end
    if next(freq) then
      local maxv, maxc = nil, 0
      for v,c in pairs(freq) do
        if c > maxc then maxc, maxv = c, v end
      end
      mode_fav = maxv
    end
    print(string.format("%-15s  unfav mode = %s  fav mode = %s",
          col,
          mode_unfav or "N/A",
          mode_fav or "N/A"))
  end
end
