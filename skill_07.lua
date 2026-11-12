#!/usr/bin/env lua
-- skill_07.lua
-- Run: lua skill_07.lua <csv_path>
--
-- Provides two NASA requests:
--   NASA-13 – Identify promising combinations of project characteristics
--             that historically delivered more software with lower
--             development effort and fewer defects.
--   NASA-14 – Recommend which project proposals should be evaluated
--             first to discover promising alternatives within a
--             limited evaluation budget.

local ezr = require "ezr"

-- Helpers from ezr
local csv    = ezr.csv
local Tbl    = ezr.Tbl
local push   = ezr.push
local sorted = ezr.sorted
local keys   = ezr.keys
local cap    = math.floor
local sqrt   = math.sqrt

-- Settings from ezr
local the = ezr.the
local size_at    = 26   -- Kloc+
local effort_at  = 27   -- Effort-
local defect_at  = 28   -- Defects-

-- ------------------------------------------------------------------
-- Read data
-- ------------------------------------------------------------------
local path = arg[1]
if not path then
  io.stderr:write("Usage: lua skill_07.lua <csv_path>\n")
  os.exit(1)
end

local tbl = Tbl(csv(path))

-- ------------------------------------------------------------------
-- Utility helpers
-- ------------------------------------------------------------------
local function col_stats(tbl, at)
  local col = tbl.cols.all[at]
  return {
    min  = col.min,
    max  = col.max,
    mean = col:mid(),
    std  = col:div()
  }
end

-- median of a numeric column
local function median_from_col(tbl, at)
  local vals = {}
  for _, r in ipairs(tbl.rows) do
    local v = r[at]
    if v ~= "?" then
      push(vals, v)
    end
  end
  local s = sorted(vals)
  local n = #s
  if n == 0 then return nil end
  if n % 2 == 1 then
    return s[(n + 1) / 2]
  else
    return (s[n / 2] + s[n / 2 + 1]) / 2
  end
end

-- compute statistics for each value of a symbolic column
local function sym_group_stats(tbl, sym_at)
  local groups = {}
  for _, r in ipairs(tbl.rows) do
    local v = r[sym_at]
    if not groups[v] then groups[v] = {n = 0} end
    local g = groups[v]
    g.n = g.n + 1
    local sz = r[size_at]
    local ef = r[effort_at]
    local df = r[defect_at]
    g.size = (g.size or 0) + (sz ~= "?" and sz or 0)
    g.effort = (g.effort or 0) + (ef ~= "?" and ef or 0)
    g.defect = (g.defect or 0) + (df ~= "?" and df or 0)
  end
  for _, g in pairs(groups) do
    g.mean_size = g.size / g.n
    g.mean_effort = g.effort / g.n
    g.mean_defect = g.defect / g.n
  end
  return groups
end

-- ------------------------------------------------------------------
-- NASA-13: promising combinations
-- ------------------------------------------------------------------
print("== NASA-13 ==")

-- thresholds from overall medians
local size_med   = median_from_col(tbl, size_at)
local effort_med = median_from_col(tbl, effort_at)
local defect_med = median_from_col(tbl, defect_at)

local promising = {}

-- iterate over all symbolic columns
for at, c in pairs(tbl.cols.all) do
  if c.__index == ezr.SYM then
    local groups = sym_group_stats(tbl, at)
    for val, g in pairs(groups) do
      if g.mean_size >= size_med
        and g.mean_effort <= effort_med
        and g.mean_defect <= defect_med then
        local score = (g.mean_effort > 0 and g.mean_defect > 0)
          and g.mean_size / (g.mean_effort * g.mean_defect)
          or g.mean_size
        push(promising, {
          feature = c.name,
          value   = val,
          size    = g.mean_size,
          effort  = g.mean_effort,
          defect  = g.mean_defect,
          score   = score
        })
      end
    end
  end
end

-- pick top 3 by score
promising = sorted(promising, function(a,b) return a.score > b.score end)
for i = 1, math.min(3, #promising) do
  local p = promising[i]
  print(string.format(
    "%d) %s=%s: Avg Size=%.2f, Avg Effort=%.2f, Avg Defect=%.2f",
    i, p.feature, tostring(p.value), p.size, p.effort, p.defect))
end
if #promising == 0 then
  print("No groups satisfy the high‑size / low‑effort / low‑defect criteria.")
end

-- ------------------------------------------------------------------
-- NASA-14: recommend projects to evaluate first
-- ------------------------------------------------------------------
print("\n== NASA-14 ==")

local function score_row(r)
  local sz = r[size_at]
  local ef = r[effort_at]
  local df = r[defect_at]
  if sz == "?" or ef == "?" or df == "?" then
    return -1e9 -- treat missing as worst
  end
  local denom = ef + df
  if denom == 0 then return 1e9 end
  return sz / denom
end

local scored = {}
for _, r in ipairs(tbl.rows) do
  push(scored, {
    row   = r,
    score = score_row(r)
  })
end
scored = sorted(scored, function(a,b) return a.score > b.score end)

local budget = the.budget or 50
local to_eval = cap(#scored, budget)

print(string.format("Evaluation budget: %d projects to evaluate first.", budget))
print("Top projects to evaluate (ranked by promising score):")
for i = 1, to_eval do
  local s = scored[i]
  local r = s.row
  print(string.format(
    "%d) ID=%s | Size=%.1f | Effort=%.1f | Defect=%d | Score=%.4f",
    i, r[1], r[size_at], r[effort_at], r[defect_at], s.score))
end
