#!/usr/bin/env lua
-- skill_05.lua
-- Run: lua skill_05.lua <csv_path>
--
-- Provides two NASA requests:
--   NASA-09 – Identify characteristics most strongly associated
--             with unusually high development effort.
--   NASA-10 – Identify characteristics most strongly associated
--             with unusually high defect counts.

local ezr = require "ezr"

-- Helpers from ezr
local csv        = ezr.csv
local Tbl        = ezr.Tbl
local push       = ezr.push
local sorted     = ezr.sorted
local map        = ezr.map
local kap        = ezr.kap
local ceil       = math.ceil
local sqrt       = math.sqrt

-- ------------------------------------------------------------------
-- Read data
-- ------------------------------------------------------------------
local path = arg[1]
if not path then
  io.stderr:write("Usage: lua skill_05.lua <csv_path>\n")
  os.exit(1)
end

local tbl = Tbl(csv(path))

-- ------------------------------------------------------------------
-- Utility helpers -----------------------------------------------
local function col_stats(tbl, at)
  local col = tbl.cols.all[at]
  return {
    min  = col.min,
    max  = col.max,
    mean = col:mid(),
    std  = col:div()
  }
end

-- mean of a numeric column in a given set of rows
local function col_mean(rows, at)
  local sum, n = 0, 0
  for _, r in ipairs(rows) do
    local v = r[at]
    if v ~= "?" then sum = sum + v; n = n + 1 end
  end
  return n > 0 and sum / n or nil
end

-- mode and its frequency in a set of rows (only for SYM columns)
local function sym_mode(rows, at)
  local freq = {}
  local maxc, mode = 0, nil
  for _, r in ipairs(rows) do
    local v = r[at]
    if v ~= "?" then
      freq[v] = (freq[v] or 0) + 1
      if freq[v] > maxc then
        maxc, mode = freq[v], v
      end
    end
  end
  return mode, maxc
end

-- difference between high‑group and overall characteristics
local function analyze_column(tbl, at, high_rows, overall_rows)
  local col = tbl.cols.all[at]
  local mt = getmetatable(col).__index
  local name = col.name
  if mt == NUM then
    local overall_mean = col:mid()
    local high_mean = col_mean(high_rows, at)
    if high_mean and overall_mean then
      return {name=name, type="num",
              diff=high_mean - overall_mean,
              abs=math.abs(high_mean - overall_mean)}
    end
  elseif mt == SYM then
    local overall_mode, overall_cnt = sym_mode(overall_rows, at)
    local high_mode, high_cnt = sym_mode(high_rows, at)
    local diff = 0
    if high_mode == overall_mode then
      diff = high_cnt - overall_cnt
    else
      diff = high_cnt
    end
    return {name=name, type="sym",
            diff=diff,
            abs=math.abs(diff)}
  end
end

-- ------------------------------------------------------------------
-- Identify unusually high effort projects --------------------------
local effort_at = 27   -- Effort- column
local effort_stats = col_stats(tbl, effort_at)
local n = #tbl.rows
local effort_se = effort_stats.std / sqrt(n)
local effort_thresh = effort_stats.mean + 1.96 * effort_se

local high_eff_rows = {}
for _, r in ipairs(tbl.rows) do
  if r[effort_at] > effort_thresh then
    push(high_eff_rows, r)
  end
end

print("== NASA-09 ==")
print(string.format(
  "Unusually high effort projects: effort > %.2f (mean+1.96*SE)",
  effort_thresh))
print(string.format("%d projects found out of %d records",
  #high_eff_rows, n))
if #high_eff_rows == 0 then
  print("No unusually high effort projects detected.")
else
  -- analyze each column
  local analyses = {}
  for at, col in pairs(tbl.cols.all) do
    if at ~= effort_at and at ~= 28 then   -- skip effort & defect
      local a = analyze_column(tbl, at, high_eff_rows, tbl.rows)
      if a then push(analyses, a) end
    end
  end
  -- sort by absolute difference descending
  analyses = sorted(analyses, function(a,b) return a.abs > b.abs end)
  print("\nCharacteristics most associated with high effort:")
  for i, a in ipairs(analyses) do
    if i > 5 then break end
    if a.type == "num" then
      print(string.format(
        "  %s: mean %.3f vs overall %.3f (Δ=%.3f)",
        a.name, col_mean(high_eff_rows, a.at or nil),
        tbl.cols.all[a.at or nil]:mid(), a.diff))
    else
      print(string.format(
        "  %s: mode count Δ=%d (mode high='%s', overall='%s')",
        a.name, a.diff, a.mode, a.overall_mode))
    end
  end
end

-- ------------------------------------------------------------------
-- Identify unusually high defect projects ----------------------------
local defect_at = 28   -- Defects- column
local defect_stats = col_stats(tbl, defect_at)
local defect_se = defect_stats.std / sqrt(n)
local defect_thresh = defect_stats.mean + 1.96 * defect_se

local high_def_rows = {}
for _, r in ipairs(tbl.rows) do
  if r[defect_at] > defect_thresh then
    push(high_def_rows, r)
  end
end

print("\n== NASA-10 ==")
print(string.format(
  "Unusually high defect projects: defects > %.2f (mean+1.96*SE)",
  defect_thresh))
print(string.format("%d projects found out of %d records",
  #high_def_rows, n))
if #high_def_rows == 0 then
  print("No unusually high defect projects detected.")
else
  local analyses = {}
  for at, col in pairs(tbl.cols.all) do
    if at ~= defect_at and at ~= 27 then   -- skip defect & effort
      local a = analyze_column(tbl, at, high_def_rows, tbl.rows)
      if a then push(analyses, a) end
    end
  end
  analyses = sorted(analyses, function(a,b) return a.abs > b.abs end)
  print("\nCharacteristics most associated with high defect count:")
  for i, a in ipairs(analyses) do
    if i > 5 then break end
    if a.type == "num" then
      print(string.format(
        "  %s: mean %.3f vs overall %.3f (Δ=%.3f)",
        a.name, col_mean(high_def_rows, a.at or nil),
        tbl.cols.all[a.at or nil]:mid(), a.diff))
    else
      print(string.format(
        "  %s: mode count Δ=%d (mode high='%s', overall='%s')",
        a.name, a.diff, a.mode, a.overall_mode))
    end
  end
end
