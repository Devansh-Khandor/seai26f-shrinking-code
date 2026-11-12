#!/usr/bin/env lua
--[[
  skill_01.lua
  A lightweight analysis of the NASA 93 demo project data.
  Usage:  lua skill_01.lua <csv-path>
  Prints:
    NASA-01: summary of size, effort, defects and best/worst projects
    NASA-02: projects with unusually favorable size/effort/defect trade‑offs
--]]

local ezr = require "ezr"

local function abort(msg)
  io.stderr:write(msg .. "\n")
  os.exit(1)
end

local csv_path = arg[1]
if not csv_path then abort("Usage: lua skill_01.lua <csv-path>") end

-- Load all rows
local rows_iter = ezr.csv(csv_path)
local rows = {}
for r in rows_iter do
  table.insert(rows, r)
end
if #rows == 0 then abort("No data found in " .. csv_path) end

-- Build a table structure for convenient column access
local tbl = ezr.Tbl(rows)

-- Map column names to column objects
local col_map = {}
for i, c in ipairs(tbl.cols.all) do
  local name = tbl.cols.names[i]
  col_map[name] = c
end

-- Helper: round numbers using ezr.round
local function fmt(v)
  return ezr.round(v)
end

-- Compute statistics for a numeric column
local function compute_stats(col, rows)
  local vals = {}
  for _, row in ipairs(rows) do
    local v = row[col.at]
    if v ~= "?" then table.insert(vals, v) end
  end
  local n = #vals
  if n == 0 then return {n=0, mean=0, std=0, median=0, min=0, max=0} end
  local sum = 0
  for _, v in ipairs(vals) do sum = sum + v end
  local mean = sum / n
  local sumsq = 0
  for _, v in ipairs(vals) do sumsq = sumsq + (v - mean)^2 end
  local std = n > 1 and math.sqrt(sumsq / (n - 1)) or 0
  table.sort(vals)
  local median = vals[math.floor((n + 1) / 2)]
  local minv = vals[1]
  local maxv = vals[n]
  return {n = n, mean = mean, std = std, median = median, min = minv, max = maxv}
end

-- Columns of interest
local sz_col    = col_map["Kloc+"]
local eff_col   = col_map["Effort-"]
local def_col   = col_map["Defects-"]

local sz_stats  = compute_stats(sz_col,  rows)
local eff_stats = compute_stats(eff_col, rows)
local def_stats = compute_stats(def_col, rows)

-- Gather per-row metrics and overall ratio stats
local min_ratio = math.huge
local max_ratio = -math.huge
local min_row   = nil
local max_row   = nil

local ratio_sum   = 0
local ratio_n     = 0
local ratio_sumsq = 0

for _, row in ipairs(rows) do
  local size   = row[sz_col.at]
  local effort = row[eff_col.at]
  local defects = row[def_col.at]
  if size and effort and defects then
    local ratio = effort / size
    if ratio < min_ratio then min_ratio = ratio; min_row = row end
    if ratio > max_ratio then max_ratio = ratio; max_row = row end
    ratio_sum = ratio_sum + ratio
    ratio_n = ratio_n + 1
    ratio_sumsq = ratio_sumsq + ratio * ratio
  end
end

local ratio_mean = ratio_sum / ratio_n
local ratio_std  = ratio_n > 1 and math.sqrt((ratio_sumsq - ratio_sum * ratio_sum / ratio_n) / (ratio_n - 1)) or 0
local favorable_thresh = ratio_mean - ratio_std

-- Find projects that are unusually favorable (ratio below threshold)
local favorable_projects = {}
for _, row in ipairs(rows) do
  local size   = row[sz_col.at]
  local effort = row[eff_col.at]
  local defects = row[def_col.at]
  if size and effort and defects then
    local ratio = effort / size
    if ratio <= favorable_thresh then
      table.insert(favorable_projects, {id=row[1], size=size, effort=effort,
                                        defects=defects, ratio=ratio})
    end
  end
end

-- Sort favorable projects by ratio ascending
table.sort(favorable_projects, function(a, b) return a.ratio < b.ratio end)

------------------------------------------------------------------
-- Output NASA-01: summary
------------------------------------------------------------------
print("== NASA-01 ==")
print(string.format("Project Size (Kloc+): mean=%s median=%s std=%s min=%s max=%s",
        fmt(sz_stats.mean), fmt(sz_stats.median), fmt(sz_stats.std), sz_stats.min, sz_stats.max))
print(string.format("Development Effort (Effort-): mean=%s median=%s std=%s min=%s max=%s",
        fmt(eff_stats.mean), fmt(eff_stats.median), fmt(eff_stats.std), eff_stats.min, eff_stats.max))
print(string.format("Defect Count (Defects-): mean=%s median=%s std=%s min=%s max=%s",
        fmt(def_stats.mean), fmt(def_stats.median), fmt(def_stats.std), def_stats.min, def_stats.max))
print("Best outcome (lowest effort/size):")
print(string.format("  id=%s  Effort-=%s  Defects-=%s  Kloc+=%s  ratio=%s",
        min_row[1], fmt(min_row[eff_col.at]), fmt(min_row[def_col.at]),
        fmt(min_row[sz_col.at]), fmt(min_ratio)))
print("Worst outcome (highest effort/size):")
print(string.format("  id=%s  Effort-=%s  Defects-=%s  Kloc+=%s  ratio=%s",
        max_row[1], fmt(max_row[eff_col.at]), fmt(max_row[def_col.at]),
        fmt(max_row[sz_col.at]), fmt(max_ratio)))

------------------------------------------------------------------
-- Output NASA-02: favorable projects
------------------------------------------------------------------
print("\n== NASA-02 ==")
if #favorable_projects == 0 then
  print("No projects found with unusually favorable size/effort/defect trade‑offs.")
else
  print(string.format("%-5s %-8s %-9s %-9s %-7s", "id", "Kloc+", "Effort-", "Defects-", "ratio"))
  for _, p in ipairs(favorable_projects) do
    print(string.format("%-5s %-8s %-9s %-9s %-7s",
            p.id, fmt(p.size), fmt(p.effort), fmt(p.defects), fmt(p.ratio)))
  end
end
