#!/usr/bin/env lua
-- skill_01.lua
-- Run: lua skill_01.lua <csv_path>
--
-- Provides two NASA requests:
--   NASA-01 – Summary of project size, effort and defects.
--   NASA-02 – Projects with favorable size/effort/defect combos.

local ezr = require "ezr"

-- Helpers from ezr
local csv   = ezr.csv
local Tbl   = ezr.Tbl
local push  = ezr.push
local sorted = ezr.sorted

-- ------------------------------------------------------------------
-- Read data
-- ------------------------------------------------------------------
local path = arg[1]
if not path then
  io.stderr:write("Usage: lua skill_01.lua <csv_path>\n")
  os.exit(1)
end

local tbl = Tbl(csv(path))

-- ------------------------------------------------------------------
-- Utility functions
-- ------------------------------------------------------------------
local function col_stats(tbl, at)
  local col = tbl.cols.all[at]
  local minv, maxv = nil, nil
  for _, r in ipairs(tbl.rows) do
    local v = r[at]
    if not minv or v < minv then minv = v end
    if not maxv or v > maxv then maxv = v end
  end
  return {
    min  = minv,
    max  = maxv,
    mean = col:mid(),
    std  = col:div()
  }
end

local function median_from_col(tbl, at)
  local vals = {}
  for _, r in ipairs(tbl.rows) do push(vals, r[at]) end
  local s = sorted(vals)
  local n = #s
  if n == 0 then return nil end
  if n % 2 == 1 then
    return s[(n + 1) / 2]
  else
    return (s[n / 2] + s[n / 2 + 1]) / 2
  end
end

-- ------------------------------------------------------------------
-- NASA-01: Summary
-- ------------------------------------------------------------------
local size_at   = 26   -- Kloc+
local effort_at = 27   -- Effort-
local defect_at = 28   -- Defects-

local size_stats   = col_stats(tbl, size_at)
local effort_stats = col_stats(tbl, effort_at)
local defect_stats = col_stats(tbl, defect_at)

-- Find best and worst by lowest/highest effort
local best_proj, worst_proj
for _, r in ipairs(tbl.rows) do
  if not best_proj or r[effort_at] < best_proj[effort_at] then
    best_proj = r
  end
  if not worst_proj or r[effort_at] > worst_proj[effort_at] then
    worst_proj = r
  end
end

print("== NASA-01 ==")
print(string.format(
  "Typical Project Size (KLOC): mean=%.2f, min=%.2f, max=%.2f, std=%.2f",
  size_stats.mean, size_stats.min, size_stats.max, size_stats.std))
print(string.format(
  "Typical Development Effort (Man-Months): mean=%.2f, min=%.2f, max=%.2f, std=%.2f",
  effort_stats.mean, effort_stats.min, effort_stats.max, effort_stats.std))
print(string.format(
  "Typical Defect Count: mean=%.2f, min=%d, max=%d, std=%.2f",
  defect_stats.mean, defect_stats.min, defect_stats.max, defect_stats.std))
print(string.format(
  "Best Project: id=%s, size=%.1f, effort=%.1f, defects=%d",
  best_proj[1], best_proj[size_at], best_proj[effort_at], best_proj[defect_at]))
print(string.format(
  "Worst Project: id=%s, size=%.1f, effort=%.1f, defects=%d",
  worst_proj[1], worst_proj[size_at], worst_proj[effort_at], worst_proj[defect_at]))

-- ------------------------------------------------------------------
-- NASA-02: Favorable projects
-- ------------------------------------------------------------------
local size_med   = median_from_col(tbl, size_at)
local effort_med = median_from_col(tbl, effort_at)
local defect_med = median_from_col(tbl, defect_at)

local favorable = {}
for _, r in ipairs(tbl.rows) do
  if r[size_at]   <= size_med and
     r[effort_at] <= effort_med and
     r[defect_at] <= defect_med then
    push(favorable, {
      id     = r[1],
      size   = r[size_at],
      effort = r[effort_at],
      defect = r[defect_at]
    })
  end
end

print("\n== NASA-02 ==")
if #favorable == 0 then
  print("No projects meet the favorable criteria.")
else
  print(string.format(
    "Projects with size ≤ %.2f, effort ≤ %.2f, defects ≤ %.2f:",
    size_med, effort_med, defect_med))
  for _, p in ipairs(favorable) do
    print(string.format(
      "  id=%s, size=%.1f, effort=%.1f, defects=%d",
      p.id, p.size, p.effort, p.defect))
  end
end
