#!/usr/bin/env lua
-- skill_04.lua
-- Run: lua skill_04.lua <csv_path>
--
-- Provides two NASA requests:
--   NASA-07 – Flag proposals with unusually high predicted effort or defects.
--   NASA-08 – Update and display the latest historical summaries.

local ezr = require "ezr"

-- Helpers from ezr
local csv     = ezr.csv
local Tbl     = ezr.Tbl
local push    = ezr.push
local sorted  = ezr.sorted
local ceil    = math.ceil
local sqrt    = math.sqrt

-- ------------------------------------------------------------------
-- Read data
-- ------------------------------------------------------------------
local path = arg[1]
if not path then
  io.stderr:write("Usage: lua skill_04.lua <csv_path>\n")
  os.exit(1)
end

local tbl = Tbl(csv(path))

-- ------------------------------------------------------------------
-- Utility functions (copy from skill_01)
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
-- Indices of columns (as per the dataset)
-- ------------------------------------------------------------------
local effort_at  = 27   -- Effort-
local defect_at  = 28   -- Defects-
local size_at    = 26   -- Kloc+

-- ------------------------------------------------------------------
-- NASA-07: Flag unusually high effort or defects
-- ------------------------------------------------------------------
local effort_stats = col_stats(tbl, effort_at)
local defect_stats = col_stats(tbl, defect_at)
local n = #tbl.rows

local effort_se = effort_stats.std / sqrt(n)
local defect_se = defect_stats.std / sqrt(n)

local effort_thresh = effort_stats.mean + 1.96 * effort_se
local defect_thresh = defect_stats.mean + 1.96 * defect_se

local flagged = {}
for _, r in ipairs(tbl.rows) do
  local eff = r[effort_at]
  local def = r[defect_at]
  if eff > effort_thresh or def > defect_thresh then
    push(flagged, {
      id     = r[1],
      effort = eff,
      defect = def
    })
  end
end

print("== NASA-07 ==")
print(string.format(
  "Thresholds: effort>%.2f, defects>%.2f (mean+1.96*SE)",
  effort_thresh, defect_thresh))
print(string.format(
  "Number of proposals flagged: %d of %d",
  #flagged, n))
if #flagged > 0 then
  print("Examples:")
  for _, p in ipairs(sorted(flagged, function(a,b) return a.id < b.id end)) do
    print(string.format(
      "  id=%s  effort=%.1f  defects=%d",
      p.id, p.effort, p.defect))
    if #flagged >= 5 then break end -- show up to 5
  end
end

-- ------------------------------------------------------------------
-- NASA-08: Updated historical summaries
-- ------------------------------------------------------------------
-- In this script the data file is always read fresh, so the
-- summaries below automatically reflect the latest records.
-- We simply recompute and print them.

local size_stats   = col_stats(tbl, size_at)
local effort_stats = col_stats(tbl, effort_at)
local defect_stats = col_stats(tbl, defect_at)

print("\n== NASA-08 ==")
print(string.format(
  "Updated Historical Summary (based on %d records)",
  n))
print(string.format(
  "Project Size (KLOC): mean=%.2f, min=%.2f, max=%.2f, std=%.2f",
  size_stats.mean, size_stats.min, size_stats.max, size_stats.std))
print(string.format(
  "Development Effort (Man-Months): mean=%.2f, min=%.2f, max=%.2f, std=%.2f",
  effort_stats.mean, effort_stats.min, effort_stats.max, effort_stats.std))
print(string.format(
  "Defect Count: mean=%.2f, min=%d, max=%d, std=%.2f",
  defect_stats.mean, defect_stats.min, defect_stats.max, defect_stats.std))
