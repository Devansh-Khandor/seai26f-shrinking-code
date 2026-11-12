#!/usr/bin/env lua
-- skill_02.lua
-- Run: lua skill_02.lua <csv_path>
--
-- Provides two NASA requests:
--   NASA-03 – Estimate development effort required for a new project.
--   NASA-04 – Estimate the likely number of defects for a new project,
--             together with an estimate of the uncertainty.

local ezr = require "ezr"

-- Helpers from ezr
local csv   = ezr.csv
local Tbl   = ezr.Tbl
local push  = ezr.push
local sorted = ezr.sorted
local sqrt  = math.sqrt

-- ------------------------------------------------------------------
-- Read data
-- ------------------------------------------------------------------
local path = arg[1]
if not path then
  io.stderr:write("Usage: lua skill_02.lua <csv_path>\n")
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

-- ------------------------------------------------------------------
-- NASA-03: Estimate effort
-- ------------------------------------------------------------------
local effort_at = 27   -- Effort- column
local effort_stats = col_stats(tbl, effort_at)
local n = #tbl.rows

-- 95% confidence interval using normal approximation
local se = effort_stats.std / sqrt(n)
local ci_low  = effort_stats.mean - 1.96 * se
local ci_high = effort_stats.mean + 1.96 * se

print("== NASA-03 ==")
print(string.format(
  "Estimated effort: %.2f +/- %.2f man-months (mean ± std)",
  effort_stats.mean, effort_stats.std))
print(string.format(
  "Typical range: min=%.2f, max=%.2f",
  effort_stats.min, effort_stats.max))
print(string.format(
  "Confidence interval (95%%): [%.2f, %.2f] man-months",
  ci_low, ci_high))

-- ------------------------------------------------------------------
-- NASA-04: Estimate defects
-- ------------------------------------------------------------------
local defect_at = 28   -- Defects- column
local defect_stats = col_stats(tbl, defect_at)

-- 95% confidence interval for defect count
local defect_se = defect_stats.std / sqrt(n)
local defect_ci_low  = defect_stats.mean - 1.96 * defect_se
local defect_ci_high = defect_stats.mean + 1.96 * defect_se

print("\n== NASA-04 ==")
print(string.format(
  "Estimated defects: %.2f +/- %.2f (mean ± std)",
  defect_stats.mean, defect_stats.std))
print(string.format(
  "Typical range: min=%d, max=%d",
  defect_stats.min, defect_stats.max))
print(string.format(
  "Confidence interval (95%%): [%.2f, %.2f] defects",
  defect_ci_low, defect_ci_high))
