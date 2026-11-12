#!/usr/bin/env lua
-- Skill 04 – Apache performance monitoring & incremental update
-- Requires the ezr library (ezr.lua & ezr-lib.lua in the same dir)

local ezr = require"ezr"

-- ------------------------------------------------------------------
-- Get the CSV file path from the command line
-- ------------------------------------------------------------------
local csv_path = arg[1]
assert(csv_path, "Usage: lua skill_04.lua <csv path>")

-- ------------------------------------------------------------------
-- Load the data into an ezr table
-- ------------------------------------------------------------------
local tbl = ezr.Tbl(ezr.csv(csv_path))

-- ------------------------------------------------------------------
-- Helper functions
-- ------------------------------------------------------------------
local function mean(lst)
  local sum = 0
  for _, v in ipairs(lst) do sum = sum + v end
  return sum / (#lst > 0 and #lst or 1)
end

local function stddev(lst, m)
  local sum = 0
  for _, v in ipairs(lst) do sum = sum + (v - m)^2 end
  return (#lst > 1 and math.sqrt(sum / (#lst - 1)) or 0)
end

-- ------------------------------------------------------------------
-- Identify key columns
-- ------------------------------------------------------------------
local perf_col = tbl.cols.klass      -- column whose name ends with '-'
local perf_at  = perf_col.at

-- ------------------------------------------------------------------
-- Collect all performance values
-- ------------------------------------------------------------------
local perf_vals = {}
for _, r in ipairs(tbl.rows) do
  perf_vals[#perf_vals + 1] = r[perf_at]
end

-- ------------------------------------------------------------------
-- APACHE-07 – Real‑time alert on unusually poor configurations
-- ------------------------------------------------------------------
print("== APACHE-07 ==")

-- Running statistics for previous rows
local prev_n      = 0
local prev_sum    = 0
local prev_sum_sq = 0
local alerts      = 0

-- If we want to print an overall summary at the end
local last_mean, last_sd

for idx, r in ipairs(tbl.rows) do
  local val = r[perf_at]

  if prev_n > 1 then
    local mean_prev = prev_sum / prev_n
    local sd_prev   = math.sqrt((prev_sum_sq - prev_n * mean_prev * mean_prev) / (prev_n - 1))
    local threshold = mean_prev + 2 * sd_prev

    if val > threshold then
      alerts = alerts + 1
      print(string.format(
        "Alert: Row %d has performance %s, above mean+2SD (threshold: %s).",
        idx, ezr.round(val), ezr.round(threshold)))
    end
  end

  -- update running sums for next iteration
  prev_n = prev_n + 1
  prev_sum = prev_sum + val
  prev_sum_sq = prev_sum_sq + val * val
  last_mean = prev_sum / prev_n
  last_sd   = math.sqrt((prev_sum_sq - prev_n * last_mean * last_mean) / (prev_n - 1))
end

-- Provide an overall summary even if no alerts were triggered
print(string.format(
  "Overall mean performance: %s, SD: %s",
  ezr.round(last_mean), ezr.round(last_sd)))

if alerts == 0 then
  print("No unusually poor performance detected.")
end

-- ------------------------------------------------------------------
-- APACHE-08 – Updated performance summary & recommended shortlist
-- ------------------------------------------------------------------
print("\n== APACHE-08 ==")

-- Summary statistics
local mean_perf = perf_col:mid()
local sd_perf   = perf_col:div()
local best_perf = math.min(unpack(perf_vals))
local worst_perf= math.max(unpack(perf_vals))

print(string.format("Mean performance: %s", ezr.round(mean_perf)))
print(string.format("Best performance: %s", ezr.round(best_perf)))
print(string.format("Worst performance: %s", ezr.round(worst_perf)))

-- Recommended configurations (2 SD below the mean)
local thresh = mean_perf - 2 * sd_perf
local fav_cfgs = {}
for idx, r in ipairs(tbl.rows) do
  local val = r[perf_at]
  if val <= thresh then
    local cfg = {}
    for name, col in pairs(tbl.cols.all) do
      cfg[name] = r[col.at]
    end
    table.insert(fav_cfgs, {idx=idx, cfg=cfg, perf=val})
  end
end

print(string.format("Threshold (mean - 2*SD): %s", ezr.round(thresh)))
if #fav_cfgs == 0 then
  print("No configurations fall below the threshold.")
else
  for _, e in ipairs(fav_cfgs) do
    print(string.format(
      "Row %d: %s (Performance=%s)",
      e.idx, ezr.show(e.cfg), ezr.round(e.perf)))
  end
end
