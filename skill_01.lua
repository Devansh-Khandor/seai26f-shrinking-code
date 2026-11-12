#!/usr/bin/env lua
-- Skill 01 – Apache performance summary & outliers
-- Requires the ezr library (ezr.lua & ezr-lib.lua in the same dir)

local ezr = require"ezr"

-- ------------------------------------------------------------------
-- Get the CSV file path from the command line
-- ------------------------------------------------------------------
local csv_path = arg[1]
assert(csv_path, "Usage: lua skill_01.lua <csv path>")

-- ------------------------------------------------------------------
-- Load the data into an ezr table
-- ------------------------------------------------------------------
local tbl = ezr.Tbl(ezr.csv(csv_path))

-- ------------------------------------------------------------------
-- Find the performance column (the goal column)
-- ------------------------------------------------------------------
local perf_col = tbl.cols.klass      -- the column whose name ends with '-'
local perf_at   = perf_col.at

-- ------------------------------------------------------------------
-- Gather all performance values
-- ------------------------------------------------------------------
local perf_vals = {}
for _, r in ipairs(tbl.rows) do
  perf_vals[#perf_vals + 1] = r[perf_at]
end

-- ------------------------------------------------------------------
-- Compute summary statistics
-- ------------------------------------------------------------------
local mean_perf = perf_col:mid()           -- mean (since goal is to minimize)
local sd_perf   = perf_col:div()           -- standard deviation
local best_perf = math.min(unpack(perf_vals))   -- lowest value
local worst_perf= math.max(unpack(perf_vals))  -- highest value

-- ------------------------------------------------------------------
-- APACHE-01 – Basic summary
-- ------------------------------------------------------------------
print("== APACHE-01 ==")
print(string.format("Typical performance (mean): %s",
                    ezr.round(mean_perf)))
print(string.format("Best performance (min):   %s",
                    ezr.round(best_perf)))
print(string.format("Worst performance (max):  %s",
                    ezr.round(worst_perf)))

-- ------------------------------------------------------------------
-- APACHE-02 – Unusually favorable configurations
-- ------------------------------------------------------------------
print("\n== APACHE-02 ==")

-- Threshold: 2 standard deviations below the mean (since lower is better)
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

if #fav_cfgs == 0 then
  print("No configurations found with performance more than 2 SD below the mean.")
else
  print(string.format("Threshold (mean - 2*SD): %s",
                      ezr.round(thresh)))
  for _, e in ipairs(fav_cfgs) do
    print(string.format("Row %d: %s (Performance=%s)",
                        e.idx, ezr.show(e.cfg), ezr.round(e.perf)))
  end
end
