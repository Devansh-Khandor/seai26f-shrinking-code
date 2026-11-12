#!/usr/bin/env lua
-- Skill 06 – Apache configuration explanations
-- Requires the ezr library (ezr.lua & ezr-lib.lua in the same dir)

local ezr = require"ezr"

-- ------------------------------------------------------------------
-- Get the CSV file path from the command line
-- ------------------------------------------------------------------
local csv_path = arg[1]
assert(csv_path, "Usage: lua skill_06.lua <csv path>")

-- ------------------------------------------------------------------
-- Load the data into an ezr table
-- ------------------------------------------------------------------
local tbl = ezr.Tbl(ezr.csv(csv_path))

-- ------------------------------------------------------------------
-- Helpers ----------------------------------------------------------
-- ------------------------------------------------------------------
-- Return the row with the minimum performance value (lowest is best)
local function best_row(rows, perf_at)
  local best = rows[1]
  for _, r in ipairs(rows) do
    if r[perf_at] < best[perf_at] then best = r end
  end
  return best
end

-- Return the row with the maximum performance value (worst)
local function worst_row(rows, perf_at)
  local worst = rows[1]
  for _, r in ipairs(rows) do
    if r[perf_at] > worst[perf_at] then worst = r end
  end
  return worst
end

-- Distance in configuration space (independent columns)
local function cfg_distance(r1, r2)
  return tbl:distx(r1, r2)
end

-- Sort rows by distance to a given reference row
local function nearest_rows(ref, rows, k)
  local t = {}
  for _, r in ipairs(rows) do
    if r ~= ref then
      table.insert(t, {row=r, dist=cfg_distance(ref, r)})
    end
  end
  table.sort(t, function(a,b) return a.dist < b.dist end)
  local res = {}
  for i=1, math.min(k, #t) do
    table.insert(res, t[i].row)
  end
  return res
end

-- Return a table describing differences between two rows
local function diff_rows(r1, r2)
  local diffs = {}
  for _, col in ipairs(tbl.cols.x) do
    local v1, v2 = r1[col.at], r2[col.at]
    if v1 ~= v2 then
      diffs[col.name] = {v1=v1, v2=v2}
    end
  end
  return diffs
end

-- Pretty print a row as "col=value" pairs
local function row_to_str(r)
  local parts = {}
  for _, col in ipairs(tbl.cols.x) do
    parts[#parts+1] = col.name .. "=" .. tostring(r[col.at])
  end
  parts[#parts+1] = tbl.cols.klass.name .. "=" .. tostring(r[tbl.cols.klass.at])
  return table.concat(parts, ", ")
end

-- ------------------------------------------------------------------
-- APACHE-11 – Explain a recommended configuration
-- ------------------------------------------------------------------
print("== APACHE-11 ==")

local perf_at = tbl.cols.klass.at
local best = best_row(tbl.rows, perf_at)
local best_perf = best[perf_at]

print(string.format("Recommended configuration: %s",
                    row_to_str(best)))
print(string.format("Performance value: %s",
                    ezr.round(best_perf)))

-- Find nearest neighbours (configuration‑wise)
local neighbors = nearest_rows(best, tbl.rows, 3)
print("\nThree nearest configurations:")
for _, nbr in ipairs(neighbors) do
  local d = cfg_distance(best, nbr)
  print(string.format("- %s (dist=%.3f, performance=%s)",
          row_to_str(nbr), d, ezr.round(nbr[perf_at])))
  local diffs = diff_rows(best, nbr)
  if next(diffs) then
    print("  Differences:")
    for name, pair in pairs(diffs) do
      print(string.format("    %s: %s -> %s",
                name, tostring(pair.v1), tostring(pair.v2)))
    end
  end
end

-- ------------------------------------------------------------------
-- APACHE-12 – Explain differing assessments of two configs
-- ------------------------------------------------------------------
print("\n== APACHE-12 ==")

local worst = worst_row(tbl.rows, perf_at)
local worst_perf = worst[perf_at]

print(string.format("Best configuration (Performance=%s): %s",
                    ezr.round(best_perf), row_to_str(best)))
print(string.format("Worst configuration (Performance=%s): %s",
                    ezr.round(worst_perf), row_to_str(worst)))

local diffs = diff_rows(best, worst)
print("\nSettings that differ between best and worst:")
for name, pair in pairs(diffs) do
  print(string.format("- %s: best=%s, worst=%s",
                name, tostring(pair.v1), tostring(pair.v2)))
end

print("\nInterpretation:")
print(string.format("The best configuration outperforms the worst by %.1f%%.",
  (worst_perf - best_perf) / best_perf * 100))
print("Key settings that differ are likely contributors to this performance gap.")
