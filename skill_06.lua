#!/usr/bin/env lua
-- skill_06.lua
-- Implements the requested APACHE-11 and APACHE-12 analyses.
--
-- Usage: lua skill_06.lua <csv path>
--
-- Requires the ezr library from the provided codebase.

local ezr = require"ezr"

-- Utility: round numbers for display, matching ezr's format.
local function display(v)
  if type(v) == "number" then
    return ezr.round(v)
  else
    return tostring(v)
  end
end

-- Utility: convert a row table to CSV string.
local function row_to_csv(row)
  local parts = {}
  for _,v in ipairs(row) do
    if v == "?" then
      table.insert(parts, "?")
    else
      table.insert(parts, tostring(v))
    end
  end
  return table.concat(parts, ",")
end

-- Load the data file.
local csvfile = arg[1]
assert(csvfile, "Usage: lua skill_06.lua <csv path>")

-- Build the table from the CSV file.
local tbl = ezr.Tbl(ezr.csv(csvfile))
local perf_col = tbl.cols.y[1]   -- the single goal column

-- Gather all performance values.
local perf_vals = ezr.map(tbl.rows, function(r) return r[perf_col.at] end)

-- Compute overall mean performance.
local mean_perf = perf_col:mid()

-- --------- APACHE-11: Recommended configuration ----------
print("== APACHE-11 ==")

-- Find the best (lowest performance) configuration.
local best_rows = ezr.keysort(tbl.rows, function(r) return r[perf_col.at] end)
local best_row = best_rows[1]
local best_idx = nil
for i,r in ipairs(tbl.rows) do
  if r == best_row then best_idx = i end
  break end
-- Print recommendation.
print(string.format("Recommended configuration #%d:", best_idx))
print(string.format("  Performance = %s", display(best_row[perf_col.at])))
print(string.format("  Config     = %s", row_to_csv(best_row)))

-- Explain why it is recommended.
local diff_from_mean = mean_perf - best_row[perf_col.at]
print(string.format("  Performance is %.3f below the overall mean (%.3f).",
  diff_from_mean, mean_perf))

-- Find the nearest neighbour (in feature space) to the best configuration.
local best_dist = math.huge
local nearest = nil
local nearest_idx = nil
for i,r in ipairs(tbl.rows) do
  if r ~= best_row then
    local d = tbl:distx(best_row, r)
    if d < best_dist then
      best_dist = d
      nearest = r
      nearest_idx = i
    end
  end
end
if nearest then
  local d_perf = nearest[perf_col.at] - best_row[perf_col.at]
  print(string.format(
    "  Nearest neighbour (#%d) is at distance %.3f with performance %.3f, ",
    nearest_idx, best_dist, nearest[perf_col.at]))
  print(string.format("   difference = %.3f.", d_perf))
end
print()

-- --------- APACHE-12: Explain difference between two configurations ----------
print("== APACHE-12 ==")

-- Find the pair with the largest absolute performance difference.
local max_diff = -1
local pair_i, pair_j = nil, nil
for i = 1, #tbl.rows-1 do
  for j = i+1, #tbl.rows do
    local diff = math.abs(tbl.rows[i][perf_col.at] - tbl.rows[j][perf_col.at])
    if diff > max_diff then
      max_diff = diff
      pair_i, pair_j = i, j
    end
  end
end

-- Output the two configurations.
local row_i = tbl.rows[pair_i]
local row_j = tbl.rows[pair_j]
print(string.format("Largest performance difference: %.3f between rows #%d and #%d.",
  max_diff, pair_i, pair_j))
print(string.format("  Row #%d performance = %s", pair_i, display(row_i[perf_col.at])))
print(string.format("  Row #%d performance = %s", pair_j, display(row_j[perf_col.at])))
print()

-- Identify which settings differ.
local diffs = {}
for _,c in ipairs(tbl.cols.all) do
  if c ~= perf_col then
    local v1, v2 = row_i[c.at], row_j[c.at]
    if v1 ~= v2 then
      table.insert(diffs, {name=c.name, v1=v1, v2=v2})
    end
  end
end

if #diffs == 0 then
  print("  No differing settings found (identical rows except performance).")
else
  print("  Differing settings:")
  for _,d in ipairs(diffs) do
    print(string.format(
      "    %-15s: row %d -> %s, row %d -> %s",
      d.name, pair_i, display(d.v1), pair_j, display(d.v2)))
  end
  print()
  print(string.format("  The performance difference (%.3f) may be partly attributable to these settings.",
    max_diff))
end
