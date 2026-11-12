#!/usr/bin/env lua
-- Skill 06 – Apache configuration recommendation & explanation
--
-- Usage:  lua skill_06.lua <csv path>
--
-- Requires the ezr library (ezr.lua, ezr-lib.lua) in the same directory.
--
-- APACHE-11: Explain why a recommended configuration is best.
-- APACHE-12: Explain the differences between two configurations
--            that receive different performance assessments.

local ezr = require("ezr")

-- ------------------------------------------------------------------
-- Utility: convert a row to a CSV string, excluding the goal column
local function row_to_csv(row, goal_at)
  local parts = {}
  for i, v in ipairs(row) do
    if i ~= goal_at then
      table.insert(parts, tostring(v))
    end
  end
  return table.concat(parts, ",")
end

-- ------------------------------------------------------------------
-- Find the index of a row in the original table
local function make_index_map(rows)
  local idx = {}
  for i, r in ipairs(rows) do
    idx[r] = i
  end
  return idx
end

-- ------------------------------------------------------------------
-- Compute the differences between two rows (excluding goal)
local function diff_settings(row1, row2, tbl, goal_at)
  local diffs = {}
  for _, c in ipairs(tbl.cols.all) do
    if c.at ~= goal_at then
      local v1 = row1[c.at]
      local v2 = row2[c.at]
      if v1 ~= v2 then
        diffs[c.name] = {v1 = v1, v2 = v2}
      end
    end
  end
  return diffs
end

-- ------------------------------------------------------------------
-- APACHE-11: Recommend the best configuration and compare it
--            with the nearest previously tested configurations.
local function recommend_and_explain(tbl, idx_map)
  print("== APACHE-11 ==")

  -- Find the row with the lowest (best) performance
  local best_row = table.min(tbl.rows, function(r) return r[tbl.cols.y[1].at] end)
  local best_idx = idx_map[best_row]
  local best_perf = best_row[tbl.cols.y[1].at]

  print(string.format("Recommended configuration (row %d, best perf = %.0f):",
                     best_idx, best_perf))
  print(string.format("  Settings: %s", row_to_csv(best_row, tbl.cols.y[1].at)))

  -- Find the three closest configurations (by feature distance)
  local distances = {}
  for _, r in ipairs(tbl.rows) do
    if r ~= best_row then
      table.insert(distances, {row = r, dist = tbl:distx(best_row, r)})
    end
  end
  table.sort(distances, function(a, b) return a.dist < b.dist end)

  local neighbors = {}
  for i = 1, math.min(3, #distances) do
    table.insert(neighbors, distances[i].row)
  end

  if #neighbors > 0 then
    print("\nClosest previously tested configurations:")
    for i, r in ipairs(neighbors) do
      local idx = idx_map[r]
      local perf = r[tbl.cols.y[1].at]
      local diffs = diff_settings(best_row, r, tbl, tbl.cols.y[1].at)
      local diff_str = {}
      for name, vals in pairs(diffs) do
        table.insert(diff_str, string.format("%s=%s→%s",
          name, tostring(vals.v1), tostring(vals.v2)))
      end
      diff_str = table.concat(diff_str, ", ")
      print(string.format("%d. Row %d (perf=%.0f):\n   Diff: %s",
                         i, idx, perf, diff_str))
    end
  else
    print("\nNo other configurations to compare.")
  end
end

-- ------------------------------------------------------------------
-- APACHE-12: Explain the differences between two configurations
--            with different performance assessments.
local function compare_two_configs(tbl, idx_map)
  print("\n== APACHE-12 ==")

  -- Identify best and worst rows by actual performance
  local best_row = table.min(tbl.rows, function(r) return r[tbl.cols.y[1].at] end)
  local worst_row = table.max(tbl.rows, function(r) return r[tbl.cols.y[1].at] end)
  local best_idx = idx_map[best_row]
  local worst_idx = idx_map[worst_row]
  local best_perf = best_row[tbl.cols.y[1].at]
  local worst_perf = worst_row[tbl.cols.y[1].at]
  local perf_diff = worst_perf - best_perf

  print(string.format("Comparing best (row %d, perf=%.0f) vs worst (row %d, perf=%.0f):",
                     best_idx, best_perf, worst_idx, worst_perf))
  print(string.format("Performance difference: %.0f", perf_diff))

  local diffs = diff_settings(best_row, worst_row, tbl, tbl.cols.y[1].at)
  if next(diffs) then
    print("Differing settings (best → worst):")
    for name, vals in pairs(diffs) do
      print(string.format("  %s: %s → %s",
                          name, tostring(vals.v1), tostring(vals.v2)))
    end
  else
    print("No differing settings (rows identical).")
  end
end

-- ------------------------------------------------------------------
-- Main ----------------------------------------------------------------
local function main(csv_path)
  if not csv_path then
    print("Usage: lua skill_06.lua <csv path>")
    os.exit(1)
  end

  -- Load data
  local tbl = ezr.Tbl(ezr.csv(csv_path))
  local idx_map = make_index_map(tbl.rows)

  -- Build a tree for predictions (used only for consistency checks)
  local tree = ezr.Tree(tbl, tbl.rows)

  -- APACHE-11
  recommend_and_explain(tbl, idx_map)

  -- APACHE-12
  compare_two_configs(tbl, idx_map)
end

main(arg[1])
