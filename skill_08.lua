#!/usr/bin/env lua
-- Skill 08 – Apache configuration improvement guidance
--
-- Usage:  lua skill_08.lua <csv path>
--
-- Requires the ezr library (ezr.lua, ezr-lib.lua) in the same directory.
--
-- APACHE-15: Recommend a small change to a poorly performing configuration
--            that could lead to a lower predicted performance value,
--            using historically observed valid configurations as alternatives.
--
-- APACHE-16: Find the smallest number of configuration-setting changes that
--            could bring a poorly performing configuration closer to a
--            specified performance target, while avoiding unsupported or
--            invalid combinations.

local ezr = require("ezr")

-- --------------------------------------------------------------
-- Utility helpers
-- --------------------------------------------------------------
local function row_to_csv(row, goal_at)
  local parts = {}
  for i, v in ipairs(row) do
    if i ~= goal_at then
      table.insert(parts, tostring(v))
    end
  end
  return table.concat(parts, ",")
end

-- Hamming distance (excluding goal column)
local function hamming(row1, row2, goal_at)
  local d = 0
  for _, c in ipairs(row1._cols) do
    if c.at ~= goal_at and row1[c.at] ~= row2[c.at] then
      d = d + 1
    end
  end
  return d
end

-- Diff list: setting name => {old, new}
local function diff_settings(row1, row2, goal_at)
  local diffs = {}
  for _, c in ipairs(row1._cols) do
    if c.at ~= goal_at and row1[c.at] ~= row2[c.at] then
      diffs[c.name] = {old = row1[c.at], new = row2[c.at]}
    end
  end
  return diffs
end

-- Find index of a row in a list
local function index_of(rows, target)
  for i, r in ipairs(rows) do
    if r == target then return i end
  end
  return nil
end

-- --------------------------------------------------------------
-- Main entry point
-- --------------------------------------------------------------
local function main(csv_path)
  if not csv_path then
    print("Usage: lua skill_08.lua <csv path>")
    os.exit(1)
  end

  -- Load data
  local tbl = ezr.Tbl(ezr.csv(csv_path))
  -- attach cols list to rows for easier diffing
  for _, r in ipairs(tbl.rows) do r._cols = tbl.cols.all end

  local goal_at = tbl.cols.y[1].at
  local perf_col = tbl.cols.y[1]

  -- Find worst and best rows by actual performance
  local worst_idx, best_idx
  local worst_perf, best_perf = -math.huge, math.huge
  for i, r in ipairs(tbl.rows) do
    local v = r[goal_at]
    if v > worst_perf then
      worst_perf, worst_idx = v, i
    end
    if v < best_perf then
      best_perf, best_idx = v, i
    end
  end
  local worst_row = tbl.rows[worst_idx]
  local best_row  = tbl.rows[best_idx]

  -- Build a predictive tree (used for predictions)
  local tree = ezr.Tree(tbl, tbl.rows)

  ------------------------------------------------------------------
  -- APACHE-15: small change recommendation
  ------------------------------------------------------------------
  print("== APACHE-15 ==")
  print(string.format(
    "Worst configuration row %d with performance %.1f",
    worst_idx, worst_perf))

  -- Predicted performance of worst row
  local worst_pred = tree:leaf(tbl, worst_row)

  -- Search candidate rows: start with distance 1, then 2, ...
  local best_candidate, best_candidate_pred, best_distance, best_diff
  for dist = 1, #tbl.rows do
    for _, r in ipairs(tbl.rows) do
      if r ~= worst_row then
        local d = hamming(worst_row, r, goal_at)
        if d == dist then
          local pred = tree:leaf(tbl, r)
          if pred < worst_pred then
            local dists = diff_settings(worst_row, r, goal_at)
            if not best_candidate or
               d < best_distance or (d == best_distance and pred < best_candidate_pred) then
              best_candidate, best_candidate_pred, best_distance = r, pred, d
              best_diff = dists
            end
          end
        end
      end
    end
    if best_candidate then break end
  end

  if best_candidate then
    local cand_idx = index_of(tbl.rows, best_candidate)
    print(string.format("Recommended change: modify %d settings to reach row %d (perf=%.1f, pred=%.1f)",
      best_distance, cand_idx, best_candidate[goal_at], best_candidate_pred))
    print("Changed settings:")
    for name, vals in pairs(best_diff) do
      print(string.format("  %s: %s -> %s", name, tostring(vals.old), tostring(vals.new)))
    end
  else
    print("No single‑change improvement found in the data.")
  end

  print()  -- blank line

  ------------------------------------------------------------------
  -- APACHE-16: minimal changes to reach target performance
  ------------------------------------------------------------------
  print("== APACHE-16 ==")

  -- Prompt user for target performance
  io.write("Enter target performance (press Enter for best): ")
  local target_input = io.read()
  local target_perf
  if target_input == "" then
    target_perf = best_perf
    io.write(string.format("Using best performance %.1f as target.\n", target_perf))
  else
    target_perf = tonumber(target_input)
    if not target_perf then
      print("Invalid number; aborting.")
      os.exit(1)
    end
    io.write(string.format("Target set to %.1f.\n", target_perf))
  end

  -- Find rows meeting target
  local feasible = {}
  for _, r in ipairs(tbl.rows) do
    if r[goal_at] <= target_perf then
      table.insert(feasible, r)
    end
  end

  if #feasible == 0 then
    print("No configurations meet the target performance.")
    os.exit(0)
  end

  -- Find minimal Hamming distance to worst row among feasible rows
  local min_dist, best_target, best_target_idx, best_target_pred, best_target_diff
  for _, r in ipairs(feasible) do
    if r ~= worst_row then
      local d = hamming(worst_row, r, goal_at)
      local pred = tree:leaf(tbl, r)
      if not min_dist or d < min_dist or (d == min_dist and pred < best_target_pred) then
        min_dist, best_target, best_target_idx, best_target_pred = d, r, index_of(tbl.rows, r), pred
        best_target_diff = diff_settings(worst_row, r, goal_at)
      end
    end
  end

  if best_target then
    print(string.format("Minimal changes needed: %d settings to reach row %d (perf=%.1f, pred=%.1f)",
      min_dist, best_target_idx, best_target[goal_at], best_target_pred))
    print("Changed settings:")
    for name, vals in pairs(best_target_diff) do
      print(string.format("  %s: %s -> %s", name, tostring(vals.old), tostring(vals.new)))
    end
  else
    print("No improvement path found.")
  end
end

main(arg[1])
