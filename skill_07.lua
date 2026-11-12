#!/usr/bin/env lua
-- Skill 07 – Apache configuration discovery & early benchmarking
--
-- Usage:  lua skill_07.lua <csv path>
--
-- Requires the ezr library (ezr.lua, ezr-lib.lua) in the same directory.
--
-- APACHE-13: Discover a low‑performance configuration while evaluating as
--            few candidate configurations as possible.
--
-- APACHE-14: When a limited benchmarking budget is available, recommend
--            which configurations should be evaluated first to discover
--            promising server settings early.
--
local ezr = require("ezr")

-- Helper: convert a row to a CSV string, excluding the goal column
local function row_to_csv(row, goal_at)
  local parts = {}
  for i, v in ipairs(row) do
    if i ~= goal_at then
      table.insert(parts, tostring(v))
    end
  end
  return table.concat(parts, ",")
end

-- Find the minimum element of a list using a key function
local function list_min(list, keyf)
  local best, bestv = nil, nil
  for _, v in ipairs(list) do
    local k = keyf(v)
    if best == nil or k < bestv then
      best, bestv = v, k
    end
  end
  return best
end

-- Main --------------------------------------------------------------------
local function main(csv_path)
  if not csv_path then
    print("Usage: lua skill_07.lua <csv path>")
    os.exit(1)
  end

  -- Load data
  local tbl = ezr.Tbl(ezr.csv(csv_path))

  -- Goal column (performance)
  local perf_col = tbl.cols.y[1]
  local perf_at = perf_col.at

  ---------------------------------------------------------------------------
  -- APACHE-13: discover a low‑performance configuration with few labels
  ---------------------------------------------------------------------------
  print("== APACHE-13 ==")

  -- Budget: try to acquire only a small set of rows.
  -- Use the library's default budget if set, otherwise 10.
  local acquire_budget = (the and the.budget) or 10

  -- Acquire rows (simulate evaluating only these)
  local labelled = tbl:acquire(tbl.rows, acquire_budget, {}, nil, nil)

  -- Among labelled rows, find the best (lowest performance)
  local best_row = list_min(labelled, function(r) return r[perf_at] end)
  local best_perf = best_row[perf_at]
  local best_idx  = ezr.keys(tbl.rows)[#tbl.rows]  -- fallback index

  -- Find original index of best_row
  for idx, r in ipairs(tbl.rows) do
    if r == best_row then
      best_idx = idx
      break
    end
  end

  print(string.format("Evaluated %d configurations.", #labelled))
  print(string.format("Best configuration found (row %d):", best_idx))
  print(string.format("  Performance = %.1f", best_perf))
  print(string.format("  Settings    = %s", row_to_csv(best_row, perf_at)))
  print()

  ---------------------------------------------------------------------------
  -- APACHE-14: recommend early benchmarking order
  ---------------------------------------------------------------------------
  print("== APACHE-14 ==")

  -- Build a predictive tree over all rows
  local tree = ezr.Tree(tbl, tbl.rows)

  -- Rank rows by predicted performance (lower is better)
  local sorted_pred = ezr.keysort(tbl.rows, function(r) return tree:leaf(tbl, r) end)

  local recommend_n = math.min(10, #sorted_pred)
  print(string.format("Top %d configurations to evaluate first (predicted low perf):", recommend_n))
  for i = 1, recommend_n do
    local r = sorted_pred[i]
    local idx = ezr.keys(tbl.rows)[i]  -- position in original order
    -- Find actual index
    for j, rr in ipairs(tbl.rows) do
      if rr == r then idx = j; break end
    end
    local pred = tree:leaf(tbl, r)
    local actual = r[perf_at]
    print(string.format("%2d. Row %d: Pred=%.1f, Actual=%.1f -> %s",
                        i, idx, pred, actual, row_to_csv(r, perf_at)))
  end
end

main(arg[1])
