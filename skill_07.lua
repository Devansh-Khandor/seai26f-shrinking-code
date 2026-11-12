#!/usr/bin/env lua
-- Skill 07: Active configuration discovery and evaluation planning
-- Provides two tasks:
--   APACHE-13 – Find a low‑performance configuration with few evaluations
--   APACHE-14 – Recommend an order for evaluating configurations to find good ones early
--
-- Reuses the ezr library and helper utilities from the previous skills.

local ezr = require "ezr"

-- ------------------------------------------------------------------
-- Utility: pretty‑print any value
-- ------------------------------------------------------------------
local function pretty(v) return ezr.show(v) end

-- ------------------------------------------------------------------
-- Sample a small subset of rows and return the best configuration
-- ------------------------------------------------------------------
local function sample_best(tbl, sample_size)
  -- Ensure we do not sample more than exist
  sample_size = math.min(sample_size, #tbl.rows)
  -- Randomly pick a subset
  local sampled = ezr.some(tbl.rows, sample_size)
  local best_row, best_perf
  local perf_col = tbl.cols.y[1].at
  for _, r in ipairs(sampled) do
    local perf = r[perf_col]
    if not best_row or perf < best_perf then
      best_row, best_perf = r, perf
    end
  end
  return best_row, sample_size
end

-- ------------------------------------------------------------------
-- Build a decision tree over all rows and order rows by predicted performance
-- ------------------------------------------------------------------
local function predict_order(tbl, top_n)
  local tree = ezr.Tree(tbl, tbl.rows)
  local scored = {}
  local perf_col = tbl.cols.y[1].at
  for _, r in ipairs(tbl.rows) do
    local pred = tree:leaf(tbl, r)
    scored[#scored + 1] = {row = r, pred = pred}
  end
  scored = ezr.sorted(scored, function(a, b) return a.pred < b.pred end)
  -- Return the top N entries (predicted, actual)
  local top = {}
  for i = 1, math.min(top_n, #scored) do
    top[i] = scored[i]
  end
  return top
end

-- ------------------------------------------------------------------
-- Main routine
-- ------------------------------------------------------------------
local function main()
  local csv_path = arg[1]
  assert(csv_path, "Usage: lua skill_07.lua <csv path>")

  -- Load data
  local tbl = ezr.Tbl(ezr.csv(csv_path))

  -- -----------------------------
  -- APACHE-13 : low‑performance discovery
  -- -----------------------------
  print("== APACHE-13 ==")
  local SAMPLE_SIZE = 10   -- can be tuned
  local best, evals = sample_best(tbl, SAMPLE_SIZE)
  local perf_col = tbl.cols.y[1].at
  print(string.format("Evaluated %d random rows.", evals))
  print("Best configuration performance:", pretty(best[perf_col]))
  print("Configuration:")
  for _, c in ipairs(tbl.cols.all) do
    -- skip performance column in display
    if c ~= tbl.cols.y[1] then
      print(string.format("  %s: %s", c.name, pretty(best[c.at])))
    end
  end

  -- -----------------------------
  -- APACHE-14 : evaluation order recommendation
  -- -----------------------------
  print("\n== APACHE-14 ==")
  local TOP_N = 10
  local order = predict_order(tbl, TOP_N)
  print(string.format("Recommended evaluation order (top %d rows):", #order))
  for i, entry in ipairs(order) do
    local r = entry.row
    local actual = r[perf_col]
    print(string.format("%d. Predicted: %s (Actual: %s)",
      i, pretty(entry.pred), pretty(actual)))
    print("   Configuration:")
    for _, c in ipairs(tbl.cols.all) do
      if c ~= tbl.cols.y[1] then
        print(string.format("    %s: %s", c.name, pretty(r[c.at])))
      end
    end
    print()
  end
end

-- Execute when run directly
if not debug.getinfo(1, "S").short_src:match("[/\\]skill_07.lua$") then
  main()
end
