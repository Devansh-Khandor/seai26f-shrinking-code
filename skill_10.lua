#!/usr/bin/env lua
-- Skill 10: Benchmark comparison of selection strategies and prediction accuracy
--
--  APACHE-19 – Compare our configuration-selection system against
--              random selection under the same benchmarking budget.
--  APACHE-20 – Compare our performance predictions against a
--              simple historical‑average estimate.
--
-- Uses the functions and data structures from the earlier skills.
-- Requires the same Lua libraries as before: ezr.

local ezr = require "ezr"

-- ------------------------------------------------------------------
-- Pretty printer (uses ezr.show)
-- ------------------------------------------------------------------
local function pretty(v) return ezr.show(v) end

-- ------------------------------------------------------------------
-- Helper: compute mean and RMSE for a list of predictions vs actuals
-- ------------------------------------------------------------------
local function compute_errors(tbl, preds)
  local perf_at = tbl.cols.y[1].at
  local n = #tbl.rows
  local sum_err = 0
  local sum_sq = 0
  for i, r in ipairs(tbl.rows) do
    local actual = r[perf_at]
    local pred = preds[i]
    local e = actual - pred
    sum_err = sum_err + e
    sum_sq = sum_sq + e * e
  end
  local mae = sum_err / n
  local rmse = math.sqrt(sum_sq / n)
  return mae, rmse
end

-- ------------------------------------------------------------------
-- APACHE-19: Random vs. model‑guided selection
-- ------------------------------------------------------------------
local function random_selection(tbl, budget)
  local sample = ezr.some(tbl.rows, budget)
  local best_perf = math.huge
  local best_row = nil
  local perf_at = tbl.cols.y[1].at
  for _, r in ipairs(sample) do
    local p = r[perf_at]
    if p < best_perf then
      best_perf, best_row = p, r
    end
  end
  return best_perf, best_row, sample
end

local function model_guided_selection(tbl, budget)
  -- Train a tree on all data (cheating, but shows model value)
  local tree = ezr.Tree(tbl, tbl.rows)
  -- Rank all rows by predicted performance
  local scored = {}
  local perf_at = tbl.cols.y[1].at
  for _, r in ipairs(tbl.rows) do
    local pred = tree:leaf(tbl, r)
    scored[#scored+1] = {row = r, pred = pred}
  end
  scored = ezr.sorted(scored, function(a, b) return a.pred < b.pred end)
  local chosen = {}
  for i = 1, math.min(budget, #scored) do
    table.insert(chosen, scored[i].row)
  end
  -- Find the best actual performance among the chosen
  local best_perf = math.huge
  local best_row = nil
  for _, r in ipairs(chosen) do
    local p = r[perf_at]
    if p < best_perf then
      best_perf, best_row = p, r
    end
  end
  return best_perf, best_row, chosen
end

-- ------------------------------------------------------------------
-- APACHE-20: Prediction accuracy vs. historical mean
-- ------------------------------------------------------------------
local function prediction_vs_mean(tbl)
  local perf_at = tbl.cols.y[1].at
  local overall_stats = ezr.performance_stats(tbl)  -- reuse existing function
  local mean_perf = overall_stats.mean

  -- Build tree predictions
  local tree = ezr.Tree(tbl, tbl.rows)
  local tree_preds = {}
  for i, r in ipairs(tbl.rows) do
    tree_preds[i] = tree:leaf(tbl, r)
  end

  -- Compute errors
  local _, rmse_tree = compute_errors(tbl, tree_preds)

  -- Naive mean predictions
  local mean_preds = {}
  for i = 1, #tbl.rows do mean_preds[i] = mean_perf end
  local _, rmse_mean = compute_errors(tbl, mean_preds)

  return {
    mean_perf = mean_perf,
    rmse_tree = rmse_tree,
    rmse_mean = rmse_mean
  }
end

-- ------------------------------------------------------------------
-- Main routine
-- ------------------------------------------------------------------
local function main()
  local csv_path = arg[1]
  assert(csv_path, "Usage: lua skill_10.lua <csv path>")

  -- Load the data
  local tbl = ezr.Tbl(ezr.csv(csv_path))

  -- ---------------------------------------------------------------
  -- APACHE-19 : Selection comparison
  -- ---------------------------------------------------------------
  print("== APACHE-19 ==")
  local budget = ezr.the.budget
  if budget > #tbl.rows then budget = #tbl.rows end

  local random_best, random_row, _ = random_selection(tbl, budget)
  local model_best, model_row, _ = model_guided_selection(tbl, budget)

  print(string.format("Budget: %d rows", budget))
  print(string.format("Random selection best performance: %s",
        pretty(random_best)))
  print(string.format("Model‑guided selection best performance: %s",
        pretty(model_best)))
  if model_best < random_best then
    print("Model‑guided selection discovered a better configuration.")
  elseif model_best > random_best then
    print("Random selection happened to find a better configuration.")
  else
    print("Both strategies found configurations of equal best performance.")
  end

  -- ---------------------------------------------------------------
  -- APACHE-20 : Prediction accuracy comparison
  -- ---------------------------------------------------------------
  print("\n== APACHE-20 ==")
  local results = prediction_vs_mean(tbl)

  print(string.format("Historical mean performance: %s",
        pretty(results.mean_perf)))
  print(string.format("Tree prediction RMSE: %s", pretty(results.rmse_tree)))
  print(string.format("Mean‑performance prediction RMSE: %s",
        pretty(results.rmse_mean)))

  if results.rmse_tree < results.rmse_mean then
    print("Tree predictions provide a meaningful improvement over the historical‑average baseline.")
  elseif results.rmse_tree > results.rmse_mean then
    print("Historical‑average predictions are surprisingly accurate.")
  else
    print("Both methods achieve identical prediction accuracy.")
  end
end

-- Execute if run directly
if not debug.getinfo(1, "S").short_src:match("[/\\]skill_10.lua$") then
  main()
end
