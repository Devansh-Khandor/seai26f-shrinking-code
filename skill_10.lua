#!/usr/bin/env lua
-- Skill 10 – Apache configuration evaluation comparison
--
-- Usage:  lua skill_10.lua <csv path>
--
-- Requires the ezr library (ezr.lua, ezr-lib.lua) in the same directory.
--
-- APACHE-19: Compare our configuration‑selection system against
--            random selection under the same benchmarking budget.
--            Show which approach discovers a better best‑performance
--            configuration.
--
-- APACHE-20: Compare our tree‑based predictions against a simple
--            historical‑average estimate and show whether our system
--            provides a meaningful improvement on unseen configurations.
--
local ezr = require("ezr")

-- ---------- utilities -------------------------------------------------
local function format_num(v, digits)
  digits = digits or 2
  local f = ("%." .. digits .. "f")
  return string.format(f, v)
end

-- ---------- APACHE-19 -------------------------------------------------
local function apache_19(tbl)
  print("== APACHE-19 ==")

  local perf_at = tbl.cols.y[1].at
  local budget = math.min(ezr.the.budget or 10, #tbl.rows)

  -- --- Acquire‑based selection ---------------------------------------
  local acquired = tbl:acquire(tbl.rows, budget, {}, nil, nil)
  local sorted_acq = ezr.keysort(acquired, tbl:Y())
  local best_acq = sorted_acq[1]
  local best_acq_perf = best_acq[perf_at]

  -- --- Random selection ---------------------------------------------
  local random_rows = ezr.some(tbl.rows, budget)
  local best_rand, best_rand_perf = nil, nil
  for _, r in ipairs(random_rows) do
    local p = r[perf_at]
    if not best_rand or p < best_rand_perf then
      best_rand, best_rand_perf = r, p
    end
  end

  -- --- Results ------------------------------------------------------
  print(string.format("Budget: %d rows", budget))
  print(string.format("Best via acquirer: %.1f (row %d)",
                      best_acq_perf,
                      ezr.keys(tbl.rows)[ezr.table.find(tbl.rows, best_acq)]))
  print(string.format("Best via random:    %.1f (row %d)",
                      best_rand_perf,
                      ezr.keys(tbl.rows)[ezr.table.find(tbl.rows, best_rand)]))
  local improvement = (best_rand_perf - best_acq_perf) / best_rand_perf * 100
  print(string.format("Improvement over random: %.1f%%", improvement))
  print()
end

-- ---------- APACHE-20 -------------------------------------------------
local function apache_20(tbl)
  print("== APACHE-20 ==")

  local perf_at = tbl.cols.y[1].at

  -- --- Split data ----------------------------------------------------
  local shuffled = ezr.shuffle(tbl.rows)
  local n_half = math.floor(#shuffled / 2)
  local train_rows = ezr.slice(shuffled, 1, n_half)
  local test_rows  = ezr.slice(shuffled, n_half + 1)

  -- --- Build tree on training ---------------------------------------
  local train_tbl = tbl:clone(train_rows)
  local tree = ezr.Tree(train_tbl, train_tbl.rows)

  local historic_mean = train_tbl.cols.y[1]:mid()

  -- --- Evaluate on test set -----------------------------------------
  local err_tree = 0
  local err_avg  = 0
  local n_test = #test_rows

  for _, r in ipairs(test_rows) do
    local actual = r[perf_at]
    local pred_tree = tree:leaf(train_tbl, r)
    err_tree = err_tree + (actual - pred_tree)^2
    err_avg  = err_avg  + (actual - historic_mean)^2
  end

  local rmse_tree = math.sqrt(err_tree / n_test)
  local rmse_avg  = math.sqrt(err_avg / n_test)
  local improv = (rmse_avg - rmse_tree) / rmse_avg * 100

  -- --- Results ------------------------------------------------------
  print(string.format("Test set size: %d rows", n_test))
  print(string.format("RMSE of tree predictor : %.2f", rmse_tree))
  print(string.format("RMSE of historical mean: %.2f", rmse_avg))
  print(string.format("Improvement over average: %.1f%%", improv))
  print()
end

-- ---------- main ------------------------------------------------------
local function main(csv_path)
  if not csv_path then
    print("Usage: lua skill_10.lua <csv path>")
    os.exit(1)
  end

  local tbl = ezr.Tbl(ezr.csv(csv_path))

  apache_19(tbl)
  apache_20(tbl)
end

main(arg[1])
