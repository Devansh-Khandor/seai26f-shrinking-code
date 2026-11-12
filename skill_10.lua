#!/usr/bin/env lua
-- Skill 10: CAR-19 and CAR-20 – Selection vs Random & Baseline Performance
-- Usage: lua skill_10.lua <csv path>

local ezr = require("ezr")
local csv = ezr.csv
local shuffle = ezr.shuffle
local sorted = ezr.sorted
local round = ezr.round

local function main(path)
  if not path then
    io.stderr:write("Usage: lua skill_10.lua <csv path>\n")
    os.exit(1)
  end

  -- Re-seed for reproducibility
  ezr.srand(1234567891)

  -- Load the full dataset
  local tbl = ezr.Tbl(csv(path))
  local all_rows = tbl.rows
  local n_rows = #all_rows
  if n_rows == 0 then
    io.stderr:write("Empty CSV\n")
    os.exit(1)
  end

  -- Identify goal column(s); use the first goal (e.g. "Mpg+")
  local goal_col = tbl.cols.y[1]
  if not goal_col then
    io.stderr:write("No goal column found\n")
    os.exit(1)
  end

  -- --- CAR-19: Selection vs Random --------------------------------
  print("== CAR-19 ==")

  -- Evaluation budget (number of labelled rows we can use)
  local budget = math.min(20, n_rows)
  -- Randomly pick training rows
  local shuffled = shuffle(all_rows)
  local train_rows = {unpack(shuffled, 1, budget)}
  local test_rows  = {unpack(shuffled, budget + 1)}

  -- Build training table and tree
  local train_tbl = tbl:clone(train_rows)
  local tree = ezr.Tree(train_tbl, train_tbl.rows)

  -- Predict for every row
  local pred_scores = {}
  for _, r in ipairs(all_rows) do
    local pred = tree:leaf(train_tbl, r)
    pred_scores[r] = pred
  end

  -- Top N candidates by predicted score
  local topN = 5
  local ranked = sorted(all_rows, function(a, b)
    return pred_scores[a] > pred_scores[b]
  end)
  local selected = {}
  for i = 1, math.min(topN, #ranked) do
    selected[i] = ranked[i]
  end

  -- Compute average true mpg+ of selected set
  local sum_true_selected = 0
  for _, r in ipairs(selected) do
    sum_true_selected = sum_true_selected + r[goal_col.at]
  end
  local avg_true_selected = sum_true_selected / (#selected)

  -- Randomly pick same number of rows for baseline
  local random_set = {}
  local rand_idx = shuffle(all_rows)
  for i = 1, #selected do
    random_set[i] = rand_idx[i]
  end
  local sum_true_random = 0
  for _, r in ipairs(random_set) do
    sum_true_random = sum_true_random + r[goal_col.at]
  end
  local avg_true_random = sum_true_random / (#random_set)

  print(string.format("Budget (labelled rows): %d", budget))
  print(string.format("Average true MPG+ of best %d candidates: %.3f",
                      #selected, round(avg_true_selected,3)))
  print(string.format("Average true MPG+ of %d random picks: %.3f",
                      #random_set, round(avg_true_random,3)))
  print("Improvement over random: %.1f%%",
        round(100 * (avg_true_selected - avg_true_random) / avg_true_random,1))

  -- --- CAR-20: Model vs Historical‑Average -------------------------
  print("\n== CAR-20 ==")

  -- Compute mean MPG+ of training set (baseline)
  local sum_train = 0
  for _, r in ipairs(train_rows) do
    sum_train = sum_train + r[goal_col.at]
  end
  local mean_train = sum_train / #train_rows

  -- MSE for tree predictions on test set
  local mse_tree, mse_mean = 0, 0
  for _, r in ipairs(test_rows) do
    local true_val = r[goal_col.at]
    local pred_val = tree:leaf(train_tbl, r)
    local diff_tree = true_val - pred_val
    local diff_mean = true_val - mean_train
    mse_tree = mse_tree + diff_tree * diff_tree
    mse_mean = mse_mean + diff_mean * diff_mean
  end
  mse_tree = mse_tree / #test_rows
  mse_mean = mse_mean / #test_rows

  print(string.format("Test set size: %d", #test_rows))
  print(string.format("MSE of tree predictions: %.4f", round(mse_tree,4)))
  print(string.format("MSE of historical‑average baseline: %.4f",
                      round(mse_mean,4)))
  print(string.format("Relative improvement: %.1f%%",
                      round(100 * (mse_mean - mse_tree) / mse_mean,1)))
end

main(arg[1])
