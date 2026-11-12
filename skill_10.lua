#!/usr/bin/env lua
-- skill_10.lua
-- APACHE-19 & APACHE-20: Benchmarking comparison and prediction quality
--
-- Usage: lua skill_10.lua <path-to-Apache_AllMeasurements.csv> [budget]
--   budget – optional, number of rows to benchmark in APACHE‑19 (default 10)
--
-- Requires only the ezr library (ezr.lua & ezr-lib.lua) in the same directory.

local ezr = require("ezr")
local csv  = ezr.csv
local TBL  = ezr.Tbl
local Tree = ezr.Tree
local sorted = ezr.sorted
local keysort = ezr.keysort
local rand   = ezr.rand
local copy   = ezr.copy
local show   = ezr.show

-- ------------------------------------------------------------------
-- Load CSV into a TBL
-- ------------------------------------------------------------------
local function load_tbl(path)
  return TBL(csv(path))
end

-- ------------------------------------------------------------------
-- Acquire order – active‑learning style selection
-- ------------------------------------------------------------------
local function acquire_order(tbl, budget)
  local rows = tbl.rows          -- all candidate rows
  local lab  = {}                -- evaluated rows
  -- pick a random starting point
  local shuffled = keysort(rows, function(_) return rand() end)
  table.insert(lab, shuffled[1])

  -- model: leaf value from a tree built on labelled rows
  local function model(rowX, k)   -- k is number of goals (1)
    local t = Tree(tbl, lab)
    local val = t:leaf(tbl, rowX)
    return {val}
  end
  tbl.model = model

  local acquired = tbl:acquire(rows, budget, lab)

  -- Return the acquisition order (indices in original tbl.rows)
  local order = {}
  for _, r in ipairs(acquired) do
    for i, rr in ipairs(rows) do
      if r == rr then
        table.insert(order, {idx=i, row=r})
        break
      end
    end
  end
  return order
end

-- ------------------------------------------------------------------
-- Evaluate best (lowest) performance among a list of rows
-- ------------------------------------------------------------------
local function best_perf(order, perf_col)
  local best = nil
  for _, item in ipairs(order) do
    local v = item.row[perf_col]
    if v ~= "?" and (best==nil or v < best) then
      best = v
    end
  end
  return best
end

-- ------------------------------------------------------------------
-- Random selection – sample without replacement
-- ------------------------------------------------------------------
local function random_selection(rows, budget)
  local shuffled = keysort(rows, function(_) return rand() end)
  local selected = {}
  for i=1,budget do
    table.insert(selected, {idx=i, row=shuffled[i]})
  end
  return selected
end

-- ------------------------------------------------------------------
-- APACHE‑19: Compare acquisition vs random
-- ------------------------------------------------------------------
local function apache19(tbl, budget)
  print("== APACHE-19 ==")

  local perf_col = tbl.cols.y[1].at
  local trials = 20
  local our_best_sum = 0
  local rand_best_sum = 0
  local our_best_min = nil
  local rand_best_min = nil

  for t=1,trials do
    -- our method
    local order = acquire_order(tbl, budget)
    local best_our = best_perf(order, perf_col)
    our_best_sum = our_best_sum + (best_our or math.huge)

    if best_our then
      if our_best_min==nil or best_our < our_best_min then
        our_best_min = best_our
      end
    end

    -- random baseline
    local random_order = random_selection(tbl.rows, budget)
    local best_rand = best_perf(random_order, perf_col)
    rand_best_sum = rand_best_sum + (best_rand or math.huge)

    if best_rand then
      if rand_best_min==nil or best_rand < rand_best_min then
        rand_best_min = best_rand
      end
    end
  end

  local avg_our = our_best_sum / trials
  local avg_rand = rand_best_sum / trials

  print(string.format("  Acquisition: avg lowest perf = %d, best seen = %d",
                      avg_our, our_best_min))
  print(string.format("  Random:      avg lowest perf = %d, best seen = %d",
                      avg_rand, rand_best_min))
  if avg_our < avg_rand then
    print("  ==> Acquisition discovers better configurations on average.")
  else
    print("  ==> Random selection is as good or better (unlikely).")
  end
  print()
end

-- ------------------------------------------------------------------
-- APACHE‑20: Prediction accuracy vs historical average
-- ------------------------------------------------------------------
local function apache20(tbl)
  print("== APACHE-20 ==")

  local perf_col = tbl.cols.y[1].at
  -- simple train/test split (50/50)
  local shuffled = keysort(tbl.rows, function(_) return rand() end)
  local n = #shuffled
  local split = math.floor(n/2)
  local train_rows = {}
  local test_rows  = {}
  for i,r in ipairs(shuffled) do
    if i <= split then table.insert(train_rows, r)
    else table.insert(test_rows, r) end
  end

  -- Build model on training set
  local tmodel = Tree(tbl, train_rows)

  -- Historical average from training set
  local sum = 0
  for _, r in ipairs(train_rows) do
    sum = sum + r[perf_col]
  end
  local mean_train = sum / #train_rows

  -- Predict and compute MAE for both approaches
  local err_our = 0
  local err_hist = 0
  local cnt = 0
  for _, r in ipairs(test_rows) do
    local actual = r[perf_col]
    local pred = tmodel:leaf(tbl, r)
    err_our = err_our + math.abs(pred - actual)
    err_hist = err_hist + math.abs(mean_train - actual)
    cnt = cnt + 1
  end
  local mae_our = err_our / cnt
  local mae_hist = err_hist / cnt

  print(string.format("  MAE (tree): %.2f", mae_our))
  print(string.format("  MAE (historical avg): %.2f", mae_hist))
  if mae_our < mae_hist then
    print("  ==> Tree predictions improve on historical average.")
  else
    print("  ==> Historical average is as good or better (unlikely).")
  end
  print()
end

-- ------------------------------------------------------------------
-- Main entry point
-- ------------------------------------------------------------------
local function main()
  local path = arg[1]
  if not path then
    io.stderr:write("Error: CSV file path required\n")
    os.exit(1)
  end

  local budget = tonumber(arg[2]) or 10

  local tbl = load_tbl(path)

  apache19(tbl, budget)
  apache20(tbl)
end

main()
