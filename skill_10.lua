#!/usr/bin/env lua
-- skill_10.lua
-- CAR-19 & CAR-20: Selection efficacy vs random, and
-- predictive accuracy vs historical average.
--
-- Usage: lua skill_10.lua <path-to-auto93.csv>
--
-- Requires only the bundled ezr library.
--

local ezr = require("ezr")
local csv  = ezr.csv
local math = math

-- --------------------------------------------------------------------
-- Utility helpers
-- --------------------------------------------------------------------

-- Find column index by header name
local function find_col(tbl, name)
  for _,c in ipairs(tbl.cols.all) do
    if c.name == name then return c.at end
  end
  return nil
end

-- Simple composite score: higher mpg, lower weight, higher acceleration
local function score(row, mpg_at, lbs_at, acc_at)
  local mpg, lbs, acc = row[mpg_at], row[lbs_at], row[acc_at]
  if mpg == "?" or lbs == "?" or acc == "?" or not mpg or not lbs or not acc then
    return nil
  end
  return (mpg / lbs) + acc
end

-- Linear regression builder (as in skill_02)
local function build_beta(rows, target_index)
  local X = {}
  local y = {}
  for _, row in ipairs(rows) do
    local tgt = row[target_index]
    if tgt ~= "?" and tgt ~= nil then
      local feat = {1}          -- bias term
      local ok = true
      for j = 1, #row do
        if j ~= target_index then
          local v = row[j]
          if v == "?" or v == nil then
            ok = false
            break
          end
          table.insert(feat, v)
        end
      end
      if ok then
        table.insert(X, feat)
        table.insert(y, tgt)
      end
    end
  end
  if #X < 2 then error("Not enough data to build model") end

  -- Compute (XᵀX)⁻¹ Xᵀy
  local p = #X[1]
  local XtX = {}
  for i = 1, p do
    XtX[i] = {}
    for j = 1, p do
      local sum = 0
      for r = 1, #X do
        sum = sum + X[r][i] * X[r][j]
      end
      XtX[i][j] = sum
    end
  end

  local Xty = {}
  for i = 1, p do
    local sum = 0
    for r = 1, #X do
      sum = sum + X[r][i] * y[r]
    end
    Xty[i] = {sum}
  end

  -- Inverse of XtX
  local function mat_inverse(A)
    local n = #A
    local M = {}
    for i = 1, n do
      M[i] = {}
      for j = 1, n do
        M[i][j] = A[i][j]
      end
    end

    -- Augment with identity
    for i = 1, n do
      for j = 1, n do
        M[i][n + j] = (i == j) and 1 or 0
      end
    end

    -- Forward elimination
    for i = 1, n do
      local piv = M[i][i]
      if math.abs(piv) < 1e-12 then
        local swap
        for k = i + 1, n do
          if math.abs(M[k][i]) > 1e-12 then
            swap = k
            break
          end
        end
        if not swap then error("Singular matrix") end
        M[i], M[swap] = M[swap], M[i]
        piv = M[i][i]
      end
      for j = i, 2 * n do
        M[i][j] = M[i][j] / piv
      end
      for k = 1, n do
        if k ~= i then
          local factor = M[k][i]
          if factor ~= 0 then
            for j = i, 2 * n do
              M[k][j] = M[k][j] - factor * M[i][j]
            end
          end
        end
      end
    end

    local Inv = {}
    for i = 1, n do
      Inv[i] = {}
      for j = 1, n do
        Inv[i][j] = M[i][n + j]
      end
    end
    return Inv
  end

  local invXtX = mat_inverse(XtX)
  local beta = {}
  for i = 1, p do
    local sum = 0
    for j = 1, p do
      sum = sum + invXtX[i][j] * Xty[j][1]
    end
    beta[i] = sum
  end
  return beta
end

-- Predict using beta coefficients
local function predict(row, beta, target_index)
  local val = beta[1]            -- bias
  local j = 2
  for i = 1, #row do
    if i ~= target_index then
      local v = row[i]
      if v == "?" or v == nil then return nil end
      val = val + beta[j] * v
      j = j + 1
    end
  end
  return val
end

-- --------------------------------------------------------------------
-- Main
-- --------------------------------------------------------------------
local function main(csv_path)
  local tbl = ezr.Tbl(csv(csv_path))
  local N = #tbl.rows

  -- ------------------ CAR-19 ------------------------------------
  print("== CAR-19 ==")
  local mpg_at = find_col(tbl, "Mpg+")
  local lbs_at = find_col(tbl, "Lbs-")
  local acc_at = find_col(tbl, "Acc+")
  if not mpg_at or not lbs_at or not acc_at then
    error("Required columns missing for CAR-19")
  end

  local budget = math.max(10, math.floor(N * 0.2))  -- evaluate 20% or at least 10 rows

  -- Scores for all rows
  local scored = {}
  for _,r in ipairs(tbl.rows) do
    local sc = score(r, mpg_at, lbs_at, acc_at)
    if sc then
      table.insert(scored, {row=r, score=sc})
    end
  end

  -- Sort descending
  table.sort(scored, function(a,b) return a.score > b.score end)

  -- Top-budget selection
  local selected = {}
  for i=1,math.min(budget,#scored) do
    selected[i] = scored[i].row
  end

  local function avg_score(rows)
    local sum, cnt = 0, 0
    for _,r in ipairs(rows) do
      local sc = score(r, mpg_at, lbs_at, acc_at)
      if sc then
        sum = sum + sc
        cnt = cnt + 1
      end
    end
    return cnt > 0 and sum / cnt or 0
  end

  local sel_avg = avg_score(selected)
  -- Random budget
  local shuffled = ezr.shuffle(tbl.rows)
  local random_rows = {}
  for i=1,budget do
    table.insert(random_rows, shuffled[i])
  end
  local rand_avg = avg_score(random_rows)

  print(string.format("Budget: %d cars evaluated", budget))
  print(string.format("Average score (selection): %.3f", sel_avg))
  print(string.format("Average score (random):    %.3f", rand_avg))
  local diff = sel_avg - rand_avg
  local pct = diff / rand_avg * 100
  print(string.format("Improvement over random:   %.3f (%.1f%%)", diff, pct))

  -- ------------------ CAR-20 ------------------------------------
  print("\n== CAR-20 ==")
  -- Split into train (first half) and test (second half)
  local half = math.floor(N / 2)
  local train_rows = {}
  local test_rows  = {}
  for i,r in ipairs(tbl.rows) do
    if i <= half then table.insert(train_rows, r) else table.insert(test_rows, r) end
  end

  -- Identify numeric columns except target
  local target_at = mpg_at
  local numeric_indices = {}
  for _,c in ipairs(tbl.cols.all) do
    if c.__index == NUM or c.__index == ezr.NUM then
      if c.at ~= target_at then
        table.insert(numeric_indices, c.at)
      end
    end
  end

  -- Build regression model on training data
  local beta = build_beta(train_rows, target_at)

  -- Historical average from training
  local sum_avg, cnt_avg = 0, 0
  for _,r in ipairs(train_rows) do
    local v = r[target_at]
    if v ~= "?" and v ~= nil then
      sum_avg = sum_avg + v
      cnt_avg = cnt_avg + 1
    end
  end
  local hist_avg = cnt_avg > 0 and sum_avg / cnt_avg or 0

  -- Evaluate on test set
  local sum_mae_model, sum_mae_hist, n_test = 0, 0, 0
  for _,r in ipairs(test_rows) do
    local actual = r[target_at]
    if actual ~= "?" and actual ~= nil then
      local pred_model = predict(r, beta, target_at)
      if pred_model then
        sum_mae_model = sum_mae_model + math.abs(pred_model - actual)
      end
      sum_mae_hist = sum_mae_hist + math.abs(hist_avg - actual)
      n_test = n_test + 1
    end
  end
  local mae_model = n_test > 0 and sum_mae_model / n_test or 0
  local mae_hist  = n_test > 0 and sum_mae_hist  / n_test or 0
  local improv = mae_hist - mae_model
  print(string.format("Test size: %d rows", n_test))
  print(string.format("Mean absolute error (model):    %.3f", mae_model))
  print(string.format("Mean absolute error (hist avg): %.3f", mae_hist))
  print(string.format("Reduction in error:            %.3f", improv))
  local pct_impr = mae_hist > 0 and improv / mae_hist * 100 or 0
  print(string.format("Percent improvement over avg:   %.1f%%", pct_impr))
end

-- Entry point
if arg[1] then
  main(arg[1])
else
  io.stderr:write("Usage: lua skill_10.lua <csv_path>\n")
end
