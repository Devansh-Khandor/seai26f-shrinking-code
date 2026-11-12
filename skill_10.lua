#!/usr/bin/env lua
-- Skill 10 – Apache performance comparison & prediction assessment
-- Requires the ezr library (ezr.lua & ezr-lib.lua in the same dir)

local ezr = require"ezr"

-- ------------------------------------------------------------------
-- Get the CSV file path from the command line
-- ------------------------------------------------------------------
local csv_path = arg[1]
assert(csv_path, "Usage: lua skill_10.lua <csv path>")

-- ------------------------------------------------------------------
-- Load the data into an ezr table
-- ------------------------------------------------------------------
local tbl = ezr.Tbl(ezr.csv(csv_path))

-- ------------------------------------------------------------------
-- Common helpers
-- ------------------------------------------------------------------
local perf_at = tbl.cols.klass.at
local budget  = the.budget or 50
local n_rows  = #tbl.rows
local rng     = math.random

-- Helper: find min and avg of a list of numbers
local function min_avg(vals)
  local min_val = math.huge
  local sum = 0
  for _,v in ipairs(vals) do
    sum = sum + v
    if v < min_val then min_val = v end
  end
  return min_val, sum / (#vals > 0 and #vals or 1)
end

-- ------------------------------------------------------------------
-- APACHE-19 – Compare configuration-selection vs random
-- ------------------------------------------------------------------
print("== APACHE-19 ==")

-- Acquire a set of configurations using the system
local lab = tbl:acquirer(budget)

-- Random selection of same budget (or less if dataset smaller)
local rand_rows = {}
for _,r in ipairs(shuffle(tbl.rows)) do
  if #rand_rows < math.min(budget, n_rows) then
    table.insert(rand_rows, r)
  else
    break
  end
end

-- Compute best performances
local acq_vals  = {}
for _,r in ipairs(lab) do table.insert(acq_vals, r[perf_at]) end
local rand_vals = {}
for _,r in ipairs(rand_rows) do table.insert(rand_vals, r[perf_at]) end

local acq_best, acq_avg = min_avg(acq_vals)
local rand_best, rand_avg = min_avg(rand_vals)

print(string.format("Budget used: %d configurations", #lab))
print(string.format("Best performance from acquirer: %s (avg %s)",
        ezr.round(acq_best), ezr.round(acq_avg)))
print(string.format("Best performance from random:   %s (avg %s)",
        ezr.round(rand_best), ezr.round(rand_avg)))

if acq_best < rand_best then
  print("Acquirer discovered better configurations than random.")
elseif rand_best < acq_best then
  print("Random selection found a better configuration than the acquirer.")
else
  print("Both methods found identical best performance.")
end

-- ------------------------------------------------------------------
-- APACHE-20 – Predictive performance vs historical average
-- ------------------------------------------------------------------
print("\n== APACHE-20 ==")

-- Build a predictive tree from the acquired set
local tree = ezr.Tree(tbl, lab)

-- Prepare training index set for fast lookup
local train_idx = {}
for idx,_ in ipairs(tbl.rows) do train_idx[idx] = false end
for _,r in ipairs(lab) do
  local pos = table.indexOf(tbl.rows, r)
  if pos then train_idx[pos] = true end
end

-- Helper: find index of a row in a table (identity comparison)
function table.indexOf(t, v)
  for i, x in ipairs(t) do
    if x == v then return i end
  end
end

-- Collect predictions and errors for unseen rows
local pred_errs = {}
local base_errs = {}
local base_mean = tbl.cols.klass:mid()   -- overall mean
for idx, r in ipairs(tbl.rows) do
  if not train_idx[idx] then
    local actual = r[perf_at]
    local pred = tree:leaf(tbl, r)
    table.insert(pred_errs, math.abs(pred - actual))
    table.insert(base_errs, math.abs(base_mean - actual))
  end
end

local function mean(lst)
  if #lst == 0 then return 0 end
  local sum = 0
  for _,v in ipairs(lst) do sum = sum + v end
  return sum / #lst
end

local mae_pred = mean(pred_errs)
local mae_base = mean(base_errs)
local improv = (mae_base - mae_pred) / mae_base * 100

print(string.format("MAE of tree predictions: %s",
        ezr.round(mae_pred)))
print(string.format("MAE of historical mean baseline: %s",
        ezr.round(mae_base)))
print(string.format("Improvement over baseline: %.1f%% (lower error better)",
        ezr.round(improv)))
