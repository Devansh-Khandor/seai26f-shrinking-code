#!/usr/bin/env lua
-- Skill 10: Comparison studies
-- CAR-19: Selection system vs random under equal evaluation budget
-- CAR-20: Tree‑based estimates vs simple historical average

local ezr = require("ezr")
local show, round, keysort = ezr.show, ezr.round, ezr.keysort
local rand, shuffle, slice = ezr.rand, ezr.shuffle, ezr.slice

-- -----------------------------------------------------------------
-- Utility helpers (copy from previous skills)
-- -----------------------------------------------------------------
local function col_by_name(tbl, name)
  for _, col in ipairs(tbl.cols.all) do
    if col.name == name then return col end
  end
  error("column " .. name .. " not found")
end

local function minmax(col, rows)
  local minv, maxv = math.huge, -math.huge
  for _, r in ipairs(rows) do
    local v = r[col.at]
    if v ~= "?" then
      if v < minv then minv = v end
      if v > maxv then maxv = v end
    end
  end
  return minv, maxv
end

local function norm(v, minv, maxv, reverse)
  if maxv == minv then return 0.5 end
  local n = (v - minv) / (maxv - minv)
  return reverse and 1 - n or n
end

local function score_for_row(row, wmin, wmax, mmin, mmax, amin, amax, wcol, mcol, acol)
  local w = row[wcol.at]
  local m = row[mcol.at]
  local a = row[acol.at]
  if w == "?" or m == "?" or a == "?" then return nil end
  local ws = 1 - norm(w, wmin, wmax)
  local ms = norm(m, mmin, mmax)
  local as = 1 - norm(a, amin, amax)
  return ws + ms + as
end

local function top_indices(rows, scores, n)
  local idx = {}
  for i = 1, #rows do idx[i] = i end
  table.sort(idx, function(i, j)
    local si, sj = scores[i] or -math.huge, scores[j] or -math.huge
    return si > sj
  end)
  local top = {}
  for i = 1, math.min(n, #idx) do
    top[i] = idx[i]
  end
  return top
end

-- -----------------------------------------------------------------
-- CAR-19: Selection vs random
-- -----------------------------------------------------------------
print("== CAR-19 ==")

local path = arg[1]
if not path then
  io.stderr:write("Usage: lua skill_10.lua <csv path>\n")
  os.exit(1)
end

local tbl = ezr.Tbl(ezr.csv(path))
local rows = tbl.rows

-- Columns for scoring
local wcol = col_by_name(tbl, "Lbs-")
local mcol = col_by_name(tbl, "Mpg+")
local acol = col_by_name(tbl, "Acc+")
local wmin, wmax = minmax(wcol, rows)
local mmin, mmax = minmax(mcol, rows)
local amin, amax = minmax(acol, rows)

-- Compute scores for all rows
local scores = {}
for i, r in ipairs(rows) do
  scores[i] = score_for_row(r, wmin, wmax, mmin, mmax, amin, amax, wcol, mcol, acol)
end

-- How many vehicles to evaluate
local eval_count = 5
if #rows < eval_count then eval_count = #rows end

-- Top vehicles
local top_ids = top_indices(rows, scores, eval_count)
local top_sum, top_cnt = 0, 0
for _, i in ipairs(top_ids) do
  local s = scores[i]
  if s then top_sum, top_cnt = top_sum + s, top_cnt + 1 end
end
local top_avg = top_sum / (top_cnt or 1)

-- Random sample
local rand_ids = {}
for i=1,eval_count do
  local idx = rand(1, #rows)
  rand_ids[i] = idx
end
local rand_sum, rand_cnt = 0, 0
for _, i in ipairs(rand_ids) do
  local s = scores[i]
  if s then rand_sum, rand_cnt = rand_sum + s, rand_cnt + 1 end
end
local rand_avg = rand_sum / (rand_cnt or 1)

-- Improvement
local improv = (top_avg - rand_avg) / (rand_avg + ezr.TINY) * 100

print(string.format(
  "Average score of best %d vehicles: %.2f",
  eval_count, round(top_avg,2)))
print(string.format(
  "Average score of %d random vehicles: %.2f",
  eval_count, round(rand_avg,2)))
print(string.format(
  "Improvement over random: %.1f%%", round(improv,1)))

-- -----------------------------------------------------------------
-- CAR-20: Tree vs historical average on unseen records
-- -----------------------------------------------------------------
print("\n== CAR-20 ==")

-- Simple helper: build a regression tree for a numeric target
local function build_tree(tbl, target_col, max_depth)
  local Y = function(row) return row[target_col.at] end
  return ezr.Tree(tbl, tbl.rows, Y, nil, max_depth or 4)
end

-- Helper to get leaf node
local function leaf_node(tree, tbl, row)
  local node = tree
  while node.at do
    local col = tbl.cols.all[node.at]
    if col:holds(row[col.at], node.v) then
      node = node.yes
    else
      node = node.no
    end
  end
  return node
end

-- Split into train / test
local shuffled = shuffle(rows)
local n = #shuffled
local split = math.floor(n / 2)
local train_rows = slice(shuffled, 1, split)
local test_rows  = slice(shuffled, split + 1)

-- Build tree on training data
local train_tbl = tbl:clone(train_rows)
local t_tree = build_tree(train_tbl, mpg_col)

-- Train mean (historical average)
local train_mean = 0
local train_count = 0
for _, r in ipairs(train_rows) do
  local v = r[mpg_col.at]
  if v ~= "?" then
    train_mean, train_count = train_mean + v, train_count + 1
  end
end
train_mean = train_count > 0 and train_mean / train_count or 0

-- Evaluate on test set
local test_errors_tree = {}
local test_errors_mean = {}
for _, r in ipairs(test_rows) do
  local actual = r[mpg_col.at]
  if actual ~= "?" then
    -- Tree estimate
    local node = leaf_node(t_tree, train_tbl, r)
    local est_tree = node.mu
    table.insert(test_errors_tree, math.abs(est_tree - actual))
    -- Baseline estimate
    table.insert(test_errors_mean, math.abs(train_mean - actual))
  end
end

local function mean(tbl)
  if #tbl == 0 then return 0 end
  local sum = 0
  for _, v in ipairs(tbl) do sum = sum + v end
  return sum / #tbl
end

local mae_tree = mean(test_errors_tree)
local mae_mean = mean(test_errors_mean)

local improv_mae = (mae_mean - mae_tree) / (mae_mean + ezr.TINY) * 100

print(string.format(
  "Tree MAE on unseen records: %.3f", round(mae_tree,3)))
print(string.format(
  "Historical‑average MAE: %.3f", round(mae_mean,3)))
print(string.format(
  "MAE improvement: %.1f%%", round(improv_mae,1)))
