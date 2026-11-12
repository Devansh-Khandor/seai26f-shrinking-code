#!/usr/bin/env lua
-- Skill 10: NASA-19 & NASA-20
-- Usage: lua skill_10.lua <csv_path>

-- Import core library
local ezr = require "ezr"

local show   = ezr.show
local push   = ezr.push
local sorted = ezr.sorted
local map    = ezr.map
local rand   = ezr.rand
local slice  = ezr.slice
local copy   = ezr.copy

-- deterministic random stream
ezr.srand(ezr.the.seed)

-- ------------------------------------------------------------------
-- Common helpers -----------------------------------------------------
local function find_col(tbl, name)
  for _,c in pairs(tbl.cols.all) do
    if c.name == name then return c end
  end
  error("column not found: " .. name)
end

-- Simple mean/median/SD helper used by several skills
local function col_stats(tbl, col)
  local vals = map(tbl.rows, function(r) return r[col.at] end)
  local sorted_vals = sorted(vals)
  local n = #sorted_vals
  local mean = col.mu
  local sd   = col:div()
  local med  = sorted_vals[(n + 1) // 2]
  return {
    name   = col.name,
    mean   = mean,
    median = med,
    sd     = sd,
    min    = sorted_vals[1],
    max    = sorted_vals[n]
  }
end

-- k‑NN prediction (copied from earlier skills)
local function knn_predict(tbl, proposal, goal_col, k)
  local labeled = {}
  for _,r in ipairs(tbl.rows) do
    if r[goal_col.at] ~= "?" then
      -- ignore rows with missing independent data
      local missing = false
      for _,c in ipairs(tbl.cols.x) do
        if r[c.at] == "?" then missing = true; break end
      end
      if not missing then push(labeled, r) end
    end
  end
  if #labeled == 0 then return nil end

  local function dist_to(r) return tbl:distx(proposal, r) end
  local sorted_rows = sorted(labeled, function(a,b) return dist_to(a) < dist_to(b) end)

  local k_use = math.min(k, #sorted_rows)
  local w_sum, val_sum = 0, 0
  for i=1,k_use do
    local r = sorted_rows[i]
    local d = dist_to(r)
    local w = 1/(d+1e-6)
    w_sum   = w_sum + w
    val_sum = val_sum + w * r[goal_col.at]
  end
  return val_sum / w_sum
end

-- ------------------------------------------------------------------
-- NASA‑19 : compare knn vs historical‑average predictions
local function nasa_19(tbl)
  print("== NASA-19 ==")

  -- Choose goal column to evaluate; use Effort- here
  local goal_col = find_col(tbl, "Effort-")
  if goal_col == nil then
    print("No goal column found for Effort-")
    return
  end

  -- Shuffle rows and split 70/30 train/test
  local rows_shuffled = shuffle(tbl.rows)
  local n_train = math.floor(#rows_shuffled * 0.7)
  local train_rows = slice(rows_shuffled, 1, n_train)
  local test_rows  = slice(rows_shuffled, n_train+1)

  -- Build training TBL with the same header
  local header = tbl.rows[1]
  local train_tbl = ezr.Tbl({header})
  for _,r in ipairs(train_rows) do train_tbl:add(r) end

  -- Training mean for historical estimate
  local train_goal_col = train_tbl.cols.all[goal_col.at]
  local mean_hist = train_goal_col.mu

  -- Collect errors
  local err_hist = {}
  local err_knn  = {}
  for _,r in ipairs(test_rows) do
    local actual = r[goal_col.at]
    if actual ~= "?" then
      local pred_hist = mean_hist
      push(err_hist, math.abs(actual - pred_hist))

      local pred_knn = knn_predict(train_tbl, r, goal_col, 5)
      if pred_knn then
        push(err_knn, math.abs(actual - pred_knn))
      end
    end
  end

  -- Compute MAE
  local function avg(lst)
    if #lst == 0 then return 0 end
    local sum = 0
    for _,v in ipairs(lst) do sum = sum + v end
    return sum / #lst
  end
  local mae_hist = avg(err_hist)
  local mae_knn  = avg(err_knn)

  print(string.format("Historical‑avg MAE: %.2f", mae_hist))
  print(string.format("k‑NN MAE      : %.2f", mae_knn))
  if mae_hist > 0 then
    local improvement = (mae_hist - mae_knn) / mae_hist * 100
    print(string.format("Improvement over history: %.1f%%", improvement))
  end
end

-- ------------------------------------------------------------------
-- NASA‑20 : compare selection approach vs random selection
local function nasa_20(tbl)
  print("\n== NASA-20 ==")

  local effort_col = find_col(tbl, "Effort-")
  local defect_col = find_col(tbl, "Defects-")

  -- Gather proposals: rows missing either goal
  local proposals = {}
  for _,r in ipairs(tbl.rows) do
    if r[effort_col.at] == "?" or r[defect_col.at] == "?" then
      push(proposals, r)
    end
  end

  if #proposals == 0 then
    print("No proposals with missing goals found.")
    return
  end

  -- Overall means used in scoring
  local eff_mean = tbl.cols.all[effort_col.at].mu
  local def_mean = tbl.cols.all[defect_col.at].mu

  -- Predict for each proposal and compute composite score
  local scored = {}
  for _,p in ipairs(proposals) do
    local pred_eff = knn_predict(tbl, p, effort_col, 5)
    local pred_def = knn_predict(tbl, p, defect_col, 5)
    if pred_eff and pred_def then
      local score = (eff_mean - pred_eff)/eff_mean + (def_mean - pred_def)/def_mean
      push(scored, {row=p, pred_eff=pred_eff, pred_def=pred_def, score=score})
    end
  end

  if #scored == 0 then
    print("No proposals could be predicted.")
    return
  end

  -- Determine evaluation budget: number of proposals we will actually evaluate
  -- We take the top half of the scored list as a reasonable budget
  local budget = math.max(1, math.floor(#scored / 2))

  -- Our selection: top‑budget by composite score
  table.sort(scored, function(a,b) return a.score > b.score end)
  local selected = slice(scored, 1, budget)

  -- Random selection: shuffle and pick the same number
  local rand_sel = shuffle(scored)
  rand_sel = slice(rand_sel, 1, budget)

  local function avg_score(list) -- average composite score
    local sum = 0
    for _,p in ipairs(list) do sum = sum + p.score end
    return sum / #list
  end

  local avg_sel = avg_score(selected)
  local avg_rand = avg_score(rand_sel)

  print(string.format("Evaluation budget (proposals evaluated): %d", budget))
  print(string.format("Average composite score of our selection: %.3f", avg_sel))
  print(string.format("Average composite score of random selection: %.3f", avg_rand))
  if avg_sel > avg_rand then
    print(string.format("Our selection outperforms random by %.3f points.", avg_sel - avg_rand))
  else
    print(string.format("Random selection is better by %.3f points.", avg_rand - avg_sel))
  end
end

-- ------------------------------------------------------------------
-- Main
local function main()
  local path = arg[1]
  if not path then
    io.stderr:write("Usage: lua skill_10.lua <csv_path>\n")
    os.exit(1)
  end

  local tbl = ezr.Tbl(ezr.csv(path))

  nasa_19(tbl)
  nasa_20(tbl)
end

main()
