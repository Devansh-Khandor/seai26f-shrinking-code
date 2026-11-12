#!/usr/bin/env lua
-- Skill 07: NASA-13 & NASA-14
-- Usage: lua skill_07.lua <csv_path>

-- ------------------------------------------------------------------
-- Import ezr and reuse helper functions from previous skills
local ezr = require "ezr"

local show   = ezr.show
local push   = ezr.push
local sorted = ezr.sorted
local map    = ezr.map
local keys   = ezr.keys
local slice  = ezr.slice
local rand   = ezr.rand

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

local function median(vals)
  local s = sorted(vals)
  local n = #s
  return s[(n + 1) // 2]
end

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

-- k‑NN prediction (copied from skill_04/06)
local function knn_predict(tbl, proposal, goal_col, k)
  local labeled = {}
  for _,r in ipairs(tbl.rows) do
    if r[goal_col.at] ~= "?" then
      local missing = false
      for _,c in ipairs(tbl.cols.x) do
        if r[c.at] == "?" then missing = true; break end
      end
      if not missing then push(labeled, r) end
    end
  end
  if #labeled == 0 then return nil end

  local function dist_to(r) return tbl:distx(proposal, r) end
  local sorted_rows = sorted(labeled, function(a,b)
    return dist_to(a) < dist_to(b)
  end)

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
-- NASA‑13 : Identify promising characteristic combinations
local function nasa_13(tbl)
  print("== NASA-13 ==")
  local effort_col = find_col(tbl, "Effort-")
  local defect_col = find_col(tbl, "Defects-")

  local effort_mean = effort_col.mu
  local defect_mean = defect_col.mu

  -- Collect numeric columns
  local num_cols = {}
  for _,c in ipairs(tbl.cols.x) do
    if type(c:mid()) == "number" then
      push(num_cols, c)
    end
  end

  -- For each numeric column, split rows into low/high by median
  local low_sets = {}
  local high_sets = {}
  for _,c in ipairs(num_cols) do
    local med = c:median()
    local low, high = {}, {}
    for _,r in ipairs(tbl.rows) do
      local v = r[c.at]
      if v ~= "?" then
        if v <= med then push(low, r) else push(high, r) end
      end
    end
    low_sets[c.at] = low
    high_sets[c.at] = high
  end

  -- Helper: average goals in a set of rows
  local function avg_goals(rows)
    local sum_eff, sum_def = 0, 0
    local cnt = 0
    for _,r in ipairs(rows) do
      if r[effort_col.at] ~= "?" and r[defect_col.at] ~= "?" then
        sum_eff = sum_eff + r[effort_col.at]
        sum_def = sum_def + r[defect_col.at]
        cnt = cnt + 1
      end
    end
    if cnt == 0 then return nil, nil end
    return sum_eff/cnt, sum_def/cnt
  end

  -- Build single‑column low combos
  local combos = {}
  for _,c in ipairs(num_cols) do
    local rows = low_sets[c.at]
    local avg_e, avg_d = avg_goals(rows)
    if avg_e and avg_d and avg_e < effort_mean and avg_d < defect_mean then
      local score = (effort_mean - avg_e)/effort_mean + (defect_mean - avg_d)/defect_mean
      push(combos, {cols={c}, avg_e=avg_e, avg_d=avg_d, score=score})
    end
  end

  -- Build two‑column low combos (intersection)
  for i=1,#num_cols do
    for j=i+1,#num_cols do
      local ci, cj = num_cols[i], num_cols[j]
      local set_i, set_j = low_sets[ci.at], low_sets[cj.at]
      -- intersection
      local intersect = {}
      local key = {}
      for _,r in ipairs(set_i) do key[r] = true end
      for _,r in ipairs(set_j) do
        if key[r] then push(intersect, r) end
      end
      local avg_e, avg_d = avg_goals(intersect)
      if avg_e and avg_d and avg_e < effort_mean and avg_d < defect_mean then
        local score = (effort_mean - avg_e)/effort_mean + (defect_mean - avg_d)/defect_mean
        push(combos, {cols={ci,cj}, avg_e=avg_e, avg_d=avg_d, score=score})
      end
    end
  end

  -- Sort combos by score descending
  table.sort(combos, function(a,b) return a.score > b.score end)

  -- Show top 3
  if #combos == 0 then
    print("No promising low‑effort/defect combinations found.")
  else
    for i=1,math.min(3,#combos) do
      local c = combos[i]
      local names = {}
      for _,col in ipairs(c.cols) do push(names, col.name .. " low") end
      print(string.format("%d. %s", i, table.concat(names, ", ")))
      print(string.format("   Avg Effort = %.1f, Avg Defects = %.1f",
            c.avg_e, c.avg_d))
    end
  end
end

-- ------------------------------------------------------------------
-- NASA‑14 : Recommend proposals to evaluate first
local function nasa_14(tbl)
  print("== NASA-14 ==")
  local effort_col = find_col(tbl, "Effort-")
  local defect_col = find_col(tbl, "Defects-")
  local effort_mean = effort_col.mu
  local defect_mean = defect_col.mu

  -- Find proposals (missing at least one goal)
  local proposals = {}
  for _,r in ipairs(tbl.rows) do
    if r[effort_col.at] == "?" or r[defect_col.at] == "?" then
      push(proposals, r)
    end
  end

  if #proposals == 0 then
    print("No proposals with missing effort or defects to evaluate.")
    return
  end

  -- Predict and score each proposal
  local scored = {}
  for _,p in ipairs(proposals) do
    local pred_eff = knn_predict(tbl, p, effort_col, 5)
    local pred_def = knn_predict(tbl, p, defect_col, 5)
    pred_eff = pred_eff or effort_mean
    pred_def = pred_def or defect_mean
    local score = (effort_mean - pred_eff)/effort_mean + (defect_mean - pred_def)/defect_mean
    push(scored, {row=p, pred_eff=pred_eff, pred_def=pred_def, score=score})
  end

  -- Sort by score descending (most promising first)
  table.sort(scored, function(a,b) return a.score > b.score end)

  -- Show top N proposals (budgeted, e.g., 5)
  local budget = math.min(5, #scored)
  print(string.format("Top %d proposals to evaluate first:", budget))
  for i=1,budget do
    local s = scored[i]
    local id = s.row[1]
    print(string.format("%d. Proposal id %s: Pred Effort=%.1f, Pred Defects=%.1f",
          i, id, s.pred_eff, s.pred_def))
  end
end

-- ------------------------------------------------------------------
-- Main
local function main()
  local path = arg[1]
  if not path then
    io.stderr:write("Usage: lua skill_07.lua <csv_path>\n")
    os.exit(1)
  end

  local tbl = ezr.Tbl(ezr.csv(path))

  nasa_13(tbl)
  print()
  nasa_14(tbl)
end

main()
