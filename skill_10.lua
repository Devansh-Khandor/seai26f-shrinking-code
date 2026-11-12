#!/usr/bin/env lua
-- Skill 10: NASA-19 & NASA-20 – Evaluation of effort model and proposal selection
-- Usage: lua skill_10.lua <csv_path>

local ezr = require("ezr-lib")
local csv  = ezr.csv
local round = ezr.round
local srand = ezr.srand
local rand  = ezr.rand

-- Simple statistics ----------------------------------------------------
local function mean(t)
  local s, n = 0, #t
  for _,v in ipairs(t) do s = s + v end
  return n>0 and s/n or 0 end

local function mse(t, p)
  local n = #t
  local ss = 0
  for i=1,n do
    local d = t[i] - p[i]
    ss = ss + d*d
  end
  return ss / n end

local function mae(t, p)
  local n = #t
  local sum = 0
  for i=1,n do
    sum = sum + math.abs(t[i] - p[i]) end
  return sum / n end

-- Linear regression ----------------------------------------------------
local function linear_regression(x, y)
  local n = #x
  if n < 2 then return 0,0,0 end
  local sx, sy = 0,0
  for i=1,n do sx = sx + x[i]; sy = sy + y[i] end
  local mx, my = sx/n, sy/n
  local cov, varx = 0,0
  for i=1,n do
    local dx = x[i] - mx
    local dy = y[i] - my
    cov = cov + dx*dy
    varx = varx + dx*dx
  end
  local slope = cov / varx
  local intercept = my - slope*mx
  return slope, intercept, math.sqrt(cov*cov / varx) -- residual SD
end

-- Predict effort and defects ------------------------------------------
local function predict(kx, slope_e, intercept_e, slope_d, intercept_d)
  local pe = slope_e * kx + intercept_e
  local pd = slope_d * kx + intercept_d
  return pe, pd end

-- ----------------------------------------------------------------------
-- Main routine ----------------------------------------------------------
-- ----------------------------------------------------------------------
local function main()
  local path = arg[1]
  if not path then
    io.stderr:write("Usage: lua skill_10.lua <csv_path>\n")
    os.exit(1)
  end

  -- Read all rows -------------------------------------------------------
  local rows, header, idx = {}, nil, {}
  for rec in csv(path) do
    if not header then
      header = rec
      for i,v in ipairs(rec) do idx[v] = i end
    else
      rows[#rows+1] = rec
    end
  end

  if #rows == 0 then
    print("No data rows found.")
    os.exit(0)
  end

  -- Separate completed projects (both Effort- and Defects- present) ----
  local completed = {}
  for _,r in ipairs(rows) do
    local e = tonumber(r[idx["Effort-"]])
    local d = tonumber(r[idx["Defects-"]])
    if e and d then
      table.insert(completed, r)
    end
  end

  if #completed < 4 then
    print("Not enough completed projects to train models.")
    os.exit(0)
  end

  -- ---------------------------------------------------------------
  -- NASA-19: Compare regression to historical average
  -- ---------------------------------------------------------------
  -- Split into train/test: 80% train, 20% test
  srand(123456789)   -- deterministic seed
  local shuffled = {}
  for _,r in ipairs(completed) do
    table.insert(shuffled, r)
  end
  for i=#shuffled,2,-1 do
    local j = rand(i)
    shuffled[i], shuffled[j] = shuffled[j], shuffled[i]
  end
  local ntrain = math.floor(#shuffled * 0.8)
  local train, test = {}, {}
  for i,r in ipairs(shuffled) do
    if i <= ntrain then table.insert(train, r)
    else table.insert(test, r) end
  end

  -- Build regression on training set
  local kx_t, y_e_t, y_d_t = {}, {}, {}
  for _,r in ipairs(train) do
    local k = tonumber(r[idx["Kloc+"]]) or tonumber(r[idx["Kloc"]])
    local e = tonumber(r[idx["Effort-"]])
    local d = tonumber(r[idx["Defects-"]])
    if k and e and d then
      table.insert(kx_t, k)
      table.insert(y_e_t, e)
      table.insert(y_d_t, d)
    end
  end
  local slope_e, intercept_e, _ = linear_regression(kx_t, y_e_t)
  local slope_d, intercept_d, _ = linear_regression(kx_t, y_d_t)

  -- Predict on test set
  local actual, pred = {}, {}
  for _,r in ipairs(test) do
    local k = tonumber(r[idx["Kloc+"]]) or tonumber(r[idx["Kloc"]])
    local e = tonumber(r[idx["Effort-"]])
    if k and e then
      local pe, _ = predict(k, slope_e, intercept_e, slope_d, intercept_d)
      table.insert(actual, e)
      table.insert(pred, pe)
    end
  end

  local mae_pred = mae(actual, pred)
  local mae_base = mae(actual, table.map(actual, function(_) return mean(actual) end))
  local improvement = (mae_base - mae_pred) / mae_base * 100

  print("== NASA-19 ==")
  print(string.format("Regression MAE   = %.2f", round(mae_pred,2)))
  print(string.format("Historical MAE   = %.2f", round(mae_base,2)))
  print(string.format("Improvement      = %.1f%%", round(improvement,1)))

  -- ---------------------------------------------------------------
  -- NASA-20: Compare proposal selection vs random
  -- ---------------------------------------------------------------
  -- Simulate proposals by holding out 30% of completed projects
  srand(987654321)
  shuffled = {}
  for _,r in ipairs(completed) do table.insert(shuffled, r) end
  for i=#shuffled,2,-1 do
    local j = rand(i)
    shuffled[i], shuffled[j] = shuffled[j], shuffled[i]
  end
  local nprop = math.floor(#shuffled * 0.3)
  local proposals, remaining = {}, {}
  for i,r in ipairs(shuffled) do
    if i <= nprop then table.insert(proposals, r)
    else table.insert(remaining, r) end
  end

  -- Use full training set (completed) to build model
  local kx_all, y_e_all, y_d_all = {}, {}, {}
  for _,r in ipairs(remaining) do
    local k = tonumber(r[idx["Kloc+"]]) or tonumber(r[idx["Kloc"]])
    local e = tonumber(r[idx["Effort-"]])
    local d = tonumber(r[idx["Defects-"]])
    if k and e and d then
      table.insert(kx_all, k)
      table.insert(y_e_all, e)
      table.insert(y_d_all, d)
    end
  end
  local slope_e_all, intercept_e_all, _ = linear_regression(kx_all, y_e_all)
  local slope_d_all, intercept_d_all, _ = linear_regression(kx_all, y_d_all)

  -- Evaluate all proposals: compute predicted sum and actual sum
  local pred_sum, act_sum = {}, {}
  for _,r in ipairs(proposals) do
    local k = tonumber(r[idx["Kloc+"]]) or tonumber(r[idx["Kloc"]])
    local e = tonumber(r[idx["Effort-"]])
    local d = tonumber(r[idx["Defects-"]])
    if k and e and d then
      local pe, pd = predict(k, slope_e_all, intercept_e_all,
                              slope_d_all, intercept_d_all)
      table.insert(pred_sum, pe + pd)
      table.insert(act_sum, e + d)
    end
  end

  -- Selection strategy: pick top N proposals with lowest predicted sum
  local N = math.max(1, math.floor(#proposals * 0.1)) -- 10% of proposals
  local idxs = {}
  for i=1,#proposals do idxs[i] = i end
  table.sort(idxs, function(a,b) return pred_sum[a] < pred_sum[b] end)
  local selected_idx = {unpack(idxs, 1, N)}

  local act_sel_sum, act_sel_count = 0, 0
  for _,i in ipairs(selected_idx) do
    act_sel_sum = act_sel_sum + act_sum[i]
    act_sel_count = act_sel_count + 1 end
  local avg_sel = act_sel_sum / act_sel_count

  -- Random selection of same number
  srand(192837465)
  local rand_idxs = {}
  for i=1,#proposals do rand_idxs[i] = i end
  table.shuffle = function(t)
    for i=#t,2,-1 do
      local j = rand(i)
      t[i],t[j] = t[j],t[i]
    end
  end
  table.shuffle(rand_idxs)
  local random_idx = {unpack(rand_idxs,1,N)}
  local act_rand_sum, act_rand_count = 0, 0
  for _,i in ipairs(random_idx) do
    act_rand_sum = act_rand_sum + act_sum[i]
    act_rand_count = act_rand_count + 1 end
  local avg_rand = act_rand_sum / act_rand_count

  print("\n== NASA-20 ==")
  print(string.format("Selected (predicted best) average total effort+defects = %.2f",
        round(avg_sel,2)))
  print(string.format("Random average total effort+defects = %.2f",
        round(avg_rand,2)))
  if avg_sel < avg_rand then
    print("Our selection strategy yields lower average cost.")
  else
    print("Random selection performs better or equal.")
  end
end

main()
