#!/usr/bin/env lua
-- Skill 09: NASA-17 & NASA-18
-- Usage: lua skill_09.lua <csv_path>

-- ------------------------------------------------------------------
-- Import ezr and reuse helper functions from earlier skills
local ezr = require "ezr"

local show   = ezr.show
local push   = ezr.push
local sorted = ezr.sorted
local map    = ezr.map
local keys   = ezr.keys
local slice  = ezr.slice
local rand   = ezr.rand
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

-- k‑NN prediction (copied from skill_08)
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
-- Helper to compute mean and sd of a numeric list
local function mean_sd(vals)
  local sum, sqsum, n = 0, 0, 0
  for _,v in ipairs(vals) do
    sum = sum + v
    sqsum = sqsum + v*v
    n = n + 1
  end
  local mu = sum / n
  local var = (sqsum / n - mu*mu)
  local sd = math.sqrt(math.max(var, 0))
  return mu, sd
end

-- ------------------------------------------------------------------
-- NASA‑17 : Flag projects unusually different from the rest
local function nasa_17(tbl)
  print("== NASA-17 ==")

  local center = tbl:mids()  -- centroid row
  local dists = map(tbl.rows, function(r) return tbl:distx(r, center) end)
  local mu, sd = mean_sd(dists)
  local thresh = mu + 2*sd

  local outliers = {}
  for i, r in ipairs(tbl.rows) do
    local d = dists[i]
    if d > thresh then
      push(outliers, {id=r[1], dist=d})
    end
  end

  local n_out = #outliers
  print(string.format("Total projects: %d", #tbl.rows))
  print(string.format("Outliers detected: %d", n_out))
  if n_out > 0 then
    print("Top 5 most atypical projects:")
    table.sort(outliers, function(a,b) return a.dist > b.dist end)
    for i, o in ipairs(slice(outliers, 1, math.min(5, n_out))) do
      print(string.format("  %d. id %s – distance %.2f", i, o.id, o.dist))
    end
  end
end

-- ------------------------------------------------------------------
-- NASA‑18 : Flag projects whose effort or defects deviate from expectation
local function nasa_18(tbl)
  print("\n== NASA-18 ==")

  local effort_col = find_col(tbl, "Effort-")
  local defect_col = find_col(tbl, "Defects-")

  local effort_preds = {}
  local defect_preds = {}
  local effort_resid = {}
  local defect_resid = {}

  for _,r in ipairs(tbl.rows) do
    -- Effort
    if r[effort_col.at] ~= "?" then
      local pred = knn_predict(tbl, r, effort_col, 5)
      local resid = r[effort_col.at] - pred
      push(effort_preds, {id=r[1], actual=r[effort_col.at], pred=pred, resid=resid})
      push(effort_resid, math.abs(resid))
    end
    -- Defects
    if r[defect_col.at] ~= "?" then
      local pred = knn_predict(tbl, r, defect_col, 5)
      local resid = r[defect_col.at] - pred
      push(defect_preds, {id=r[1], actual=r[defect_col.at], pred=pred, resid=resid})
      push(defect_resid, math.abs(resid))
    end
  end

  -- Thresholds: mean + 2*sd of absolute residuals
  local eff_mu, eff_sd = mean_sd(effort_resid)
  local def_mu, def_sd = mean_sd(defect_resid)
  local eff_thresh = eff_mu + 2*eff_sd
  local def_thresh = def_mu + 2*def_sd

  local eff_outliers = {}
  for _,p in ipairs(effort_preds) do
    if math.abs(p.resid) > eff_thresh then
      push(eff_outliers, p)
    end
  end

  local def_outliers = {}
  for _,p in ipairs(defect_preds) do
    if math.abs(p.resid) > def_thresh then
      push(def_outliers, p)
    end
  end

  print(string.format("Effort outliers: %d", #eff_outliers))
  if #eff_outliers > 0 then
    print("  Top 5 effort outliers:")
    table.sort(eff_outliers, function(a,b) return math.abs(a.resid) > math.abs(b.resid) end)
    for i, p in ipairs(slice(eff_outliers, 1, math.min(5, #eff_outliers))) do
      print(string.format("    %d. id %s – actual %.1f, predicted %.1f, diff %.1f",
            i, p.id, p.actual, p.pred, p.resid))
    end
  end

  print(string.format("Defect outliers: %d", #def_outliers))
  if #def_outliers > 0 then
    print("  Top 5 defect outliers:")
    table.sort(def_outliers, function(a,b) return math.abs(a.resid) > math.abs(b.resid) end)
    for i, p in ipairs(slice(def_outliers, 1, math.min(5, #def_outliers))) do
      print(string.format("    %d. id %s – actual %.1f, predicted %.1f, diff %.1f",
            i, p.id, p.actual, p.pred, p.resid))
    end
  end
end

-- ------------------------------------------------------------------
-- Main
local function main()
  local path = arg[1]
  if not path then
    io.stderr:write("Usage: lua skill_09.lua <csv_path>\n")
    os.exit(1)
  end

  local tbl = ezr.Tbl(ezr.csv(path))

  nasa_17(tbl)
  nasa_18(tbl)
end

main()
