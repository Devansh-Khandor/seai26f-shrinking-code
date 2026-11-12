#!/usr/bin/env lua
-- Skill 09 – Outlier detection and record‑accuracy flagging
-- Usage:  lua skill_09.lua <csv path>

local ezr = require "ezr"

-- ------------------------------------------------------------------
-- Helper: compute a best linear predictor for a target column
-- ------------------------------------------------------------------
local function best_predictor(tbl, target_name)
  local target_col = tbl.cols.all[target_name]
  assert(target_col, "Target column not found: " .. target_name)

  local best = nil
  local best_corr = 0
  local best_coeff = 0
  local best_intercept = 0
  local best_resid_sd = 0
  local best_predictor_col = nil

  local n_all = 0
  -- try every other numeric predictor
  for _, c in ipairs(tbl.cols.x) do
    local pred_col = tbl.cols.all[c.at]
    if pred_col ~= target_col then
      local n, sum_xy, sum_x, sum_y = 0, 0, 0, 0
      for _, r in ipairs(tbl.rows) do
        local x = r[pred_col.at]
        local y = r[target_col.at]
        if x ~= "?" and y ~= "?" then
          sum_x = sum_x + x
          sum_y = sum_y + y
          sum_xy = sum_xy + x * y
          n = n + 1
        end
      end
      if n > 1 then
        local mu_x = sum_x / n
        local mu_y = sum_y / n
        local cov = (sum_xy - n * mu_x * mu_y) / n
        local sd_x = pred_col:div()
        local sd_y = target_col:div()
        local corr = cov / (sd_x * sd_y)
        if math.abs(corr) > best_corr then
          best_corr = math.abs(corr)
          best_predictor_col = pred_col
          local slope = cov / (sd_x * sd_x)
          local intercept = mu_y - slope * mu_x
          local sum_sq_res = 0
          for _, r in ipairs(tbl.rows) do
            local xv = r[pred_col.at]
            local yv = r[target_col.at]
            if xv ~= "?" and yv ~= "?" then
              local pred = intercept + slope * xv
              local res = yv - pred
              sum_sq_res = sum_sq_res + res * res
            end
          end
          local resid_sd = math.sqrt(sum_sq_res / (n - 2))
          best_coeff = slope
          best_intercept = intercept
          best_resid_sd = resid_sd
          n_all = n
        end
      end
    end
  end

  return {
    predictor = best_predictor_col,
    slope     = best_coeff,
    intercept = best_intercept,
    resid_sd  = best_resid_sd,
    n         = n_all
  }
end

-- ------------------------------------------------------------------
-- CAR-17 : Flag records that are far from the inventory centroid
-- ------------------------------------------------------------------
local function flag_outliers(tbl)
  local center = tbl:mid()          -- centroid row
  local rows = {}
  for _, r in ipairs(tbl.rows) do
    local d = tbl:distx(r, center)
    table.insert(rows, {row=r, dist=d})
  end
  table.sort(rows, function(a,b) return a.dist > b.dist end)
  return rows
end

-- ------------------------------------------------------------------
-- CAR-18 : Flag records whose target deviates from linear prediction
-- ------------------------------------------------------------------
local function flag_prediction_errors(tbl, target_name)
  local cfg = best_predictor(tbl, target_name)
  local res = {}
  if not cfg.predictor then return res end
  local target_col = tbl.cols.all[target_name]
  for _, r in ipairs(tbl.rows) do
    local actual = r[target_col.at]
    local pred_val = r[cfg.predictor.at]
    if actual ~= "?" and pred_val ~= "?" then
      local predicted = cfg.intercept + cfg.slope * pred_val
      local diff = actual - predicted
      if math.abs(diff) > 2 * cfg.resid_sd then
        table.insert(res, {row=r, diff=diff, sd=cfg.resid_sd,
                          target=target_name,
                          predicted=predicted})
      end
    end
  end
  return res
end

-- ------------------------------------------------------------------
-- Main
-- ------------------------------------------------------------------
local csv_path = arg[1]
if not csv_path then
  io.stderr:write("Error: CSV file path required\n")
  os.exit(1)
end

local tbl = ezr.Tbl(ezr.csv(csv_path))

-- ---------- CAR-17 ----------
print("== CAR-17 ==")
local outliers = flag_outliers(tbl)
if #outliers == 0 then
  print("No outliers detected.")
else
  for i=1,math.min(5,#outliers) do
    local o = outliers[i]
    print(string.format("%2d) dist=%.4f  %s", i, o.dist, ezr.show(o.row)))
  end
  if #outliers > 5 then
    print(string.format("...and %d more outliers", #outliers-5))
  end
end

-- ---------- CAR-18 ----------
print("\n== CAR-18 ==")
local flags = {}
-- weight
for _, f in ipairs(flag_prediction_errors(tbl, "Lbs-")) do
  table.insert(flags, f)
end
-- fuel efficiency
for _, f in ipairs(flag_prediction_errors(tbl, "Mpg+")) do
  table.insert(flags, f)
end

if #flags == 0 then
  print("No records appear to have recording errors.")
else
  -- sort by magnitude of deviation (absolute)
  table.sort(flags, function(a,b) return math.abs(a.diff) > math.abs(b.diff) end)
  local printed = 0
  for _, f in ipairs(flags) do
    if printed >= 5 then break end
    local r = f.row
    print(string.format(
      "%s record: %s\n  Expected %s=%.2f, Actual=%.2f, Δ=%.2f (sd=%.2f)",
      f.target,
      ezr.show(r),
      f.target,
      f.predicted,
      r[tbl.cols.all[f.target].at],
      f.diff,
      f.sd))
    printed = printed + 1
  end
end
