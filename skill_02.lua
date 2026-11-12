#!/usr/bin/env lua
-- Skill 02 – Fuel‑efficiency and weight estimation
-- Usage:  lua skill_02.lua <csv path>

local ezr = require "ezr"

-- ------------------------------------------------------------------
-- Helper: compute best linear predictor for a target column
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

  local n_all = 0 -- total rows considered for this target
  -- First, compute covariance with each numeric predictor
  for _, c in ipairs(tbl.cols.x) do
    local pred_col = tbl.cols.all[c.at]
    if pred_col ~= target_col then
      local n = 0
      local sum_xy = 0
      local sum_x = 0
      local sum_y = 0
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

          -- slope and intercept
          local slope = cov / (sd_x * sd_x)
          local intercept = mu_y - slope * mu_x

          -- compute residual standard deviation
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
-- Estimation and display
-- ------------------------------------------------------------------
local function estimate_and_show(tbl, target_name)
  local cfg = best_predictor(tbl, target_name)
  if not cfg.predictor then
    print("  No suitable predictor found for " .. target_name)
    return
  end

  local pred_col = cfg.predictor
  local slope = cfg.slope
  local intercept = cfg.intercept
  local resid_sd = cfg.resid_sd

  print("  Predictor: " .. pred_col.name .. " (corr ≈ "
        .. string.format("%.2f", cfg.n and cfg.n > 0 and
               (slope * pred_col:div()) / cfg.resid_sd or 0) .. ")")
  print("  Residual SD: " .. string.format("%.3f", resid_sd))
  print()

  local count = 0
  for _, r in ipairs(tbl.rows) do
    local val = r[tbl.cols.all[target_name].at]
    if val == "?" then
      local x = r[pred_col.at]
      if x ~= "?" then
        local est = intercept + slope * x
        print(string.format("    Row %3d: %s -> est = %.3f ± %.3f",
                  count + 1, ezr.show(r), est, resid_sd))
        count = count + 1
        if count >= 5 then break end
      end
    end
  end
  if count == 0 then
    print("    No missing values found for " .. target_name)
  end
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

-- CAR-03: Estimate missing fuel‑efficiency (Mpg+)
print("== CAR-03 ==")
estimate_and_show(tbl, "Mpg+")

-- CAR-04: Estimate missing weight (Lbs-)
print("\n== CAR-04 ==")
estimate_and_show(tbl, "Lbs-")
