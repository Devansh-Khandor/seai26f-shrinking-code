#!/usr/bin/env lua
--[[
  skill_08.lua
  NASA-93 data analysis – advice for reducing effort or defects
  in a project proposal.
  Usage:  lua skill_08.lua <csv-path>
  Prints:
    NASA-15: suggestions to lower predicted effort.
    NASA-16: smallest set of changes to improve predicted quality.
]]--

local ezr = require "ezr"

-- ------------------------------------------------------------------
-- Helpers
-- ------------------------------------------------------------------
local function abort(msg)
  io.stderr:write(msg .. "\n")
  os.exit(1)
end

local function round(v)
  return ezr.round(v)
end

-- Simple linear regression (least squares)
local function linreg(x, y)
  local n = #x
  if n < 2 then abort("Not enough points for regression") end
  local sx, sy, sx2, sy2, sxy = 0, 0, 0, 0, 0
  for i = 1, n do
    local xi, yi = x[i], y[i]
    sx  = sx + xi
    sy  = sy + yi
    sx2 = sx2 + xi * xi
    sy2 = sy2 + yi * yi
    sxy = sxy + xi * yi
  end
  local mean_x = sx / n
  local mean_y = sy / n
  local var_x  = sx2 / n - mean_x * mean_x
  local var_y  = sy2 / n - mean_y * mean_y
  local cov_xy = sxy / n - mean_x * mean_y
  local b = cov_xy / var_x
  local a = mean_y - b * mean_x
  -- residual standard deviation
  local ss_res = 0
  for i = 1, n do
    local yi_hat = a + b * x[i]
    ss_res = ss_res + (y[i] - yi_hat) ^ 2
  end
  local residual_std = math.sqrt(ss_res / (n - 2))
  return {a=a,b=b,mean_x=mean_x,mean_y=mean_y,
          var_x=var_x,var_y=var_y,residual_std=residual_std}
end

-- Median of numeric array
local function median(vals)
  if #vals == 0 then return 0 end
  table.sort(vals)
  local mid = math.floor((#vals + 1) / 2)
  return vals[mid]
end

-- Compute median target per category for a symbolic column
local function category_medians(rows, col, target_col)
  local cat_vals = {}
  for _, r in ipairs(rows) do
    local cat = r[col.at]
    local val = r[target_col.at]
    if cat ~= "?" and val ~= "?" then
      cat_vals[cat] = cat_vals[cat] or {}
      table.insert(cat_vals[cat], val)
    end
  end
  local med = {}
  for k, v in pairs(cat_vals) do
    med[k] = median(v)
  end
  return med
end

-- Find best (lowest median) category for each column
local function best_category(med)
  local best = nil
  for cat, val in pairs(med) do
    if best == nil or val < best then
      best = val
      best_cat = cat
    end
  end
  return best_cat, best
end

-- ------------------------------------------------------------------
-- Main
-- ------------------------------------------------------------------
local csv_path = arg[1]
if not csv_path then abort("Usage: lua skill_08.lua <csv-path>") end

-- Load rows
local rows = {}
for r in ezr.csv(csv_path) do table.insert(rows, r) end
if #rows == 0 then abort("No data found in " .. csv_path) end

-- Build table structure
local tbl = ezr.Tbl(rows)

-- Map columns by name
local col_map = {}
for i,c in ipairs(tbl.cols.all) do
  col_map[tbl.cols.names[i]] = c
end

-- Regression models
local x_eff, y_eff = {}, {}
local x_def, y_def = {}, {}
for _, r in ipairs(rows) do
  local k = r[col_map["Kloc+"].at]
  local e = r[col_map["Effort-"].at]
  local d = r[col_map["Defects-"].at]
  if k ~= "?" and e ~= "?" then
    table.insert(x_eff, k); table.insert(y_eff, e)
  end
  if k ~= "?" and d ~= "?" then
    table.insert(x_def, k); table.insert(y_def, d)
  end
end

local eff_model = linreg(x_eff, y_eff)
local def_model = linreg(x_def, y_def)

-- Compute predictions
local pred_eff, pred_def = {}, {}
for _, r in ipairs(rows) do
  local k = r[col_map["Kloc+"].at]
  if k ~= "?" then
    local pe = eff_model.a + eff_model.b * k
    local pd = def_model.a + def_model.b * k
    table.insert(pred_eff, pe)
    table.insert(pred_def, pd)
  else
    table.insert(pred_eff, nil)
    table.insert(pred_def, nil)
  end
end

-- Find row with max predicted effort
local max_eff_idx = 1
for i, val in ipairs(pred_eff) do
  if val and (not pred_eff[max_eff_idx] or val > pred_eff[max_eff_idx]) then
    max_eff_idx = i
  end
end
local max_eff_row = rows[max_eff_idx]
local max_eff_pred = pred_eff[max_eff_idx]

-- Find row with max predicted defects
local max_def_idx = 1
for i, val in ipairs(pred_def) do
  if val and (not pred_def[max_def_idx] or val > pred_def[max_def_idx]) then
    max_def_idx = i
  end
end
local max_def_row = rows[max_def_idx]
local max_def_pred = pred_def[max_def_idx]

-- Identify adjustable categorical columns (non-numeric, not target, not id)
local adjust_cols = {}
local skip_names = {
  ["idX"]=true, ["Effort-"]=true, ["Defects-"]=true,
  ["Kloc+"]=true, ["YearX"]=true, ["MonthsX"]=true
}
for _,c in ipairs(tbl.cols.all) do
  if not skip_names[c.name] then
    local first = rows[1][c.at]
    if type(first) ~= "number" then
      table.insert(adjust_cols, c)
    end
  end
end

-- Precompute medians for each column
local col_medians = {}
for _,c in ipairs(adjust_cols) do
  col_medians[c.name] = category_medians(rows, c, col_map["Effort-"])
end

-- Compute best medians for effort
local effort_best_cat = {}
for _,c in ipairs(adjust_cols) do
  local best_cat, best_med = best_category(col_medians[c.name])
  effort_best_cat[c.name] = {cat=best_cat, med=best_med}
end

-- For NASA-15: find top 3 plausible changes to reduce effort
local effort_improvements = {}
for _,c in ipairs(adjust_cols) do
  local curr_val = max_eff_row[c.at]
  local med_dict = col_medians[c.name]
  local curr_med = med_dict[curr_val]
  local best_med = effort_best_cat[c.name].med
  local improv = curr_med and (curr_med - best_med) or 0
  if improv > 0 then
    table.insert(effort_improvements, {
      col=c.name,
      from=curr_val,
      to=effort_best_cat[c.name].cat,
      reduce=improv
    })
  end
end
table.sort(effort_improvements, function(a,b) return a.reduce > b.reduce end)

-- For NASA-16: best single change to reduce defects
-- Recompute medians for defects
local defect_medians = {}
for _,c in ipairs(adjust_cols) do
  defect_medians[c.name] = category_medians(rows, c, col_map["Defects-"])
end
local defect_best_cat = {}
for _,c in ipairs(adjust_cols) do
  local best_cat, best_med = best_category(defect_medians[c.name])
  defect_best_cat[c.name] = {cat=best_cat, med=best_med}
end

local defect_improvements = {}
for _,c in ipairs(adjust_cols) do
  local curr_val = max_def_row[c.at]
  local med_dict = defect_medians[c.name]
  local curr_med = med_dict[curr_val]
  local best_med = defect_best_cat[c.name].med
  local improv = curr_med and (curr_med - best_med) or 0
  if improv > 0 then
    table.insert(defect_improvements, {
      col=c.name,
      from=curr_val,
      to=defect_best_cat[c.name].cat,
      reduce=improv
    })
  end
end
table.sort(defect_improvements, function(a,b) return a.reduce > b.reduce end)

-- ------------------------------------------------------------------
-- Output
-- ------------------------------------------------------------------
print("== NASA-15 ==")
print(string.format("Project %s with predicted effort %.1f is the highest.",
       tostring(max_eff_row[1]), round(max_eff_pred)))
print("Top 3 adjustments that could reduce its predicted effort:")
for i,chg in ipairs(effort_improvements) do
  if i > 3 then break end
  print(string.format(
    "%d. Change %s from '%s' to '%s' could lower effort by %.2f units.",
    i, chg.col, chg.from, chg.to, round(chg.reduce)))
end
if #effort_improvements == 0 then
  print("No actionable adjustments found that would lower effort.")
end

print("\n== NASA-16 ==")
print(string.format("Project %s with predicted defects %.1f is the highest.",
       tostring(max_def_row[1]), round(max_def_pred)))
print("Best single change to improve predicted quality:")
if #defect_improvements > 0 then
  local chg = defect_improvements[1]
  print(string.format(
    "Change %s from '%s' to '%s' could reduce predicted defects by %.2f units.",
    chg.col, chg.from, chg.to, round(chg.reduce)))
  print("Trade‑off: The change may require additional training, tooling, or process adjustments, but the potential quality gain can justify the investment.")
else
  print("No single categorical change is expected to improve defect prediction.")
end
