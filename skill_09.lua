#!/usr/bin/env lua
--[[
  skill_09.lua
  NASA-93 data analysis – outlier detection.
  Usage:  lua skill_09.lua <csv-path>
  Prints:
    NASA-17: projects whose characteristic combination is unusually distant from the rest.
    NASA-18: projects whose effort or defects deviate substantially from regression predictions.
]]--

local ezr = require "ezr"

-- ------------------------------------------------------------------
-- helpers
-- ------------------------------------------------------------------
local function abort(msg)
  io.stderr:write(msg .. "\n")
  os.exit(1)
end

local function round(v)
  return ezr.round(v)
end

-- mean & std for a numeric list
local function mean_std(vals)
  local n = #vals
  if n == 0 then return 0,0 end
  local sum = 0
  for _,v in ipairs(vals) do sum = sum + v end
  local mean = sum / n
  local var = 0
  for _,v in ipairs(vals) do
    var = var + (v - mean)*(v - mean)
  end
  var = var/(n-1)
  return mean, math.sqrt(var)
end

-- simple linear regression (returns {a,b,sd})
local function linreg(x, y)
  local n = #x
  local sx, sx2, sy, sy2, sxy = 0,0,0,0,0
  for i=1,n do
    local xi, yi = x[i], y[i]
    sx  = sx + xi
    sx2 = sx2 + xi*xi
    sy  = sy + yi
    sy2 = sy2 + yi*yi
    sxy = sxy + xi*yi
  end
  local mean_x = sx/n
  local mean_y = sy/n
  local var_x  = sx2/n - mean_x*mean_x
  local var_y  = sy2/n - mean_y*mean_y
  local cov_xy = sxy/n - mean_x*mean_y
  local b = cov_xy / var_x
  local a = mean_y - b*mean_x
  -- residual sd
  local ss_res = 0
  for i=1,n do
    local yhat = a + b*x[i]
    ss_res = ss_res + (y[i]-yhat)*(y[i]-yhat)
  end
  local residual_sd = math.sqrt(ss_res/(n-2))
  return {a=a,b=b,sd=residual_sd}
end

-- ------------------------------------------------------------------
-- main
-- ------------------------------------------------------------------
local csv_path = arg[1]
if not csv_path then abort("Usage: lua skill_09.lua <csv-path>") end

-- load rows
local rows = {}
for r in ezr.csv(csv_path) do table.insert(rows, r) end
if #rows == 0 then abort("No data found in "..csv_path) end

-- build table structure
local tbl = ezr.Tbl(rows)

-- map column names
local col_map = {}
for i,c in ipairs(tbl.cols.all) do
  col_map[tbl.cols.names[i]] = c
end

-- ------------------------------------------------------------------
-- NASA-17: unusual combination of characteristics
-- ------------------------------------------------------------------
print("== NASA-17 ==")

-- use all independent columns as distance metric
-- centroid row
local mid_row = tbl:mids()

-- compute distances to centroid
local dists = {}
for _,r in ipairs(rows) do
  table.insert(dists, tbl:distx(r, mid_row))
end

local mean_d, sd_d = mean_std(dists)

-- threshold: mean + 2*sd; ensure at least 3 flagged
local thresh = mean_d + 2*sd_d
-- if too few flagged, lower threshold to mean + sd
local flagged = {}
for i,r in ipairs(rows) do
  local d = dists[i]
  if d > thresh then table.insert(flagged, {row=r, dist=d}) end
end
if #flagged < 3 then
  thresh = mean_d + sd_d
  flagged = {}
  for i,r in ipairs(rows) do
    local d = dists[i]
    if d > thresh then table.insert(flagged, {row=r, dist=d}) end
  end
end
-- if still <3, take top 3 by distance
if #flagged < 3 then
  table.sort(dists, function(a,b) return a>b end) -- but need mapping
  table.sort(rows, function(a,b) return tbl:distx(a, mid_row) > tbl:distx(b, mid_row) end)
  flagged = {}
  for i=1,math.min(3,#rows) do
    table.insert(flagged, {row=rows[i], dist=tbl:distx(rows[i], mid_row)})
  end
end

for _,f in ipairs(flagged) do
  local r = f.row
  local id = r[1]
  print(string.format("  ID=%s, distance=%.3f", tostring(id), round(f.dist)))
end

-- ------------------------------------------------------------------
-- NASA-18: projects with effort or defects deviating from predictions
-- ------------------------------------------------------------------
print("\n== NASA-18 ==")

-- regression models
local k_col = col_map["Kloc+"]
local eff_col = col_map["Effort-"]
local def_col = col_map["Defects-"]

local k_vals, eff_vals, def_vals = {}, {}, {}
for _,r in ipairs(rows) do
  local k = r[k_col.at]
  local e = r[eff_col.at]
  local d = r[def_col.at]
  if k~="?" and e~="?" then table.insert(k_vals,k); table.insert(eff_vals,e) end
  if k~="?" and d~="?" then table.insert(k_vals,k); table.insert(def_vals,d) end
end

local eff_model = linreg(k_vals, eff_vals)
local def_model = linreg(k_vals, def_vals)

-- compute residuals
local eff_resid, def_resid = {}, {}
for _,r in ipairs(rows) do
  local k = r[k_col.at]
  local e = r[eff_col.at]
  local d = r[def_col.at]
  if k~="?" and e~="?" then
    local pred = eff_model.a + eff_model.b*k
    table.insert(eff_resid, e - pred)
  else
    table.insert(eff_resid, nil)
  end
  if k~="?" and d~="?" then
    local pred = def_model.a + def_model.b*k
    table.insert(def_resid, d - pred)
  else
    table.insert(def_resid, nil)
  end
end

local mean_eff_res, sd_eff_res = mean_std(eff_resid)
local mean_def_res, sd_def_res = mean_std(def_resid)

-- thresholds on absolute residuals
local thresh_eff = mean_eff_res + 2*sd_eff_res
local thresh_def = mean_def_res + 2*sd_def_res

local flagged2 = {}
for i,r in ipairs(rows) do
  local k = r[k_col.at]
  local e = r[eff_col.at]
  local d = r[def_col.at]
  if k~="?" and e~"?" then
    local resid = e - (eff_model.a + eff_model.b*k)
    if math.abs(resid) > thresh_eff then
      table.insert(flagged2, {row=r, eff_res=resid})
      goto continue
    end
  end
  if k~="?" and d~"?" then
    local resid = d - (def_model.a + def_model.b*k)
    if math.abs(resid) > thresh_def then
      table.insert(flagged2, {row=r, def_res=resid})
    end
  end
  ::continue::
end

-- if fewer than 3 flagged, take top 3 largest abs residuals
if #flagged2 < 3 then
  local all_res = {}
  for i,r in ipairs(rows) do
    local k = r[k_col.at]
    if k~="?" then
      local e = r[eff_col.at]
      if e~="?" then
        local resid = e - (eff_model.a + eff_model.b*k)
        table.insert(all_res, {row=r, eff_res=resid})
      end
      local d = r[def_col.at]
      if d~="?" then
        local resid = d - (def_model.a + def_model.b*k)
        table.insert(all_res, {row=r, def_res=resid})
      end
    end
  end
  table.sort(all_res, function(a,b)
    local va = a.eff_res and math.abs(a.eff_res) or 0
    local vb = b.eff_res and math.abs(b.eff_res) or 0
    return va > vb
  end)
  flagged2 = {}
  for i=1,math.min(3,#all_res) do
    table.insert(flagged2, all_res[i])
  end
end

for _,f in ipairs(flagged2) do
  local r = f.row
  local id = r[1]
  if f.eff_res then
    print(string.format("  ID=%s, effort residual=%.2f", tostring(id), round(f.eff_res)))
  else
    print(string.format("  ID=%s, defect residual=%.2f", tostring(id), round(f.def_res)))
  end
end
