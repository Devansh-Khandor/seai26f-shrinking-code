#!/usr/bin/env lua
-- Skill 02: Predict effort and defects for NASA software projects
-- Usage: lua skill_02.lua <csv_path>

local ezr = require("ezr-lib")   -- library providing csv, etc.
local csv = ezr.csv

-- ---------- Utility helpers ------------------------------------
local function map(t, f)
  local r = {}
  for _, v in ipairs(t) do r[#r+1] = f(v) end
  return r
end

local function sort_copy(t)
  local s = map(t, function(v) return v end)
  table.sort(s)
  return s
end

local function mean(t)
  local sum, n = 0, #t
  for _, v in ipairs(t) do sum = sum + v end
  return sum / n
end

local function sd(t, m)
  local sum, n = 0, #t
  m = m or mean(t)
  for _, v in ipairs(t) do sum = sum + (v - m) ^ 2 end
  return math.sqrt(sum / n)
end

local function median(t)
  local s = sort_copy(t)
  local n = #s
  if n % 2 == 1 then return s[(n+1)/2] end
  return (s[n/2] + s[n/2 + 1]) / 2
end

-- ---------- Regression helpers ---------------------------------
local function linear_regression(x, y)
  local n = #x
  local mx, my = mean(x), mean(y)
  local cov, var = 0, 0
  for i = 1, n do
    local dx, dy = x[i] - mx, y[i] - my
    cov = cov + dx * dy
    var = var + dx * dx
  end
  local slope = cov / var
  local intercept = my - slope * mx
  -- residual standard error
  local res = 0
  for i = 1, n do
    local pred = slope * x[i] + intercept
    res = res + (y[i] - pred) ^ 2
  end
  res = math.sqrt(res / (n - 2))
  return slope, intercept, res
end

-- ---------- Read CSV ---------------------------------------------
local path = arg[1]
if not path then
  io.stderr:write("Usage: lua skill_02.lua <csv_path>\n")
  os.exit(1)
end

local rows = {}
local header, idx = nil, {}

for rec in csv(path) do
  if not header then
    header = rec
    for i, name in ipairs(rec) do idx[name] = i end
  else
    local r = {}
    -- parse numeric columns
    r.Kloc   = tonumber(rec[idx["Kloc+"]])
    r.Effort = tonumber(rec[idx["Effort-"]])
    r.Defects = tonumber(rec[idx["Defects-"]])
    if r.Kloc and r.Effort and r.Defects then
      rows[#rows+1] = r
    end
  end
end

-- ---------- Compute stats ---------------------------------------
local Klocs   = map(rows, function(r) return r.Kloc end)
local Efforts = map(rows, function(r) return r.Effort end)
local Defects = map(rows, function(r) return r.Defects end)

local eff_mean = mean(Efforts)
local eff_sd   = sd(Efforts)
local def_mean = mean(Defects)
local def_sd   = sd(Defects)

-- regression on Kloc
local slope_e, intercept_e, se_e = linear_regression(Klocs, Efforts)
local slope_d, intercept_d, se_d = linear_regression(Klocs, Defects)

local mean_kloc = mean(Klocs)

local pred_eff = slope_e * mean_kloc + intercept_e
local pred_def = slope_d * mean_kloc + intercept_d

-- ---------- Output ------------------------------------------------
print("== NASA-03 ==")
print(string.format(
  "Effort estimate (based on average KLOC): %.2f ± %.2f (95%% CI)",
  pred_eff, 2 * se_e))
print(string.format(
  "Defects estimate (based on average KLOC): %.2f ± %.2f (95%% CI)",
  pred_def, 2 * se_d))
print()
print("== NASA-04 ==")
print(string.format(
  "Mean Effort: %.2f, SD: %.2f", eff_mean, eff_sd))
print(string.format(
  "Mean Defects: %.2f, SD: %.2f", def_mean, def_sd))
