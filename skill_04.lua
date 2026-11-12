#!/usr/bin/env lua
-- Skill 04: NASA-07 & NASA-08 – Real‑time proposal assessment and incremental summary update
-- Usage: lua skill_04.lua <csv_path>

local ezr = require("ezr-lib")
local csv  = ezr.csv
local show = ezr.show
local round = ezr.round

--------------------------------------------------------------------
-- Helper: simple linear regression (slope, intercept, residual sd)
--------------------------------------------------------------------
local function linear_regression(x, y)
  local n = #x
  local mx, my = 0, 0
  for i=1,n do mx = mx + x[i] end
  for i=1,n do my = my + y[i] end
  mx = mx / n
  my = my / n

  local cov, varx = 0, 0
  for i=1,n do
    local dx, dy = x[i]-mx, y[i]-my
    cov = cov + dx*dy
    varx = varx + dx*dx
  end

  local slope = cov / varx
  local intercept = my - slope*mx

  local se = 0
  for i=1,n do
    local pred = slope*x[i] + intercept
    se = se + (y[i]-pred)^2
  end
  se = math.sqrt(se / (n-2))  -- residual SD

  return slope, intercept, se
end

--------------------------------------------------------------------
-- Statistics helper
--------------------------------------------------------------------
local function mean(t)
  if #t==0 then return 0 end
  local sum=0
  for _,v in ipairs(t) do sum=sum+v end
  return sum/#t
end

local function sd(t, m)
  if #t==0 then return 0 end
  m = m or mean(t)
  local s=0
  for _,v in ipairs(t) do s=s+(v-m)^2 end
  return math.sqrt(s/#t)
end

--------------------------------------------------------------------
-- Main routine
--------------------------------------------------------------------
local function main()
  local path = arg[1]
  if not path then
    io.stderr:write("Usage: lua skill_04.lua <csv_path>\n")
    os.exit(1)
  end

  -- read csv
  local rows = {}
  local header = nil
  local idx = {}
  for rec in csv(path) do
    if not header then
      header = rec
      for i, name in ipairs(rec) do idx[name] = i end
    else
      -- store a minimal copy
      local r = {
        id = tostring(rec[idx["idX"]]),
        Kloc = tonumber(rec[idx["Kloc+"]]),
        Effort = tonumber(rec[idx["Effort-"]]),
        Defects = tonumber(rec[idx["Defects-"]])
      }
      table.insert(rows, r)
    end
  end

  -- split into completed (both effort & defects) and proposals
  local completed = {}
  local proposals = {}
  for _,r in ipairs(rows) do
    if r.Effort and r.Defects and r.Kloc then
      table.insert(completed, r)
    else
      table.insert(proposals, r)
    end
  end

  ----------------------------------------------------------------
  -- Build regression models on completed data
  ----------------------------------------------------------------
  local slope_e, intercept_e, se_e = 0,0,0
  local slope_d, intercept_d, se_d = 0,0,0
  if #completed>2 then
    local kx, y_e, y_d = {}, {}, {}
    for _,r in ipairs(completed) do
      table.insert(kx, r.Kloc)
      table.insert(y_e, r.Effort)
      table.insert(y_d, r.Defects)
    end
    slope_e, intercept_e, se_e = linear_regression(kx, y_e)
    slope_d, intercept_d, se_d = linear_regression(kx, y_d)
  end

  ----------------------------------------------------------------
  -- NASA-07: assess proposals
  ----------------------------------------------------------------
  print("== NASA-07 ==")
  if #proposals==0 then
    print("No proposal records to assess.")
  else
    -- compute predictions for all proposals first
    local pred_eff, pred_def = {}, {}
    for _,p in ipairs(proposals) do
      if p.Kloc then
        local pe = slope_e * p.Kloc + intercept_e
        local pd = slope_d * p.Kloc + intercept_d
        table.insert(pred_eff, pe)
        table.insert(pred_def, pd)
      end
    end

    local meff = mean(pred_eff)
    local seff = sd(pred_eff, meff)
    local mdef = mean(pred_def)
    local sdef = sd(pred_def, mdef)

    local alert_count = 0
    for i,p in ipairs(proposals) do
      if p.Kloc then
        local pe = slope_e * p.Kloc + intercept_e
        local pd = slope_d * p.Kloc + intercept_d
        local alert = (pe > meff+seff) or (pd > mdef+sdef)
        if alert then alert_count = alert_count + 1 end
        print(string.format(
          "Proposal ID=%s: Pred Effort=%s, Pred Defects=%s, Alert=%s",
          p.id, round(pe,2), round(pd,2), alert and "YES" or "NO"))
      end
    end
    print(string.format("Total proposals: %d, Alerts: %d", #proposals, alert_count))
  end

  ----------------------------------------------------------------
  -- NASA-08: update and report historical summaries
  ----------------------------------------------------------------
  print("\n== NASA-08 ==")
  if #completed==0 then
    print("No completed projects in dataset; cannot compute summaries.")
  else
    -- collect statistics
    local kloc_vals, effort_vals, defect_vals = {}, {}, {}
    for _,r in ipairs(completed) do
      table.insert(kloc_vals, r.Kloc)
      table.insert(effort_vals, r.Effort)
      table.insert(defect_vals, r.Defects)
    end

    local mean_kloc = mean(kloc_vals)
    local sd_kloc = sd(kloc_vals, mean_kloc)

    local mean_effort = mean(effort_vals)
    local sd_effort = sd(effort_vals, mean_effort)

    local mean_defects = mean(defect_vals)
    local sd_defects = sd(defect_vals, mean_defects)

    print(string.format("Completed projects: %d", #completed))
    print(string.format("Mean Kloc+: %.2f (SD=%.2f)", mean_kloc, sd_kloc))
    print(string.format("Mean Effort-: %.2f (SD=%.2f)", mean_effort, sd_effort))
    print(string.format("Mean Defects-: %.2f (SD=%.2f)", mean_defects, sd_defects))

    print(string.format("Regression Effort: slope=%.4f, intercept=%.4f (residual SD=%.4f)",
      slope_e, intercept_e, se_e))
    print(string.format("Regression Defects: slope=%.4f, intercept=%.4f (residual SD=%.4f)",
      slope_d, intercept_d, se_d))
  end
end

main()
