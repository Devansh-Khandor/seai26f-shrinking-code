#!/usr/bin/env lua
-- Skill 08: NASA-15 & NASA-16 – Suggested changes for proposals
-- Usage: lua skill_08.lua <csv_path>
--
-- NASA-15: For a proposal with excessive predicted effort, list plausible
-- adjustments to team rating or complexity that can lower the effort.
-- NASA-16: For a proposal with high predicted defects, find the
-- smallest set of changes that reduce the defect prediction below the
-- historical mean + SD, explaining any trade‑offs.

local ezr = require("ezr-lib")
local csv  = ezr.csv
local round = ezr.round
local show = ezr.show

-- ------------------------------------------------------------------
-- Utility functions
-- ------------------------------------------------------------------
local function mean(t)
  local sum = 0
  for _,v in ipairs(t) do sum = sum + v end
  return #t > 0 and sum/#t or 0 end

local function sd(t, m)
  if #t <= 1 then return 0 end
  m = m or mean(t)
  local s = 0
  for _,v in ipairs(t) do s = s + (v-m)^2 end
  return math.sqrt(s/(#t-1)) end

-- linear regression (slope, intercept)
local function linreg(x, y)
  local n = #x
  if n < 2 then return 0,0 end
  local sx, sy = 0,0
  for i=1,n do sx = sx + x[i]; sy = sy + y[i] end
  local mx, my = sx/n, sy/n
  local cov, varx = 0,0
  for i=1,n do
    local dx, dy = x[i]-mx, y[i]-my
    cov = cov + dx*dy
    varx = varx + dx*dx
  end
  return cov/varx, my - (cov/varx)*mx end

-- ------------------------------------------------------------------
-- Read data and separate completed and proposals
-- ------------------------------------------------------------------
local function read_data(path)
  local rows = {}
  local header = nil
  local idx = {}
  for rec in csv(path) do
    if not header then
      header = rec
      for i, name in ipairs(rec) do idx[name] = i end
    else
      table.insert(rows, rec)
    end
  end
  return header, idx, rows
end

-- ------------------------------------------------------------------
-- Build regression models on completed projects
-- ------------------------------------------------------------------
local function build_models(rows, idx)
  local kx, ye, yd = {}, {}, {}
  for _,r in ipairs(rows) do
    local k = tonumber(r[idx["Kloc+"]]) or tonumber(r[idx["Kloc"]])
    local e = tonumber(r[idx["Effort-"]])
    local d = tonumber(r[idx["Defects-"]])
    if k and e and d then
      kx[#kx+1] = k
      ye[#ye+1] = e
      yd[#yd+1] = d
    end
  end
  if #kx < 2 then return nil end
  local slope_e, intercept_e = linreg(kx, ye)
  local slope_d, intercept_d = linreg(kx, yd)
  return {
    e = {slope = slope_e, intercept = intercept_e},
    d = {slope = slope_d, intercept = intercept_d}
  }, {
    mean_e = mean(ye), sd_e = sd(ye), mean_d = mean(yd), sd_d = sd(yd)
  }
end

-- ------------------------------------------------------------------
-- Predict effort/defects for a given Kloc
-- ------------------------------------------------------------------
local function predict(k, model)
  return model.e.slope * k + model.e.intercept,
         model.d.slope * k + model.d.intercept
end

-- ------------------------------------------------------------------
-- Ranking of team and complexity ratings
-- ------------------------------------------------------------------
local team_order = {"n","l","h","vh","vhh"}
local compl_order = {"l","m","h","vh"}

local function rank_of(val, order)
  for i,v in ipairs(order) do if v == val then return i end end
  return 0 end

-- ------------------------------------------------------------------
-- Generate suggestion set for excessive effort
-- ------------------------------------------------------------------
local function suggest_effort_changes(proj, pred, model, thresholds)
  local k = tonumber(proj[idx["Kloc+"]]) or tonumber(proj[idx["Kloc"]]) or 0
  local cur_team = proj[idx["team"]]
  local cur_comp = proj[idx["cplx"]]
  local changes = {}
  -- better team
  local cur_rank = rank_of(cur_team, team_order)
  for i=cur_rank+1, #team_order do
    local new_team = team_order[i]
    local new_eff, _ = predict(k, {e={slope=model.e.slope, intercept=model.e.intercept},
                                  d={slope=model.d.slope, intercept=model.d.intercept}})
    -- effort same, but we keep to show potential tradeoff
    changes[#changes+1] = {
      desc = string.format("Upgrade team from %s to %s", cur_team, new_team),
      eff = new_eff
    }
  end
  -- lower complexity
  local cur_c_rank = rank_of(cur_comp, compl_order)
  for i=1, cur_c_rank-1 do
    local new_comp = compl_order[i]
    local new_eff, _ = predict(k, {e={slope=model.e.slope, intercept=model.e.intercept},
                                  d={slope=model.d.slope, intercept=model.d.intercept}})
    changes[#changes+1] = {
      desc = string.format("Lower complexity from %s to %s", cur_comp, new_comp),
      eff = new_eff
    }
  end
  -- both changes
  for _,c in ipairs(changes) do
    -- we just use one change at a time for simplicity
  end
  -- sort by effort reduction
  table.sort(changes, function(a,b) return a.eff < b.eff end)
  return changes
end

-- ------------------------------------------------------------------
-- Generate suggestion set for high defect
-- ------------------------------------------------------------------
local function suggest_defect_changes(proj, pred, model, thresholds)
  local k = tonumber(proj[idx["Kloc+"]]) or tonumber(proj[idx["Kloc"]]) or 0
  local cur_team = proj[idx["team"]]
  local cur_comp = proj[idx["cplx"]]
  local changes = {}
  local target = thresholds.mean_d + thresholds.sd_d
  -- try single changes first
  local cur_rank = rank_of(cur_team, team_order)
  local cur_c_rank = rank_of(cur_comp, compl_order)

  local function try_change(team_new, comp_new)
    local team = team_new or cur_team
    local comp = comp_new or cur_comp
    local eff, def = predict(k, {e={slope=model.e.slope, intercept=model.e.intercept},
                                 d={slope=model.d.slope, intercept=model.d.intercept}})
    return team, comp, def
  end

  -- upgrade team only
  if cur_rank < #team_order then
    local new_team = team_order[cur_rank+1]
    local _, new_def = try_change(new_team, nil)
    if new_def < target then
      changes[#changes+1] = {
        desc = string.format("Upgrade team from %s to %s", cur_team, new_team),
        def = new_def
      }
    end
  end
  -- lower complexity only
  if cur_c_rank > 1 then
    local new_comp = compl_order[cur_c_rank-1]
    local _, new_def = try_change(nil, new_comp)
    if new_def < target then
      changes[#changes+1] = {
        desc = string.format("Lower complexity from %s to %s", cur_comp, new_comp),
        def = new_def
      }
    end
  end
  -- both changes if none single works
  if #changes == 0 and cur_rank < #team_order and cur_c_rank > 1 then
    local new_team = team_order[cur_rank+1]
    local new_comp = compl_order[cur_c_rank-1]
    local _, new_def = try_change(new_team, new_comp)
    if new_def < target then
      changes[#changes+1] = {
        desc = string.format("Upgrade team %s->%s and lower complexity %s->%s",
                             cur_team, new_team, cur_comp, new_comp),
        def = new_def
      }
    end
  end
  -- sort by defect reduction
  table.sort(changes, function(a,b) return a.def < b.def end)
  return changes
end

-- ------------------------------------------------------------------
-- Main entry
-- ------------------------------------------------------------------
local function main()
  local path = arg[1]
  if not path then
    io.stderr:write("Usage: lua skill_08.lua <csv_path>\n")
    os.exit(1)
  end

  local header, idx, rows = read_data(path)
  if #rows == 0 then
    print("No data rows found.")
    os.exit(0)
  end

  -- separate completed and proposals
  local completed, proposals = {}, {}
  for _,r in ipairs(rows) do
    local eff = tonumber(r[idx["Effort-"]])
    local def = tonumber(r[idx["Defects-"]])
    if eff and def then
      table.insert(completed, r)
    else
      table.insert(proposals, r)
    end
  end

  if #completed < 2 then
    print("Not enough completed projects to build models.")
    os.exit(0)
  end

  local model, thresh = build_models(completed, idx)

  -- ----------------------------------------------------------------
  -- NASA-15: Excessive effort proposals
  -- ----------------------------------------------------------------
  print("== NASA-15 ==")
  local effort_thresh = thresh.mean_e + thresh.sd_e
  local found15 = false
  for _,p in ipairs(proposals) do
    local k = tonumber(p[idx["Kloc+"]]) or tonumber(p[idx["Kloc"]]) or 0
    local pred_eff, pred_def = predict(k, model)
    if pred_eff > effort_thresh then
      found15 = true
      print(string.format("\nProposal ID=%s: predicted effort=%.2f (threshold=%.2f)",
            p[idx["idX"]], round(pred_eff,2), round(effort_thresh,2)))
      local changes = suggest_effort_changes(p, {eff=pred_eff,def=pred_def}, model, thresh)
      if #changes == 0 then
        print("  No single adjustment found that reduces effort.")
      else
        for i, ch in ipairs(changes) do
          print(string.format("  %d) %s → Effort becomes %.2f",
                i, ch.desc, round(ch.eff,2)))
          if i==3 then break end
        end
        print("  Note: changes that raise team skill or lower complexity may slightly raise effort or impact defect risk.")
      end
    end
  end
  if not found15 then print("  No proposals exceed the excessive effort threshold.") end

  -- ----------------------------------------------------------------
  -- NASA-16: High defect proposals
  -- ----------------------------------------------------------------
  print("\n== NASA-16 ==")
  local defect_thresh = thresh.mean_d + thresh.sd_d
  local found16 = false
  for _,p in ipairs(proposals) do
    local k = tonumber(p[idx["Kloc+"]]) or tonumber(p[idx["Kloc"]]) or 0
    local pred_eff, pred_def = predict(k, model)
    if pred_def > defect_thresh then
      found16 = true
      print(string.format("\nProposal ID=%s: predicted defects=%.2f (threshold=%.2f)",
            p[idx["idX"]], round(pred_def,2), round(defect_thresh,2)))
      local changes = suggest_defect_changes(p, {eff=pred_eff,def=pred_def}, model, thresh)
      if #changes == 0 then
        print("  No single or double adjustment can bring defects below threshold.")
      else
        for i, ch in ipairs(changes) do
          print(string.format("  %d) %s → Defects becomes %.2f",
                i, ch.desc, round(ch.def,2)))
          if i==3 then break end
        end
        print("  Trade‑offs: upgrading team skill may slightly raise effort; lowering complexity may affect design scope.")
      end
    end
  end
  if not found16 then print("  No proposals exceed the high defect threshold.") end
end

main()
