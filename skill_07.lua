#!/usr/bin/env lua
-- Skill 07: NASA-13 & NASA-14 – Promising combinations & proposal prioritisation
-- Usage: lua skill_07.lua <csv_path>

local ezr = require("ezr-lib")
local csv  = ezr.csv
local show = ezr.show
local round = ezr.round

-- Helper: linear regression (slope, intercept) for y vs x
local function linreg(x, y)
  local n = #x
  if n < 2 then return 0,0 end
  local sx = 0
  local sy = 0
  for i=1,n do sx = sx + x[i]; sy = sy + y[i] end
  local mx = sx / n
  local my = sy / n
  local cov = 0
  local varx = 0
  for i=1,n do
    local dx = x[i] - mx
    local dy = y[i] - my
    cov = cov + dx*dy
    varx = varx + dx*dx
  end
  local slope = cov / varx
  local intercept = my - slope * mx
  return slope, intercept
end

-- Read all rows, separate completed and proposals
local function read_rows(path)
  local rows = {}
  local header = nil
  local idx = {}
  for rec in csv(path) do
    if not header then
      header = rec
      for i,name in ipairs(rec) do idx[name] = i end
    else
      table.insert(rows, rec)
    end
  end
  return header, idx, rows
end

-- Identify numeric columns by checking first data row
local function detect_col_types(header, rows)
  local types = {}
  for i,name in ipairs(header) do
    local val = rows[1] and rows[1][i]
    types[i] = tonumber(val) and "num" or "sym"
  end
  return types
end

-- Build regression models from completed projects
local function build_models(rows, idx, types)
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
  if #kx < 2 then return nil, nil end
  local slope_e, intercept_e = linreg(kx, ye)
  local slope_d, intercept_d = linreg(kx, yd)
  return {e={slope=slope_e, intercept=intercept_e},
          d={slope=slope_d, intercept=intercept_d}}
end

-- Predict effort/defects for a given Kloc using models
local function predict(k, models)
  return models.e.slope*k + models.e.intercept,
         models.d.slope*k + models.d.intercept
end

-- NASA-13: find promising combinations of symbolic attributes
local function nasa13(rows, idx, types)
  -- Determine which columns are symbolic
  local syms = {}
  for i,t in ipairs(types) do
    if t == "sym" then table.insert(syms, i) end
  end
  -- Build groups
  local groups = {}
  for _,r in ipairs(rows) do
    local key = {}
    for _,i in ipairs(syms) do key[#key+1] = tostring(r[i] or "nil") end
    local k = table.concat(key, "|")
    groups[k] = groups[k] or {cnt=0, kl=0, e=0, d=0, vals={}}
    groups[k].cnt = groups[k].cnt + 1
    local kloc = tonumber(r[idx["Kloc+"]]) or tonumber(r[idx["Kloc"]]) or 0
    local eff  = tonumber(r[idx["Effort-"]]) or 0
    local def  = tonumber(r[idx["Defects-"]]) or 0
    groups[k].kl = groups[k].kl + kloc
    groups[k].e  = groups[k].e  + eff
    groups[k].d  = groups[k].d  + def
    groups[k].vals[#groups[k].vals+1] = r
  end
  -- Compute averages and score
  local combos = {}
  for k,g in pairs(groups) do
    local n = g.cnt
    local avgkl = g.kl / n
    local avge = g.e / n
    local avgd = g.d / n
    local score = avgkl / (avge + avgd + 1e-6)
    combos[#combos+1] = {key=k, avgkl=avgkl, avge=avge, avgd=avgd, score=score}
  end
  -- Sort by score descending
  table.sort(combos, function(a,b) return a.score > b.score end)
  return combos
end

-- NASA-14: recommend proposals to evaluate first
local function nasa14(rows, idx, models)
  local proposals = {}
  for _,r in ipairs(rows) do
    local eff = tonumber(r[idx["Effort-"]])
    local def = tonumber(r[idx["Defects-"]])
    if not eff or not def then
      local k = tonumber(r[idx["Kloc+"]]) or tonumber(r[idx["Kloc"]]) or 0
      if k > 0 then
        local pe, pd = predict(k, models)
        local score = k / (pe + pd + 1e-6)
        proposals[#proposals+1] = {
          id = r[idx["idX"]],
          k = k, pe = pe, pd = pd, score = score
        }
      end
    end
  end
  table.sort(proposals, function(a,b) return a.score > b.score end)
  return proposals
end

-- Main entry
local function main()
  local path = arg[1]
  if not path then
    io.stderr:write("Usage: lua skill_07.lua <csv_path>\n")
    os.exit(1)
  end

  local header, idx, rows = read_rows(path)
  if #rows == 0 then
    print("No data rows found.")
    os.exit(0)
  end

  local types = detect_col_types(header, rows)

  -- Split into completed and proposals
  local completed = {}
  for _,r in ipairs(rows) do
    local eff = tonumber(r[idx["Effort-"]])
    local def = tonumber(r[idx["Defects-"]])
    if eff and def then
      table.insert(completed, r)
    end
  end

  local models = build_models(completed, idx, types)
  if not models then
    print("Not enough completed projects to build models.")
    os.exit(0)
  end

  -- NASA-13
  print("== NASA-13 ==")
  local combos = nasa13(completed, idx, types)
  if #combos == 0 then
    print("No symbolic combinations found.")
  else
    for i=1, math.min(5, #combos) do
      local c = combos[i]
      print(string.format("Combo %d: %s => Avg Kloc=%.1f, Avg Effort=%.1f, Avg Defects=%.1f, Score=%.3f",
          i, c.key, round(c.avgkl), round(c.avge), round(c.avgd), round(c.score,3)))
    end
  end

  -- NASA-14
  print("\n== NASA-14 ==")
  local proposals = nasa14(rows, idx, models)
  if #proposals == 0 then
    print("No proposals with missing Effort or Defects.")
  else
    for i=1, math.min(5, #proposals) do
      local p = proposals[i]
      print(string.format("Proposal %d: ID=%s, Kloc=%.1f, Pred Effort=%.1f, Pred Defects=%.1f, Score=%.3f",
          i, show(p.id), round(p.k), round(p.pe), round(p.pd), round(p.score,3)))
    end
  end
end

main()
