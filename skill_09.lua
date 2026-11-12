#!/usr/bin/env lua
-- Skill 09: NASA-17 & NASA-18 – Outlier detection on characteristics and target values
-- Usage: lua skill_09.lua <csv_path>

local ezr = require("ezr-lib")
local csv  = ezr.csv
local show = ezr.show
local round = ezr.round
local keys  = ezr.keys
local sorted = ezr.sorted
local keysort = ezr.keysort

-- ------------------------------------------------------------------
-- Helper: detect numeric columns (based on first non-header row)
-- ------------------------------------------------------------------
local function detect_numeric(header, rows)
  local numeric = {}
  if #rows == 0 then return numeric end
  local first = rows[1]
  for i, val in ipairs(first) do
    if tonumber(val) ~= nil then numeric[i] = true end
  end
  return numeric
end

-- ------------------------------------------------------------------
-- Helper: compute min and max for each numeric column
-- ------------------------------------------------------------------
local function minmax_numeric(rows, num_idx)
  local mn, mx = {}, {}
  for _,row in ipairs(rows) do
    for _,i in ipairs(num_idx) do
      local v = tonumber(row[i])
      if v then
        if mn[i] == nil or v < mn[i] then mn[i] = v end
        if mx[i] == nil or v > mx[i] then mx[i] = v end
      end
    end
  end
  return mn, mx
end

-- ------------------------------------------------------------------
-- Helper: normalize a numeric value given min/max
-- ------------------------------------------------------------------
local function norm_val(v, mn, mx)
  if v == nil or mn == nil or mx == nil or mx == mn then return 0 end
  return (v - mn) / (mx - mn)
end

-- ------------------------------------------------------------------
-- NASA-17: Flag historical projects with unusual combinations
-- ------------------------------------------------------------------
local function flag_unusual_projects(header, rows, num_idx, char_idx)
  -- Compute centroid of normalized characteristics
  local cent = {}
  local count = 0
  for _,row in ipairs(rows) do
    for _,i in ipairs(char_idx) do
      local v = tonumber(row[i])
      local n = norm_val(v, minmax[i].mn, minmax[i].mx)
      cent[i] = (cent[i] or 0) + n
    end
    count = count + 1
  end
  for i,avg in pairs(cent) do cent[i] = avg / count end

  -- Compute distance from centroid for each project
  local outliers = {}
  for _,row in ipairs(rows) do
    local d2 = 0
    for _,i in ipairs(char_idx) do
      local v = tonumber(row[i])
      local n = norm_val(v, minmax[i].mn, minmax[i].mx)
      local diff = n - cent[i]
      d2 = d2 + diff * diff
    end
    table.insert(outliers, {row=row, dist=math.sqrt(d2)})
  end

  -- Sort by descending distance and pick top 5
  table.sort(outliers, function(a,b) return a.dist > b.dist end)
  return table.move(outliers, 1, math.min(5, #outliers), 1, {})
end

-- ------------------------------------------------------------------
-- NASA-18: Flag projects whose Effort/Defects deviate from model
-- ------------------------------------------------------------------
local function flag_discrepancies(header, rows, idx)
  -- Build regression models on completed projects
  local kx, ye, yd = {}, {}, {}
  for _,r in ipairs(rows) do
    local e = tonumber(r[idx["Effort-"]])
    local d = tonumber(r[idx["Defects-"]])
    local k = tonumber(r[idx["Kloc+"]]) or tonumber(r[idx["Kloc"]])
    if e and d and k then
      kx[#kx+1] = k
      ye[#ye+1] = e
      yd[#yd+1] = d
    end
  end
  if #kx < 2 then return {}, {}, {} end

  local function linreg(x, y)
    local n = #x
    local sx, sy = 0, 0
    for i=1,n do sx = sx + x[i]; sy = sy + y[i] end
    local mx, my = sx/n, sy/n
    local cov, varx = 0, 0
    for i=1,n do
      local dx, dy = x[i]-mx, y[i]-my
      cov = cov + dx*dy
      varx = varx + dx*dx
    end
    return cov/varx, my - (cov/varx)*mx
  end

  local slope_e, intercept_e = linreg(kx, ye)
  local slope_d, intercept_d = linreg(kx, yd)

  -- Residuals for completed projects
  local res_e, res_d = {}, {}
  for i,k in ipairs(kx) do
    local pe = slope_e * k + intercept_e
    local pd = slope_d * k + intercept_d
    res_e[#res_e+1] = ye[i] - pe
    res_d[#res_d+1] = yd[i] - pd
  end

  local function mean(t)
    local s=0
    for _,v in ipairs(t) do s=s+v end
    return #t>0 and s/#t or 0
  end
  local function sd(t, m)
    if #t<2 then return 0 end
    local ss=0
    for _,v in ipairs(t) do ss=ss+(v-m)^2 end
    return math.sqrt(ss/(#t-1))
  end

  local mu_e = mean(res_e); local sd_e = sd(res_e, mu_e)
  local mu_d = mean(res_d); local sd_d = sd(res_d, mu_d)

  -- Flag projects
  local flagged_e, flagged_d = {}, {}
  for _,r in ipairs(rows) do
    local e = tonumber(r[idx["Effort-"]])
    local d = tonumber(r[idx["Defects-"]])
    local k = tonumber(r[idx["Kloc+"]]) or tonumber(r[idx["Kloc"]])
    local id = r[idx["idX"]]
    if e and d and k then
      local pe = slope_e * k + intercept_e
      local pd = slope_d * k + intercept_d
      local z_e = (e - pe) / (sd_e+1e-9)
      local z_d = (d - pd) / (sd_d+1e-9)
      if math.abs(z_e) > 2 then
        table.insert(flagged_e, {id=id, actual=e, pred=pe, z= z_e, diff= e-pe})
      end
      if math.abs(z_d) > 2 then
        table.insert(flagged_d, {id=id, actual=d, pred=pd, z= z_d, diff= d-pd})
      end
    end
  end
  return flagged_e, flagged_d
end

-- ------------------------------------------------------------------
-- Main
-- ------------------------------------------------------------------
local function main()
  local path = arg[1]
  if not path then
    io.stderr:write("Usage: lua skill_09.lua <csv_path>\n")
    os.exit(1)
  end

  local rows = {}
  local header = nil
  local idx = {}
  for rec in csv(path) do
    if not header then
      header = rec
      for i,name in ipairs(rec) do idx[name] = i end
    else
      rows[#rows+1] = rec
    end
  end

  if #rows == 0 then
    print("No data rows found.")
    os.exit(0)
  end

  -- Detect numeric columns
  local numeric = detect_numeric(header, rows)
  local num_idx = {}
  local char_idx = {}
  for i, val in pairs(numeric) do
    local name = header[i]
    if name == "Effort-" or name == "Defects-" or name == "Kloc+" then
      -- target or goal columns, skip for characteristics
    else
      num_idx[#num_idx+1] = i
      table.insert(char_idx, i)
    end
  end

  -- Compute min/max for numeric characteristics
  local minmax = {}
  for _,i in ipairs(num_idx) do
    minmax[i] = {}
  end
  local mn, mx = minmax_numeric(rows, num_idx)
  for i,_ in ipairs(num_idx) do
    minmax[i].mn = mn[i]
    minmax[i].mx = mx[i]
  end

  -- NASA-17: flag unusual characteristic combos
  print("== NASA-17 ==")
  local outliers = flag_unusual_projects(header, rows, num_idx, char_idx)
  if #outliers == 0 then
    print("No historical projects found.")
  else
    for _,info in ipairs(outliers) do
      local r = info.row
      local id = r[idx["idX"]]
      local dist = round(info.dist, 4)
      local kloc = r[idx["Kloc+"]]
      local eff = r[idx["Effort-"]]
      local def = r[idx["Defects-"]]
      print(string.format("ID=%s: Dist=%.4f, Kloc+=%s, Effort-=%s, Defects-=%s",
            show(id), dist, show(kloc), show(eff), show(def)))
    end
  end

  -- NASA-18: flag discrepant Effort/Defects
  print("\n== NASA-18 ==")
  local flagged_e, flagged_d = flag_discrepancies(header, rows, idx)
  if #flagged_e == 0 then
    print("No Effort outliers detected.")
  else
    print("Effort outliers (|z| > 2):")
    for _,f in ipairs(flagged_e) do
      print(string.format("ID=%s: Actual=%.2f, Pred=%.2f, Δ=%.2f, z=%.2f",
            show(f.id), f.actual, round(f.pred,2), round(f.diff,2), round(f.z,2)))
    end
  end
  if #flagged_d == 0 then
    print("\nNo Defects outliers detected.")
  else
    print("\nDefects outliers (|z| > 2):")
    for _,f in ipairs(flagged_d) do
      print(string.format("ID=%s: Actual=%.2f, Pred=%.2f, Δ=%.2f, z=%.2f",
            show(f.id), f.actual, round(f.pred,2), round(f.diff,2), round(f.z,2)))
    end
  end
end

main()
