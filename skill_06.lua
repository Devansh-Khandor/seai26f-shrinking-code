#!/usr/bin/env lua
-- Skill 06: NASA-11 & NASA-12 – Explain effort estimates and compare risk
-- Usage: lua skill_06.lua <csv_path>

local ezr = require("ezr-lib")
local csv  = ezr.csv
local show = ezr.show
local round = ezr.round
local keys = ezr.keys
local sorted = ezr.sorted

-- --------------------------------------------------------------------
-- Simple helper functions
-- --------------------------------------------------------------------
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

local function linear_regression(x, y)
  local n = #x
  if n < 2 then return 0,0,0 end
  local mx, my = mean(x), mean(y)
  local cov, varx = 0, 0
  for i=1,n do
    local dx, dy = x[i]-mx, y[i]-my
    cov = cov + dx*dy
    varx = varx + dx*dx
  end
  local slope = cov/varx
  local intercept = my - slope*mx
  -- residual standard error
  local se = 0
  for i=1,n do
    local pred = slope*x[i] + intercept
    se = se + (y[i]-pred)^2
  end
  se = math.sqrt(se/(n-2))
  return slope, intercept, se end

-- --------------------------------------------------------------------
-- Read data and split into completed and proposals
-- --------------------------------------------------------------------
local function read_data(path)
  local rows = {}
  local header, idx = nil, {}
  for rec in csv(path) do
    if not header then
      header = rec
      for i, name in ipairs(rec) do idx[name] = i end
    else
      local r = {}
      for i,v in ipairs(rec) do r[i] = v end
      rows[#rows+1] = r
    end
  end
  return header, idx, rows end

local function is_numeric(val)
  return tonumber(val) ~= nil end

-- --------------------------------------------------------------------
-- Main logic
-- --------------------------------------------------------------------
local function main()
  local path = arg[1]
  if not path then
    io.stderr:write("Usage: lua skill_06.lua <csv_path>\n")
    os.exit(1)
  end

  local header, idx, rows = read_data(path)
  if #rows == 0 then
    print("No data rows found.")
    os.exit(0)
  end

  -- Identify column indices
  local kloc_i   = idx["Kloc+"] or idx["Kloc"]
  local effort_i = idx["Effort-"]
  local defect_i = idx["Defects-"]
  local team_i   = idx["team"]
  local cplx_i   = idx["cplx"]

  -- Build regression models on completed projects
  local comp = {}
  local kx, ye, yd = {}, {}, {}
  for _,r in ipairs(rows) do
    local e = tonumber(r[effort_i])
    local d = tonumber(r[defect_i])
    local k = tonumber(r[kloc_i])
    if e and d and k then
      comp[#comp+1] = r
      kx[#kx+1] = k
      ye[#ye+1] = e
      yd[#yd+1] = d
    end
  end

  if #comp < 2 then
    print("Not enough completed projects to build a model.")
    os.exit(0)
  end

  local slope_e, intercept_e, se_e = linear_regression(kx, ye)
  local slope_d, intercept_d, se_d = linear_regression(kx, yd)

  local mean_e = mean(ye)
  local sd_e   = sd(ye, mean_e)
  local mean_d = mean(yd)
  local sd_d   = sd(yd, mean_d)

  -- Identify proposals (missing Effort or Defects)
  local proposals = {}
  for _,r in ipairs(rows) do
    if not tonumber(r[effort_i]) or not tonumber(r[defect_i]) then
      proposals[#proposals+1] = r
    end
  end

  if #proposals == 0 then
    print("No proposal records found.")
    os.exit(0)
  end

  -- Compute predictions and risk scores
  local risk_scores = {}
  for _,p in ipairs(proposals) do
    local k = tonumber(p[kloc_i]) or 0
    local pred_e = slope_e * k + intercept_e
    local pred_d = slope_d * k + intercept_d
    -- Standardize for a composite risk
    local z_e = (pred_e - mean_e)/sd_e
    local z_d = (pred_d - mean_d)/sd_d
    local risk = z_e + z_d
    risk_scores[p] = risk
  end

  -- ------------------------------------------------------------------
  -- NASA-11: Explain effort estimates using nearest historical projects
  -- ------------------------------------------------------------------
  print("== NASA-11 ==")
  for _,p in ipairs(proposals) do
    local id = p[idx["idX"]]
    local k = tonumber(p[kloc_i]) or 0
    local pred_e = slope_e * k + intercept_e
    local pred_d = slope_d * k + intercept_d
    print(string.format("Proposal ID=%s: Pred Effort=%.1f, Pred Defects=%.1f",
           show(id), round(pred_e,1), round(pred_d,1)))
    print("  Based on historical projects close in Kloc:")

    -- Find three nearest historical projects by Kloc distance
    local neighbors = {}
    for _,h in ipairs(comp) do
      local hk = tonumber(h[kloc_i])
      if hk then
        local dist = math.abs(k - hk)
        neighbors[#neighbors+1] = {row=h, dist=dist}
      end
    end
    sorted(neighbors, function(a,b) return a.dist < b.dist end)
    for i=1,math.min(3,#neighbors) do
      local h = neighbors[i]
      local hid = h.row[idx["idX"]]
      local he = tonumber(h[row[effort_i]]) or 0
      local hd = tonumber(h[row[defect_i]]) or 0
      print(string.format("    [%d] ID=%s, Kloc=%.1f, Effort=%.1f, Defects=%.1f, |ΔKloc|=%.1f",
             i, show(hid), round(k,1), round(he,1), round(hd,1), round(h.dist,1)))
    end
    print()
  end

  -- ------------------------------------------------------------------
  -- NASA-12: Compare two proposals with different risk assessments
  -- ------------------------------------------------------------------
  print("== NASA-12 ==")
  -- Sort proposals by risk
  local sorted_proposals = sorted(proposals, function(a,b)
    return risk_scores[a] < risk_scores[b] end)

  local low = sorted_proposals[1]
  local high = sorted_proposals[#sorted_proposals]

  local function proj_summary(p)
    return string.format("ID=%s, Kloc=%.1f, team=%s, complexity=%s",
           show(p[idx["idX"]]), tonumber(p[kloc_i]) or 0,
           p[team_i] or "?", p[cplx_i] or "?") end

  print(string.format("Lowest risk proposal: %s (risk=%.2f)",
         proj_summary(low), round(risk_scores[low],2)))
  print(string.format("Highest risk proposal: %s (risk=%.2f)",
         proj_summary(high), round(risk_scores[high],2)))

  print("\nMain differences contributing to risk:")
  local attrs = {"Kloc+", "team", "cplx"}
  for _,attr in ipairs(attrs) do
    local idx_attr = idx[attr]
    local val_low = low[idx_attr] or "?"
    local val_high = high[idx_attr] or "?"
    if val_low ~= val_high then
      print(string.format("  %s: low=%s, high=%s",
             attr, show(val_low), show(val_high)))
    end
  end

  print("\nTrade‑off insights:")
  print("  • A higher Kloc generally leads to higher effort and defect risk.")
  print("  • Stronger team ratings (e.g., 'vh' vs 'h') can offset some effort costs.")
  print("  • Complexity ratings affect defect likelihood; higher complexity tends to raise defect risk.")
end

main()
