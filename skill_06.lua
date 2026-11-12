#!/usr/bin/env lua
-- Skill 06: NASA-11 & NASA-12
-- Usage: lua skill_06.lua <csv path>
-- Requires the ezr library (ezr.lua / ezr-lib.lua)

local ezr = require("ezr")

local path = arg[1]
if not path then
  error("Usage: lua skill_06.lua <csv path>")
end

-- ------------------------------------------------------------------
-- Helpers
-- ------------------------------------------------------------------
local function read_csv(path)
  local header, rows = nil, {}
  for r in ezr.csv(path) do
    if not header then
      header = r
      for i, n in ipairs(header) do header[i] = tostring(n) end
    else
      table.insert(rows, r)
    end
  end
  return header, rows
end

local function numeric_columns(header, rows)
  local nums = {}
  local first = rows[1]
  for i, name in ipairs(header) do
    if type(first[i]) == "number" then
      nums[name] = i
    end
  end
  return nums
end

-- Simple linear regression (from skill_04)
local function linear_fit(rows, yname, features, colidx)
  local n = #rows
  local p = 1 + #features
  local xtx = {}
  local xty = {}
  for i = 1, p do
    xtx[i] = {}
    for j = 1, p do xtx[i][j] = 0 end
    xty[i] = 0
  end
  for _, row in ipairs(rows) do
    local x = {1}
    for _, fn in ipairs(features) do
      x[#x+1] = row[colidx[fn]]
    end
    local y = row[colidx[yname]]
    for i = 1, p do
      for j = 1, p do
        xtx[i][j] = xtx[i][j] + x[i] * x[j]
      end
      xty[i] = xty[i] + x[i] * y
    end
  end
  local beta = {}
  local A = {}
  for i = 1, p do
    A[i] = {}
    for j = 1, p do A[i][j] = xtx[i][j] end
    A[i][p+1] = xty[i]
  end
  for i = 1, p do
    local diag = A[i][i]
    if diag == 0 then error("Singular matrix") end
    for j = i, p+1 do
      A[i][j] = A[i][j] / diag
    end
    for k = i+1, p do
      local factor = A[k][i]
      for j = i, p+1 do
        A[k][j] = A[k][j] - factor * A[i][j]
      end
    end
  end
  for i = p, 1, -1 do
    beta[i] = A[i][p+1]
    for j = i+1, p do
      beta[i] = beta[i] - A[i][j] * beta[j]
    end
  end
  return beta
end

local function predict(beta, row, features, colidx)
  local val = beta[1]
  for i = 1, #features do
    val = val + beta[i+1] * row[colidx[features[i]]]
  end
  return val
end

local function mean(v)
  local s = 0
  for _, x in ipairs(v) do s = s + x end
  return s / #v
end

local function sd(v, m)
  local ss = 0
  for _, x in ipairs(v) do ss = ss + (x - m) * (x - m) end
  return math.sqrt(ss / (#v - 1))
end

-- ------------------------------------------------------------------
-- Data ingestion
-- ------------------------------------------------------------------
local header, rows = read_csv(path)
local idx = {}
for i, n in ipairs(header) do idx[n] = i end

local numeric = numeric_columns(header, rows)

-- Identify known vs proposal rows
local known = {}
local proposals = {}
for _, r in ipairs(rows) do
  local eff = r[idx["Effort-"]]
  local def = r[idx["Defects-"]]
  if eff ~= "?" and def ~= "?" then
    table.insert(known, r)
  else
    table.insert(proposals, r)
  end
end

if #known == 0 then
  error("No completed-project records found for training")
end

-- ------------------------------------------------------------------
-- Build models
-- ------------------------------------------------------------------
local effort_features = {}
local defect_features = {}
for name,_ in pairs(numeric) do
  if name ~= "idX" and name ~= "Effort-" then table.insert(effort_features, name) end
  if name ~= "idX" and name ~= "Defects-" then table.insert(defect_features, name) end
end
table.sort(effort_features)
table.sort(defect_features)

local effort_beta = linear_fit(known, "Effort-", effort_features, idx)
local defect_beta = linear_fit(known, "Defects-", defect_features, idx)

-- Historical stats for z-score
local eff_hist = {}
local def_hist = {}
for _, r in ipairs(known) do
  table.insert(eff_hist, r[idx["Effort-"]])
  table.insert(def_hist, r[idx["Defects-"]])
end
local eff_mean, eff_sd = mean(eff_hist), sd(eff_hist, mean(eff_hist))
local def_mean, def_sd = mean(def_hist), sd(def_hist, mean(def_hist))

-- ------------------------------------------------------------------
-- Standardized numeric features (for similarity)
-- ------------------------------------------------------------------
local std_stats = {}
for name,i in pairs(numeric) do
  local vals = {}
  for _, r in ipairs(known) do
    table.insert(vals, r[i])
  end
  local mu = mean(vals)
  local sigma = sd(vals, mu)
  std_stats[name] = {mu=mu, sigma=sigma}
end

local function std_val(v, name)
  local s = std_stats[name]
  return (v - s.mu) / (s.sigma + 1e-12)
end

local function dist_between(a,b)
  local d = 0
  for name,_ in pairs(numeric) do
    if name ~= "idX" and name ~= "Effort-" and name ~= "Defects-" then
      local va = std_val(a[idx[name]], name)
      local vb = std_val(b[idx[name]], name)
      d = d + (va - vb)*(va - vb)
    end
  end
  return math.sqrt(d)
end

-- ------------------------------------------------------------------
-- NASA-11: Explanation for each proposal
-- ------------------------------------------------------------------
print("== NASA-11 ==")
for _, prop in ipairs(proposals) do
  local pid = prop[idx["idX"]]
  -- Predict effort & defects
  local est_eff = predict(effort_beta, prop, effort_features, idx)
  local est_def = predict(defect_beta, prop, defect_features, idx)
  local eff_z = eff_sd > 0 and (est_eff - eff_mean)/eff_sd or 0
  local def_z = def_sd > 0 and (est_def - def_mean)/def_sd or 0
  -- Find 3 nearest historical projects
  local neighbors = {}
  for _, r in ipairs(known) do
    local d = dist_between(prop, r)
    table.insert(neighbors, {row=r, dist=d})
  end
  table.sort(neighbors, function(a,b) return a.dist < b.dist end)
  local top = {}
  for i=1, math.min(3, #neighbors) do
    local r = neighbors[i].row
    table.insert(top, {
      id = r[idx["idX"]],
      eff = r[idx["Effort-"]],
      def = r[idx["Defects-"]],
      dist = neighbors[i].dist
    })
  end
  -- Weighted mean estimate
  local w_eff,w_def,totw = 0,0,0
  for _, n in ipairs(top) do
    local w = 1/(n.dist+1e-6)
    w_eff = w_eff + n.eff * w
    w_def = w_def + n.def * w
    totw = totw + w
  end
  local w_est_eff, w_est_def = w_eff/totw, w_def/totw
  print(string.format("Project %s:", pid))
  print(string.format("  Predicted effort  = %s LOC (z=%.2f)",
        ezr.round(est_eff), ezr.round(eff_z)))
  print(string.format("  Predicted defects = %s (z=%.2f)",
        ezr.round(est_def), ezr.round(def_z)))
  print("  Nearest historical projects:")
  for _, n in ipairs(top) do
    print(string.format("    ID %s: effort %.1f, defects %.1f (dist=%.3f)",
          n.id, ezr.round(n.eff), ezr.round(n.def), n.dist))
  end
  print("  Weighted estimate from neighbors: effort %.1f, defects %.1f",
        ezr.round(w_est_eff), ezr.round(w_est_def))
end

-- ------------------------------------------------------------------
-- NASA-12: Compare two proposals
-- ------------------------------------------------------------------
if #proposals >= 2 then
  local p1, p2 = proposals[1], proposals[2]
  local id1, id2 = p1[idx["idX"]], p2[idx["idX"]]
  local eff1 = predict(effort_beta, p1, effort_features, idx)
  local def1 = predict(defect_beta, p1, defect_features, idx)
  local eff2 = predict(effort_beta, p2, effort_features, idx)
  local def2 = predict(defect_beta, p2, defect_features, idx)
  local risk1 = (eff1 - eff_mean)/eff_sd + (def1 - def_mean)/def_sd
  local risk2 = (eff2 - eff_mean)/eff_sd + (def2 - def_mean)/def_sd
  local risk_diff = risk1 - risk2

  print("\n== NASA-12 ==")
  print(string.format("Comparing proposals %s vs %s", id1, id2))
  print(string.format("  Risk score %s: %.2f, %s: %.2f, Δ=%.2f",
        id1, ezr.round(risk1,2), id2, ezr.round(risk2,2), ezr.round(risk_diff,2)))

  -- Identify top contributing features (by coefficient * delta)
  local contribs = {}
  for _, name in ipairs(effort_features) do
    local delta = p1[idx[name]] - p2[idx[name]]
    local coef = effort_beta[table.indexOf(effort_features, name)+1]
    table.insert(contribs, {name=name, delta=delta, contrib=coef*delta})
  end
  for _, name in ipairs(defect_features) do
    local delta = p1[idx[name]] - p2[idx[name]]
    local coef = defect_beta[table.indexOf(defect_features, name)+1]
    table.insert(contribs, {name=name, delta=delta, contrib=coef*delta})
  end
  table.sort(contribs, function(a,b) return math.abs(a.contrib) > math.abs(b.contrib) end)
  print("  Main differences by feature:")
  for i=1, math.min(3, #contribs) do
    local c = contribs[i]
    print(string.format("    %s: Δ=%.2f, effect=%.2f",
          c.name, c.delta, c.contrib))
  end
else
  print("\n== NASA-12 ==")
  print("Not enough proposal rows to compare risk assessments.")
end
