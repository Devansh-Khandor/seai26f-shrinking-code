#!/usr/bin/env lua
-- Skill 03: NASA-05 and NASA-06 – Programmer capability & Complexity impact
-- Usage: lua skill_03.lua <csv_path>

local ezr = require("ezr-lib")
local csv  = ezr.csv
local show = ezr.show
local round = ezr.round

-- ------------------------------------------------------------------
-- Helper: compute mean, standard deviation and Cohen's d
local function stats(values)
  local n = #values
  if n == 0 then return {n=0, mean=0, sd=0} end
  local sum = 0
  for _,v in ipairs(values) do sum = sum + v end
  local mean = sum / n
  local ssq = 0
  for _,v in ipairs(values) do
    ssq = ssq + (v - mean)^2
  end
  local sd = math.sqrt(ssq / n)
  return {n=n, mean=mean, sd=sd}
end

local function cohen_d(a,b)
  if a.sd==0 or b.sd==0 then return 0 end
  local pooled = math.sqrt((a.sd^2 + b.sd^2)/2)
  return (a.mean - b.mean) / pooled
end

-- ------------------------------------------------------------------
-- Read the CSV and collect the columns we need
local function read_data(path)
  local rows = {}
  local header = nil
  local idx = {}
  for rec in csv(path) do
    if not header then
      header = rec
      for i,name in ipairs(rec) do idx[name] = i end
    else
      local row = {}
      -- Programmer capability rating (symbolic)
      row.team = rec[idx["team"]]
      -- Complexity rating (symbolic)
      row.cplx = rec[idx["cplx"]]
      -- Effort (numeric)
      row.Effort = tonumber(rec[idx["Effort-"]])
      -- Defects (numeric)
      row.Defects = tonumber(rec[idx["Defects-"]])
      if row.Effort and row.Defects and row.team and row.cplx then
        table.insert(rows, row)
      end
    end
  end
  return rows
end

-- ------------------------------------------------------------------
-- Group rows by a key and compute stats for a numeric field
local function group_stats(rows, key, field)
  local groups = {}
  for _,r in ipairs(rows) do
    local k = r[key]
    local v = r[field]
    if v then
      groups[k] = groups[k] or {}
      table.insert(groups[k], v)
    end
  end
  local stats_table = {}
  for k,vals in pairs(groups) do
    stats_table[k] = stats(vals)
  end
  return stats_table
end

-- ------------------------------------------------------------------
-- Determine if the difference between two groups is meaningful
local function compare_groups(stats_a, stats_b)
  local diff = stats_a.mean - stats_b.mean
  local d = cohen_d(stats_a, stats_b)
  local conclusion
  if math.abs(d) >= 0.5 then
    conclusion = "The difference is large enough to inform planning."
  else
    conclusion = "The evidence is insufficient; differences may be due to noise."
  end
  return {
    mean_diff = diff,
    cohen_d = d,
    conclusion = conclusion
  }
end

-- ------------------------------------------------------------------
-- Main routine
local function main()
  local path = arg[1]
  if not path then
    io.stderr:write("Usage: lua skill_03.lua <csv_path>\n")
    os.exit(1)
  end

  local rows = read_data(path)

  -- NASA-05: Programmer capability vs Effort
  local effort_by_team = group_stats(rows, "team", "Effort")
  io.write("== NASA-05 ==\n")
  for team,st in pairs(effort_by_team) do
    io.write(string.format(
      "Team rating '%s': mean effort=%.2f, sd=%.2f, n=%d\n",
      team, st.mean, st.sd, st.n))
  end
  -- compare highest vs lowest rating
  local teams = {}
  for t,_ in pairs(effort_by_team) do table.insert(teams, t) end
  table.sort(teams)
  if #teams >= 2 then
    local best = effort_by_team[teams[1]]
    local worst = effort_by_team[teams[#teams]]
    local comp = compare_groups(best, worst)
    io.write(string.format(
      "Difference between lowest '%s' and highest '%s' ratings: mean diff=%.2f, Cohen's d=%.2f\n",
      teams[1], teams[#teams], comp.mean_diff, comp.cohen_d))
    io.write(comp.conclusion .. "\n")
  else
    io.write("Not enough rating groups to compare.\n")
  end

  -- NASA-06: Complexity vs Defects
  local defects_by_cplx = group_stats(rows, "cplx", "Defects")
  io.write("\n== NASA-06 ==\n")
  for cplx,st in pairs(defects_by_cplx) do
    io.write(string.format(
      "Complexity rating '%s': mean defects=%.2f, sd=%.2f, n=%d\n",
      cplx, st.mean, st.sd, st.n))
  end
  local cpls = {}
  for c,_ in pairs(defects_by_cplx) do table.insert(cpls, c) end
  table.sort(cpls)
  if #cpls >= 2 then
    local best = defects_by_cplx[cpls[1]]
    local worst = defects_by_cplx[cpls[#cpls]]
    local comp = compare_groups(best, worst)
    io.write(string.format(
      "Difference between lowest '%s' and highest '%s' ratings: mean diff=%.2f, Cohen's d=%.2f\n",
      cpls[1], cpls[#cpls], comp.mean_diff, comp.cohen_d))
    io.write(comp.conclusion .. "\n")
  else
    io.write("Not enough complexity groups to compare.\n")
  end
end

main()
