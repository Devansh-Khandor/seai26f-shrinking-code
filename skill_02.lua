#!/usr/bin/env lua
-- Skill 02: Effort and defect estimation for NASA-93 projects
-- Usage: lua skill_02.lua <csv_path>

-- Import ezr and re‑use helpers from skill_01
local ezr = require "ezr"
local show = ezr.show
local push = ezr.push
local sorted = ezr.sorted
local keys = ezr.keys
local rand = ezr.rand
local map = ezr.map

-- deterministic random stream
ezr.srand(ezr.the.seed)

-- Helper: return the first column in tbl.cols.all with this name
local function find_col(tbl, name)
  for _,c in pairs(tbl.cols.all) do
    if c.name == name then return c end
  end
  error("column not found: " .. name)
end

-- Helper: median of a numeric list
local function median(vals)
  local s = sorted(vals)
  local n = #s
  return s[(n + 1) // 2]
end

-- Build the table from CSV path
local function load_tbl(path)
  return ezr.Tbl(ezr.csv(path))
end

-- Compute statistics for a numeric column
local function col_stats(tbl, col)
  local vals = map(tbl.rows, function(r) return r[col.at] end)
  local sorted_vals = sorted(vals)
  local n = #sorted_vals
  local mean = col.mu
  local sd   = col.div()
  local med  = sorted_vals[(n + 1) // 2]
  return {
    name = col.name,
    mean = mean,
    median = med,
    sd = sd,
    min = sorted_vals[1],
    max = sorted_vals[n]
  }
end

-- NASA‑03: Estimate effort for each project (using historical mean)
local function nasa_03(tbl)
  print("== NASA-03 ==")
  local effort_col = find_col(tbl, "Effort-")
  local effort_stats = col_stats(tbl, effort_col)

  -- Simple estimate: use overall mean; uncertainty is sd
  print(string.format("Estimated Effort-: mean=%.1f, sd=%.1f",
        effort_stats.mean, effort_stats.sd))

  -- Show the estimate for the first few projects to illustrate
  print("Example estimates for first 5 projects:")
  for i = 1, math.min(5, #tbl.rows) do
    local r = tbl.rows[i]
    -- Use the same mean estimate for each project
    print(string.format("  Project id %s: Effort- ≈ %.1f ± %.1f",
          r[1], effort_stats.mean, effort_stats.sd))
  end
end

-- NASA‑04: Estimate defect count for each project (using historical mean)
local function nasa_04(tbl)
  print("\n== NASA-04 ==")
  local defect_col = find_col(tbl, "Defects-")
  local defect_stats = col_stats(tbl, defect_col)

  print(string.format("Estimated Defects-: mean=%.1f, sd=%.1f",
        defect_stats.mean, defect_stats.sd))

  print("Example estimates for first 5 projects:")
  for i = 1, math.min(5, #tbl.rows) do
    local r = tbl.rows[i]
    print(string.format("  Project id %s: Defects- ≈ %.1f ± %.1f",
          r[1], defect_stats.mean, defect_stats.sd))
  end
end

-- Main
local function main()
  local path = arg[1]
  if not path then
    io.stderr:write("Usage: lua skill_02.lua <csv_path>\n")
    os.exit(1)
  end
  local tbl = load_tbl(path)
  nasa_03(tbl)
  nasa_04(tbl)
end

main()
