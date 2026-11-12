#!/usr/bin/env lua
-- Skill 01: Summaries and outlier projects for NASA-93 data
-- Usage: lua skill_01.lua <csv_path>

local ezr = require "ezr"
local show = ezr.show
local push = ezr.push
local sorted = ezr.sorted
local keys = ezr.keys
local rand = ezr.rand
local map = ezr.map

-- ensure deterministic random stream
ezr.srand(ezr.the.seed)

-- Helper: return the first column in tbl.cols.all with this name
local function find_col(tbl, name)
  for _,c in pairs(tbl.cols.all) do
    if c.name == name then return c end
  end
  error("column not found: " .. name)
end

-- Helper: numeric column test
local function is_num(c)
  return type(c:mid()) == "number"
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

-- Find best and worst rows w.r.t a goal column (lower is better)
local function best_worst(tbl, goal_col)
  local best, worst = nil, nil
  for _,r in ipairs(tbl.rows) do
    local v = r[goal_col.at]
    if v ~= "?" then
      if not best or v < best.val then
        best = {row = r, val = v}
      end
      if not worst or v > worst.val then
        worst = {row = r, val = v}
      end
    end
  end
  return best, worst
end

-- NASA-01: Summary of project size, effort, defects
local function nasa_01(tbl)
  print("== NASA-01 ==")
  local size_col   = find_col(tbl, "Kloc+")
  local effort_col = find_col(tbl, "Effort-")
  local defect_col = find_col(tbl, "Defects-")

  local size_stats   = col_stats(tbl, size_col)
  local effort_stats = col_stats(tbl, effort_col)
  local defect_stats = col_stats(tbl, defect_col)

  -- Print stats
  local function fmt_stats(s)
    return string.format("%s: mean %.1f, median %.1f, sd %.1f, min %.1f, max %.1f",
      s.name, s.mean, s.median, s.sd, s.min, s.max)
  end

  print(fmt_stats(size_stats))
  print(fmt_stats(effort_stats))
  print(fmt_stats(defect_stats))

  -- Best / worst by Effort- (lower effort better)
  local best, worst = best_worst(tbl, effort_col)
  print(string.format("Best effort: id %s (Effort- %.1f)",
        best.row[best.row.at], best.val))
  print(string.format("Worst effort: id %s (Effort- %.1f)",
        worst.row[worst.row.at], worst.val))
end

-- NASA-02: Projects with unusually favorable combinations
local function nasa_02(tbl)
  print("== NASA-02 ==")

  local size_col   = find_col(tbl, "Kloc+")
  local effort_col = find_col(tbl, "Effort-")
  local defect_col = find_col(tbl, "Defects-")

  -- Compute medians
  local size_vals   = map(tbl.rows, function(r) return r[size_col.at] end)
  local effort_vals = map(tbl.rows, function(r) return r[effort_col.at] end)
  local defect_vals = map(tbl.rows, function(r) return r[defect_col.at] end)

  local size_med   = median(size_vals)
  local effort_med = median(effort_vals)
  local defect_med = median(defect_vals)

  -- Select projects below or equal to median on all three metrics
  local selected = {}
  for _,r in ipairs(tbl.rows) do
    local sz = r[size_col.at]
    local ef = r[effort_col.at]
    local df = r[defect_col.at]
    if sz <= size_med and ef <= effort_med and df <= defect_med then
      push(selected, {id = r[1], size = sz, effort = ef, defects = df})
    end
  end

  -- Print selected projects
  if #selected == 0 then
    print("No projects meet all three median thresholds.")
  else
    print(string.format("Projects with size<=%.1f, effort<=%.1f, defects<=%.1f:",
          size_med, effort_med, defect_med))
    for _,p in ipairs(selected) do
      print(string.format("id %s: size %.1f, effort %.1f, defects %.1f",
            p.id, p.size, p.effort, p.defects))
    end
  end
end

-- Main
local function main()
  local path = arg[1]
  if not path then
    io.stderr:write("Usage: lua skill_01.lua <csv_path>\n")
    os.exit(1)
  end
  local tbl = load_tbl(path)
  nasa_01(tbl)
  nasa_02(tbl)
end

main()
