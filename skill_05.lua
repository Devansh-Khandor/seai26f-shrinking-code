#!/usr/bin/env lua
-- Skill 05: Identify characteristics driving high effort and defect projects
-- Usage: lua skill_05.lua <csv_path>

local ezr = require "ezr"
local show  = ezr.show
local push  = ezr.push
local sorted = ezr.sorted
local rand  = ezr.rand
local map   = ezr.map
local keys  = ezr.keys

-- deterministic random stream
ezr.srand(ezr.the.seed)

-- ------------------------------------------------------------------
-- Helper: first column by name
local function find_col(tbl, name)
  for _,c in pairs(tbl.cols.all) do
    if c.name == name then return c end
  end
  error("column not found: " .. name)
end

-- Helper: numeric column test
local function is_num(col)
  return type(col:mid()) == "number"
end

-- Helper: compute statistics for numeric column
local function col_stats(tbl, col)
  local vals = map(tbl.rows, function(r) return r[col.at] end)
  local sorted_vals = sorted(vals)
  local n = #sorted_vals
  local mean = col.mu
  local sd   = col.div()
  local med  = sorted_vals[(n + 1) // 2]
  return {
    name   = col.name,
    mean   = mean,
    median = med,
    sd     = sd,
    min    = sorted_vals[1],
    max    = sorted_vals[n]
  }
end

-- ------------------------------------------------------------------
-- Return rows whose goal value >= mean + sd
local function high_rows(tbl, goal_col)
  local thresh = goal_col.mu + goal_col:div()
  local out = {}
  for _,r in ipairs(tbl.rows) do
    local v = r[goal_col.at]
    if v ~= "?" and v >= thresh then push(out, r) end
  end
  return out
end

-- ------------------------------------------------------------------
-- For a given goal (Effort- or Defects-), find the 3 characteristics
-- most strongly associated with high values.
local function analyze(tbl, goal_col, title)
  print("== " .. title .. " ==")
  local high = high_rows(tbl, goal_col)
  if #high == 0 then
    print("No projects exceed mean + sd for " .. goal_col.name)
    return
  end

  -- Compute overall means for numeric cols, overall mode counts for sym cols
  local overall = {}
  for _,c in ipairs(tbl.cols.x) do
    if is_num(c) then
      overall[c.at] = {num=true, mean=c.mu, sd=c:div()}
    else
      overall[c.at] = {num=false, mode=c:mid(), freq=c.has[c:mid()] or 0, total=c.n}
    end
  end

  -- Compute high-group stats
  local diff_list = {}
  for _,c in ipairs(tbl.cols.x) do
    if is_num(c) then
      -- mean difference
      local sum = 0
      local cnt = 0
      for _,r in ipairs(high) do
        local v = r[c.at]
        if v ~= "?" then
          sum = sum + v
          cnt = cnt + 1
        end
      end
      if cnt > 0 then
        local high_mean = sum / cnt
        local diff = math.abs(high_mean - overall[c.at].mean)
        push(diff_list, {col=c, diff=diff, kind="num", high_mean=high_mean})
      end
    else
      -- symbolic: compare proportion of mode
      local mode = overall[c.at].mode
      local mode_cnt = 0
      local total_cnt = 0
      for _,r in ipairs(high) do
        local v = r[c.at]
        if v ~= "?" then
          total_cnt = total_cnt + 1
          if v == mode then mode_cnt = mode_cnt + 1 end
        end
      end
      if total_cnt > 0 then
        local prop = mode_cnt / total_cnt
        local overall_prop = overall[c.at].freq / overall[c.at].total
        local diff = math.abs(prop - overall_prop)
        push(diff_list, {col=c, diff=diff, kind="sym", prop=prop, overall_prop=overall_prop})
      end
    end
  end

  -- sort by diff descending
  table.sort(diff_list, function(a,b) return a.diff > b.diff end)

  -- print top 3
  for i=1,math.min(3,#diff_list) do
    local d = diff_list[i]
    if d.kind == "num" then
      print(string.format("%d. %s (numeric) – high mean %.1f vs overall %.1f",
          i, d.col.name, d.high_mean, overall[d.col.at].mean))
    else
      print(string.format("%d. %s (symbolic) – mode proportion %.2f vs overall %.2f",
          i, d.col.name, d.prop, d.overall_prop))
    end
  end
end

-- ------------------------------------------------------------------
-- Main
local function main()
  local path = arg[1]
  if not path then
    io.stderr:write("Usage: lua skill_05.lua <csv_path>\n")
    os.exit(1)
  end

  local tbl = ezr.Tbl(ezr.csv(path))

  local effort_col = find_col(tbl, "Effort-")
  local defect_col = find_col(tbl, "Defects-")

  analyze(tbl, effort_col, "NASA-09")
  analyze(tbl, defect_col, "NASA-10")
end

main()
