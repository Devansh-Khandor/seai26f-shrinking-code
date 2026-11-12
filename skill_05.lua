#!/usr/bin/env lua
-- Skill 05 – Apache configuration blame analysis
--
-- Usage:  lua skill_05.lua <csv path>
--
-- Requires the ezr library (ezr.lua, ezr-lib.lua) in the same directory.
--
-- APACHE-09: Identify individual flags most strongly linked to poor performance.
-- APACHE-10: Find pairs of flags that co‑occur in the worst configurations and
--             compare their average performance to the overall mean.

local ezr = require("ezr")

-----------------------------------------------------------------------
-- Helper utilities
-----------------------------------------------------------------------
-- Convert a row to a CSV string (excluding the goal column)
local function row_to_csv(row, goal_at)
  local parts = {}
  for i, v in ipairs(row) do
    if i ~= goal_at then
      table.insert(parts, tostring(v))
    end
  end
  return table.concat(parts, ",")
end

-- Detect flag columns (numeric or symbolic) that only take the values 0 and 1.
local function flag_columns(tbl)
  local flags = {}
  local goal_at = tbl.cols.y[1].at
  for _, c in ipairs(tbl.cols.all) do
    if c.at == goal_at then
      -- Skip the goal column
    else
      local seen = {}
      local is_flag = true
      for _, r in ipairs(tbl.rows) do
        local v = r[c.at]
        if v ~= 0 and v ~= 1 then
          is_flag = false
          break
        end
        seen[v] = true
      end
      if is_flag and #ezr.keys(seen) <= 2 then
        table.insert(flags, c)
      end
    end
  end
  return flags
end

-----------------------------------------------------------------------
-- APACHE-09: Individual flag impact
-----------------------------------------------------------------------
local function report_flag_impact(tbl, flags)
  local goal_at = tbl.cols.y[1].at
  print("== APACHE-09 ==")
  print(string.format("%-20s %-10s %-10s %-10s",
                      "Flag", "Mean1", "Mean0", "Diff"))
  local rows = tbl.rows
  for _, c in ipairs(flags) do
    local sum1, n1, sum0, n0 = 0, 0, 0, 0
    for _, r in ipairs(rows) do
      local v = r[c.at]
      if v == 1 then
        sum1, n1 = sum1 + r[goal_at], n1 + 1
      elseif v == 0 then
        sum0, n0 = sum0 + r[goal_at], n0 + 1
      end
    end
    local mean1 = n1 > 0 and sum1 / n1 or 0
    local mean0 = n0 > 0 and sum0 / n0 or 0
    local diff = mean1 - mean0
    print(string.format("%-20s %10.2f %10.2f %10.2f",
                        c.name, mean1, mean0, diff))
  end
  print()
end

-----------------------------------------------------------------------
-- APACHE-10: Flag pair blame
-----------------------------------------------------------------------
local function report_flag_pairs(tbl, flags)
  local goal_at = tbl.cols.y[1].at
  print("== APACHE-10 ==")

  -- Determine the worst 25 % of rows
  local sorted = ezr.keysort(tbl.rows, function(r) return r[goal_at] end)
  local worst_n = math.max(1, math.floor(#sorted * 0.25))
  local worst_rows = ezr.slice(sorted, 1, worst_n)

  -- Count pair co‑occurrence in worst rows
  local pair_counts = {}
  local overall_pairs = {}
  for _, r in ipairs(tbl.rows) do
    local present = {}
    for _, c in ipairs(flags) do
      if r[c.at] == 1 then present[c.name] = true end
    end
    -- Build sorted list of flag names that are 1
    local names = {}
    for n in pairs(present) do table.insert(names, n) end
    table.sort(names)
    -- All unique pairs
    for i = 1, #names-1 do
      for j = i+1, #names do
        local key = names[i] .. " & " .. names[j]
        overall_pairs[key] = (overall_pairs[key] or 0) + 1
      end
    end
  end

  for _, r in ipairs(worst_rows) do
    local present = {}
    for _, c in ipairs(flags) do
      if r[c.at] == 1 then present[c.name] = true end
    end
    local names = {}
    for n in pairs(present) do table.insert(names, n) end
    table.sort(names)
    for i = 1, #names-1 do
      for j = i+1, #names do
        local key = names[i] .. " & " .. names[j]
        pair_counts[key] = (pair_counts[key] or 0) + 1
      end
    end
  end

  -- Compute mean performance for each pair (both flags = 1)
  local pair_perf = {}
  for key, _ in pairs(overall_pairs) do
    local f1, f2 = key:match("(.+) & (.+)")
    local sum, n = 0, 0
    for _, r in ipairs(tbl.rows) do
      if r[ezr.find_col_index(tbl, f1)].at and
         r[ezr.find_col_index(tbl, f2)].at then
        local v1 = r[ezr.find_col_index(tbl, f1)].at
        local v2 = r[ezr.find_col_index(tbl, f2)].at
        if r[ezr.find_col_index(tbl, f1)].at == 1 and r[ezr.find_col_index(tbl, f2)].at == 1 then
          sum, n = sum + r[goal_at], n + 1
        end
      end
    end
    pair_perf[key] = n > 0 and sum / n or nil
  end

  -- Overall mean performance
  local overall_mean = tbl.cols.y[1]:mid()

  -- Build report table
  local report = {}
  for key, cnt in pairs(pair_counts) do
    if pair_perf[key] then
      local diff = pair_perf[key] - overall_mean
      table.insert(report, {key=key, count=cnt, mean=pair_perf[key], diff=diff})
    end
  end

  -- Sort by worst mean (descending)
  table.sort(report, function(a,b) return a.mean > b.mean end)

  print(string.format("%-30s %-6s %-10s %-10s",
                      "Pair", "Freq", "Mean", "Δfrom overall"))
  for i = 1, math.min(3, #report) do
    local r = report[i]
    print(string.format("%-30s %6d %10.2f %10.2f",
                        r.key, r.count, r.mean, r.diff))
  end
  print()
end

-----------------------------------------------------------------------
-- Main routine
-----------------------------------------------------------------------
local function main(csv_path)
  if not csv_path then
    print("Usage: lua skill_05.lua <csv path>")
    os.exit(1)
  end

  -- Load data
  local tbl = ezr.Tbl(ezr.csv(csv_path))

  -- Identify flag columns
  local flags = flag_columns(tbl)
  if #flags == 0 then
    print("No flag columns (0/1) found.")
    os.exit(1)
  end

  -- APACHE-09
  report_flag_impact(tbl, flags)

  -- APACHE-10
  report_flag_pairs(tbl, flags)
end

main(arg[1])
