#!/usr/bin/env lua
--[[
  skill_07.lua
  NASA-93 data analysis – feature‑based recommendation of promising
  project character combinations and an evaluation strategy.
  Usage:  lua skill_07.lua <csv-path>
  Prints:
    NASA-13: combinations of characteristics that historically
             delivered more software with lower effort and fewer defects.
    NASA-14: suggested order of proposals to evaluate first under a
             limited evaluation budget.
]]--

local ezr = require "ezr"

-- ------------------------------------------------------------------
-- Helpers
-- ------------------------------------------------------------------
local function abort(msg)
  io.stderr:write(msg .. "\n")
  os.exit(1)
end

local function round(v)
  return ezr.round(v)
end

-- Find all symbolic column indices (first cell not a number)
local function symbolic_cols(tbl)
  local syms = {}
  for _,c in ipairs(tbl.cols.all) do
    local val = tbl.rows[1][c.at]
    if type(val) ~= "number" then
      table.insert(syms, c)
    end
  end
  return syms
end

-- Group rows by the values of given symbolic columns.
local function group_rows(rows, cols)
  local groups = {}
  for _,row in ipairs(rows) do
    local key = {}
    for _,c in ipairs(cols) do
      table.insert(key, tostring(row[c.at]))
    end
    local k = table.concat(key, "|")
    groups[k] = groups[k] or {rows={}, size=0, cols=cols}
    table.insert(groups[k].rows, row)
    groups[k].size = groups[k].size + 1
  end
  return groups
end

-- Compute aggregated stats for a group
local function group_stats(group, sz_col, eff_col, def_col)
  local n = #group.rows
  if n == 0 then return nil end
  local sum_sz, sum_eff, sum_def = 0, 0, 0
  local cnt_sz, cnt_eff, cnt_def = 0, 0, 0
  for _,r in ipairs(group.rows) do
    local sz = r[sz_col.at]
    local eff = r[eff_col.at]
    local def = r[def_col.at]
    if sz ~= "?" then sum_sz, cnt_sz = sum_sz + sz, cnt_sz + 1 end
    if eff ~= "?" then sum_eff, cnt_eff = sum_eff + eff, cnt_eff + 1 end
    if def ~= "?" then sum_def, cnt_def = sum_def + def, cnt_def + 1 end
  end
  local avg_sz = cnt_sz > 0 and sum_sz / cnt_sz or 0
  local avg_eff = cnt_eff > 0 and sum_eff / cnt_eff or 0
  local avg_def = cnt_def > 0 and sum_def / cnt_def or 0
  local score = avg_sz > 0 and (avg_eff + avg_def) / avg_sz or math.huge
  return {n=n, sz=avg_sz, eff=avg_eff, def=avg_def, score=score}
end

-- ------------------------------------------------------------------
-- Main
-- ------------------------------------------------------------------
local csv_path = arg[1]
if not csv_path then abort("Usage: lua skill_07.lua <csv-path>") end

-- Load rows
local rows = {}
for r in ezr.csv(csv_path) do table.insert(rows, r) end
if #rows == 0 then abort("No data found in " .. csv_path) end

-- Build table structure
local tbl = ezr.Tbl(rows)

-- Map columns by name
local col_map = {}
for i,c in ipairs(tbl.cols.all) do
  col_map[tbl.cols.names[i]] = c
end

-- Columns of interest
local sz_col   = col_map["Kloc+"]
local eff_col  = col_map["Effort-"]
local def_col  = col_map["Defects-"]

-- ------------------------------------------------------------------
-- NASA-13: promising characteristic combinations
-- ------------------------------------------------------------------
print("== NASA-13 ==")

local syms = symbolic_cols(tbl)
local combos = {}

-- 1‑column groups
for _,c in ipairs(syms) do
  local groups = group_rows(rows, {c})
  for k,g in pairs(groups) do
    local stats = group_stats(g, sz_col, eff_col, def_col)
    if stats then
      table.insert(combos, {cols={c}, key=k, stats=stats})
    end
  end
end

-- 2‑column groups
for i=1,#syms-1 do
  for j=i+1,#syms do
    local c1, c2 = syms[i], syms[j]
    local groups = group_rows(rows, {c1, c2})
    for k,g in pairs(groups) do
      local stats = group_stats(g, sz_col, eff_col, def_col)
      if stats then
        table.insert(combos, {cols={c1,c2}, key=k, stats=stats})
      end
    end
  end
end

-- Rank combos: lower score (eff+def)/size best, then larger size
table.sort(combos, function(a,b)
  if a.stats.score ~= b.stats.score then
    return a.stats.score < b.stats.score
  end
  return a.stats.sz > b.stats.sz
end)

-- Show top 3 combos (or fewer if not enough)
local max_show = math.min(3, #combos)
for i=1,max_show do
  local c = combos[i]
  local col_names = {}
  for _,col in ipairs(c.cols) do table.insert(col_names, col.name) end
  print(string.format("%d. %s: size=%.0f, eff=%.1f, def=%.1f, n=%d",
          i, table.concat(col_names, "+"),
          round(c.stats.sz), round(c.stats.eff), round(c.stats.def), c.stats.n))
end

-- ------------------------------------------------------------------
-- NASA-14: recommend evaluation order
-- ------------------------------------------------------------------
print("\n== NASA-14 ==")

-- Determine budget: 5% of total rows, min 5
local budget = math.max(5, math.floor(#rows * 0.05))

-- Compute a score for each row: (eff/size)+(def/size), lower better
local candidates = {}
for _,row in ipairs(rows) do
  local sz = row[sz_col.at]
  local eff = row[eff_col.at]
  local def = row[def_col.at]
  if sz ~= "?" and eff ~= "?" and def ~= "?" then
    local score = (eff + def) / sz
    table.insert(candidates, {row=row, score=score})
  end
end

-- Sort ascending by score
table.sort(candidates, function(a,b) return a.score < b.score end)

-- Select top budget rows
local selected = {}
for i=1,math.min(budget, #candidates) do
  table.insert(selected, candidates[i].row)
end

-- Display selected proposals
print(string.format("Suggested %d proposals to evaluate first (budget=%d):", #selected, budget))
print(string.format("%-5s %-6s %-8s %-8s %-10s", "ID", "Size", "Effort", "Defect", "Score"))
for _,row in ipairs(selected) do
  local sz = row[sz_col.at]
  local eff = row[eff_col.at]
  local def = row[def_col.at]
  local score = (eff + def) / sz
  print(string.format("%-5s %-6.0f %-8.1f %-8.1f %-10.3f",
          tostring(row[1]), round(sz), round(eff), round(def), score))
end
