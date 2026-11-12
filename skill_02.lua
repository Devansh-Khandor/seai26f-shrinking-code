#!/usr/bin/env lua
-- Skill 02: Estimate missing MPG and weight, with uncertainty
-- Usage: lua skill_02.lua <csv path>

local ezr = require("ezr")
local show, round, keysort = ezr.show, ezr.round, ezr.keysort

-- Helper: find a column object by its name
local function col_by_name(tbl, name)
  for at, col in ipairs(tbl.cols.all) do
    if col.name == name then return col end
  end
  error("column " .. name .. " not found")
end

-- Helper: build a tree for a single numeric target column
-- Y function extracts the raw target value from a row
local function build_tree(tbl, target_col, max_depth)
  local Y = function(row) return row[target_col.at] end
  return ezr.Tree(tbl, tbl.rows, Y, nil, max_depth or 4)
end

-- Helper: walk a tree to the leaf that a row falls into
local function leaf_node(tree, tbl, row)
  local node = tree
  while node.at do
    local col = tbl.cols.all[node.at]
    if col:holds(row[col.at], node.v) then
      node = node.yes
    else
      node = node.no
    end
  end
  return node
end

-- Load CSV and build the table
local path = arg[1]
if not path then
  io.stderr:write("Usage: lua skill_02.lua <csv path>\n")
  os.exit(1)
end
local tbl = ezr.Tbl(ezr.csv(path))

-- === CAR-03: Estimate missing MPG =================================
print("== CAR-03 ==")

local mpg_col = col_by_name(tbl, "Mpg+")
local t_mpg   = build_tree(tbl, mpg_col)

local estimates_mpg = {}
for _, row in ipairs(tbl.rows) do
  if row[mpg_col.at] == "?" or row[mpg_col.at] == nil then
    local node = leaf_node(t_mpg, tbl, row)
    local est  = node.mu
    local uncert = node.loss  -- standard deviation in leaf
    table.insert(estimates_mpg, {row=row, est=est, unc=uncert})
  end
end

-- If no missing MPG, still show predictions for first 5 rows
if #estimates_mpg == 0 then
  for i = 1, math.min(5, #tbl.rows) do
    local row = tbl.rows[i]
    local node = leaf_node(t_mpg, tbl, row)
    table.insert(estimates_mpg, {row=row, est=node.mu, unc=node.loss})
  end
end

print(string.format("%-10s %-6s %-6s", "Model", "MPG Est", "Unc"))
for i, rec in ipairs(estimates_mpg) do
  local model = rec.row[col_by_name(tbl,"Model").at]
  print(string.format("%-10s %-6.2f %-6.2f",
        tostring(model), round(rec.est), round(rec.unc)))
  if i >= 3 then break end
end

-- === CAR-04: Estimate missing Weight =================================
print("\n== CAR-04 ==")

local w_col = col_by_name(tbl, "Lbs-")
local t_wt   = build_tree(tbl, w_col)

local estimates_wt = {}
for _, row in ipairs(tbl.rows) do
  if row[w_col.at] == "?" or row[w_col.at] == nil then
    local node = leaf_node(t_wt, tbl, row)
    local est  = node.mu
    local uncert = node.loss
    table.insert(estimates_wt, {row=row, est=est, unc=uncert})
  end
end

-- If no missing weight, show predictions for first 5 rows
if #estimates_wt == 0 then
  for i = 1, math.min(5, #tbl.rows) do
    local row = tbl.rows[i]
    local node = leaf_node(t_wt, tbl, row)
    table.insert(estimates_wt, {row=row, est=node.mu, unc=node.loss})
  end
end

print(string.format("%-10s %-6s %-6s", "Model", "Weight Est", "Unc"))
for i, rec in ipairs(estimates_wt) do
  local model = rec.row[col_by_name(tbl,"Model").at]
  print(string.format("%-10s %-6.0f %-6.1f",
        tostring(model), round(rec.est), round(rec.unc)))
  if i >= 3 then break end
end
