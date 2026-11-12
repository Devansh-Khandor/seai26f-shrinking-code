#!/usr/bin/env lua
-- Skill 02 – Apache performance estimation & uncertainty analysis
--
-- Usage:  lua skill_02.lua <csv path>
--
-- Requires the ezr library (ezr.lua, ezr-lib.lua) in the same directory.

local ezr = require("ezr")

-- Find the leaf node for a row in a tree
local function leaf_node(node, tbl, row)
  while node.at do
    local col = tbl.cols.all[node.at]
    local go = col:holds(row[col.at], node.v)
    node = go and node.lo or node.hi
  end
  return node
end

-- Compute Hamming distance between two rows (exclude goal column)
local function dist1(row1, row2, goal_at)
  local d = 0
  for _, c in ipairs(row1) do
    if c ~= nil then
      local at = c.at
      if at ~= goal_at and row1[at] ~= row2[at] then
        d = d + 1
        if d > 1 then return d end
      end
    end
  end
  return d
end

local function main(csv_path)
  if not csv_path then
    print("Usage: lua skill_02.lua <csv path>")
    os.exit(1)
  end

  -- Load data
  local tbl = ezr.Tbl(ezr.csv(csv_path))
  local goal_col = tbl.cols.y[1]
  local goal_at = goal_col.at

  -- Build a full tree for predictions
  local tree = ezr.Tree(tbl, tbl.rows)

  -- ===== APACHE-03 =====
  print("== APACHE-03 ==")
  local max_show = math.min(#tbl.rows, 5)
  for i = 1, max_show do
    local row = tbl.rows[i]
    local pred = tree:leaf(tbl, row)
    local actual = row[goal_at]
    print(string.format("Row %d: predicted = %s, actual = %s",
            i, ezr.show(pred), ezr.show(actual)))
  end
  print()

  -- ===== APACHE-04 =====
  print("== APACHE-04 ==")
  local pairs_shown = 0
  for i = 1, #tbl.rows-1 do
    for j = i+1, #tbl.rows do
      if dist1(tbl.rows[i], tbl.rows[j], goal_at) == 1 then
        local node_i = leaf_node(tree, tbl, tbl.rows[i])
        local node_j = leaf_node(tree, tbl, tbl.rows[j])
        local var_i = node_i.here.cols.y[1]:div()
        local var_j = node_j.here.cols.y[1]:div()
        local uncert = math.sqrt(var_i*var_i + var_j*var_j)
        local pred_diff = node_j.mu - node_i.mu
        local actual_diff = tbl.rows[j][goal_at] - tbl.rows[i][goal_at]
        print(string.format("Pair (%d,%d): predicted Δ=%.2f, actual Δ=%.2f, uncertainty=±%.2f",
                i, j, pred_diff, actual_diff, uncert))
        pairs_shown = pairs_shown + 1
        if pairs_shown >= 3 then
          return
        end
      end
    end
  end
end

main(arg[1])
