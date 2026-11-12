#!/usr/bin/env lua
-- Skill 08: NASA-15 & NASA-16
-- Usage: lua skill_08.lua <csv_path>

-- ------------------------------------------------------------------
-- Import ezr and reuse helper functions from previous skills
local ezr = require "ezr"

local show   = ezr.show
local push   = ezr.push
local sorted = ezr.sorted
local map    = ezr.map
local keys   = ezr.keys
local slice  = ezr.slice
local rand   = ezr.rand
local copy   = ezr.copy

-- deterministic random stream
ezr.srand(ezr.the.seed)

-- ------------------------------------------------------------------
-- Common helpers -----------------------------------------------------
local function find_col(tbl, name)
  for _,c in pairs(tbl.cols.all) do
    if c.name == name then return c end
  end
  error("column not found: " .. name)
end

local function median(vals)
  local s = sorted(vals)
  local n = #s
  return s[(n + 1) // 2]
end

local function col_stats(tbl, col)
  local vals = map(tbl.rows, function(r) return r[col.at] end)
  local sorted_vals = sorted(vals)
  local n = #sorted_vals
  local mean = col.mu
  local sd   = col:div()
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

-- k‑NN prediction (copied from skill_04/06)
local function knn_predict(tbl, proposal, goal_col, k)
  local labeled = {}
  for _,r in ipairs(tbl.rows) do
    if r[goal_col.at] ~= "?" then
      local missing = false
      for _,c in ipairs(tbl.cols.x) do
        if r[c.at] == "?" then missing = true; break end
      end
      if not missing then push(labeled, r) end
    end
  end
  if #labeled == 0 then return nil end

  local function dist_to(r) return tbl:distx(proposal, r) end
  local sorted_rows = sorted(labeled, function(a,b)
    return dist_to(a) < dist_to(b)
  end)

  local k_use = math.min(k, #sorted_rows)
  local w_sum, val_sum = 0, 0
  for i=1,k_use do
    local r = sorted_rows[i]
    local d = dist_to(r)
    local w = 1/(d+1e-6)
    w_sum   = w_sum + w
    val_sum = val_sum + w * r[goal_col.at]
  end
  return val_sum / w_sum
end

-- ------------------------------------------------------------------
-- Find min value of a numeric column (excluding missing)
local function numeric_min(tbl, col)
  local mn = nil
  for _,r in ipairs(tbl.rows) do
    local v = r[col.at]
    if type(v) == "number" then
      if mn == nil or v < mn then mn = v end
    end
  end
  return mn
end

-- Find symbolic value that gives lowest average Effort- (or Defects-)
local function best_symbolic_value(tbl, col, goal_col)
  local best_val, best_avg = nil, math.huge
  for v,_ in pairs(col.has) do
    local sum, cnt = 0, 0
    for _,r in ipairs(tbl.rows) do
      if r[col.at] == v and r[goal_col.at] ~= "?" then
        sum = sum + r[goal_col.at]
        cnt = cnt + 1
      end
    end
    if cnt > 0 then
      local avg = sum / cnt
      if avg < best_avg then
        best_avg, best_val = avg, v
      end
    end
  end
  return best_val
end

-- ------------------------------------------------------------------
-- NASA‑15 : Suggest effort‑reducing changes
local function nasa_15(tbl)
  print("== NASA-15 ==")
  local effort_col = find_col(tbl, "Effort-")
  local effort_stats = col_stats(tbl, effort_col)

  -- Find proposals with missing Effort-
  local proposals = {}
  for _,r in ipairs(tbl.rows) do
    if r[effort_col.at] == "?" then
      push(proposals, r)
    end
  end

  if #proposals == 0 then
    print("No proposals with missing effort.")
    return
  end

  -- Predict effort for each proposal
  local threshold = effort_stats.mean + effort_stats.sd
  for _,p in ipairs(proposals) do
    local pred = knn_predict(tbl, p, effort_col, 5)
    if pred and pred > threshold then
      print("\nProposal id "..p[1]..": predicted effort = "..string.format("%.1f",pred))
      local changes = {}

      -- examine each adjustable independent column
      for _,c in ipairs(tbl.cols.x) do
        if c.name ~= "idX" and c.name ~= "YearX" and c ~= effort_col then
          local new_val
          if type(c:mid()) == "number" then
            new_val = numeric_min(tbl, c)
          else
            new_val = best_symbolic_value(tbl, c, effort_col)
          end
          if new_val ~= nil then
            local new_row = copy(p)
            new_row[c.at] = new_val
            local new_pred = knn_predict(tbl, new_row, effort_col, 5)
            if new_pred and new_pred < pred then
              push(changes, {
                col=c.name,
                from=p[c.at],
                to=new_val,
                new_pred=new_pred,
                reduce=pred - new_pred
              })
            end
          end
        end
      end

      if #changes == 0 then
        print("  No single change found that reduces predicted effort.")
      else
        table.sort(changes, function(a,b) return a.reduce > b.reduce end)
        local top = math.min(3,#changes)
        for i=1,top do
          local c = changes[i]
          print(string.format("  Change %s from %s to %s -> new effort %.1f (reduced %.1f)",
            c.col, tostring(c.from), tostring(c.to), c.new_pred, c.reduce))
        end
      end
    end
  end
end

-- ------------------------------------------------------------------
-- NASA‑16 : Suggest defect‑reducing changes
local function nasa_16(tbl)
  print("\n== NASA-16 ==")
  local defect_col = find_col(tbl, "Defects-")
  local defect_stats = col_stats(tbl, defect_col)

  -- Find proposals with missing Defects-
  local proposals = {}
  for _,r in ipairs(tbl.rows) do
    if r[defect_col.at] == "?" then
      push(proposals, r)
    end
  end

  if #proposals == 0 then
    print("No proposals with missing defect count.")
    return
  end

  local threshold = defect_stats.mean + defect_stats.sd
  for _,p in ipairs(proposals) do
    local pred = knn_predict(tbl, p, defect_col, 5)
    if pred and pred > threshold then
      print("\nProposal id "..p[1]..": predicted defects = "..string.format("%.1f",pred))
      local changes = {}

      -- examine each adjustable independent column
      for _,c in ipairs(tbl.cols.x) do
        if c.name ~= "idX" and c.name ~= "YearX" and c ~= defect_col then
          local new_val
          if type(c:mid()) == "number" then
            new_val = numeric_min(tbl, c)
          else
            new_val = best_symbolic_value(tbl, c, defect_col)
          end
          if new_val ~= nil then
            local new_row = copy(p)
            new_row[c.at] = new_val
            local new_pred = knn_predict(tbl, new_row, defect_col, 5)
            if new_pred and new_pred < pred then
              push(changes, {
                col=c.name,
                from=p[c.at],
                to=new_val,
                new_pred=new_pred,
                reduce=pred - new_pred
              })
            end
          end
        end
      end

      if #changes == 0 then
        print("  No single change found that reduces predicted defects.")
      else
        table.sort(changes, function(a,b) return a.reduce > b.reduce end)
        -- take best single change
        local best = changes[1]
        -- try a second change
        local second_best = nil
        if #changes > 1 then
          second_best = changes[2]
        end
        -- compute combined effect
        local combined_row = copy(p)
        combined_row[find_col(tbl, best.col).at] = best.to
        if second_best then
          combined_row[find_col(tbl, second_best.col).at] = second_best.to
        end
        local combined_pred = knn_predict(tbl, combined_row, defect_col, 5)
        local combined_reduce = pred - combined_pred

        print(string.format("  Best change: %s from %s to %s -> new defects %.1f (reduced %.1f)",
          best.col, tostring(best.from), tostring(best.to), best.new_pred, best.reduce))
        if second_best then
          print(string.format("  Second change: %s from %s to %s -> new defects %.1f (reduced %.1f)",
            second_best.col, tostring(second_best.from), tostring(second_best.to),
            second_best.new_pred, second_best.reduce))
        end
        print(string.format("  Combined changes reduce defects by %.1f to %.1f",
          combined_reduce, combined_pred))
      end
    end
  end
end

-- ------------------------------------------------------------------
-- Main
local function main()
  local path = arg[1]
  if not path then
    io.stderr:write("Usage: lua skill_08.lua <csv_path>\n")
    os.exit(1)
  end

  local tbl = ezr.Tbl(ezr.csv(path))

  nasa_15(tbl)
  nasa_16(tbl)
end

main()
