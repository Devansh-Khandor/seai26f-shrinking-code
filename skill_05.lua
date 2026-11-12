#!/usr/bin/env lua
-- Skill 05: NASA-09 & NASA-10 – High effort and high defect drivers
-- Usage: lua skill_05.lua <csv_path>

local ezr   = require("ezr-lib")
local csv   = ezr.csv
local round = ezr.round
local show  = ezr.show

-- ---------- helper: compute mean and sd ------------------------------------
local function mean(t)
  local n = #t
  if n == 0 then return 0 end
  local sum = 0
  for _,v in ipairs(t) do sum = sum + v end
  return sum / n
end

local function sd(t, m)
  local n = #t
  if n == 0 then return 0 end
  m = m or mean(t)
  local s = 0
  for _,v in ipairs(t) do s = s + (v - m)^2 end
  return math.sqrt(s / n)
end

-- ---------- read all rows --------------------------------------------------
local path = arg[1]
if not path then
  io.stderr:write("Usage: lua skill_05.lua <csv_path>\n")
  os.exit(1)
end

local header, rows = nil, {}
for rec in csv(path) do
  if not header then
    header = rec
  else
    rows[#rows + 1] = rec
  end
end

local nrows = #rows
if nrows == 0 then
  print("No data rows found.")
  os.exit(0)
end

-- ---------- discover column types and gather overall stats -----------------
local colstats = {}   -- {name, numeric, nums={}, syms={}, mode=nil, mean=nil}
for coli, name in ipairs(header) do
  local col = {name=name, numeric=false, nums={}, syms={}}
  for _,row in ipairs(rows) do
    local val = row[coli]
    local num = tonumber(val)
    if num then
      col.numeric = true
      col.nums[#col.nums+1] = num
    else
      col.syms[val] = (col.syms[val] or 0) + 1
    end
  end
  if col.numeric then
    col.mean = mean(col.nums)
    col.sd   = sd(col.nums, col.mean)
  end
  -- mode
  local bestk, bestc = nil, 0
  for k,c in pairs(col.syms) do
    if c > bestc then bestk, bestc = k, c end
  end
  col.mode = bestk
  colstats[coli] = col
end

-- ---------- compute Effort and Defects overall stats -----------------------
local effort_col  = nil
local defects_col = nil
for i,c in ipairs(colstats) do
  if c.name == "Effort-" then effort_col = c end
  if c.name == "Defects-" then defects_col = c end
end
if not effort_col or not defects_col then
  print("Required columns Effort- or Defects- missing.")
  os.exit(1)
end

local eff_mean  = effort_col.mean
local eff_sd    = effort_col.sd
local def_mean  = defects_col.mean
local def_sd    = defects_col.sd

-- ---------- identify high‐effort and high‐defect projects ------------------
local highEff = {}
local highDef = {}
for _,row in ipairs(rows) do
  local eff = tonumber(row[effort_col.at]) or eff_col.nums[#eff_col.nums]
  local def = tonumber(row[defects_col.at]) or defects_col.nums[#defects_col.nums]
  if eff and eff > eff_mean + eff_sd then
    highEff[#highEff+1] = row
  end
  if def and def > def_mean + def_sd then
    highDef[#highDef+1] = row
  end
end

-- ---------- compute per‑column differences --------------------------------
local function column_differences(highset)
  local diff = {}
  for i,col in ipairs(colstats) do
    if col.numeric then
      local nums = {}
      for _,row in ipairs(highset) do
        local v = tonumber(row[i])
        if v then nums[#nums+1] = v end
      end
      if #nums > 0 then
        local hmean = mean(nums)
        diff[i] = {name=col.name, type="num",
                   diff=hmean - col.mean,
                   hmean=hmean, omean=col.mean}
      end
    else
      local counts = {}
      local total = #highset
      for _,row in ipairs(highset) do
        local v = row[i]
        counts[v] = (counts[v] or 0) + 1
      end
      local hmode, hcnt = nil, 0
      for k,c in pairs(counts) do
        if c > hcnt then hmode, hcnt = k, c end
      end
      local omode, ocnt = col.mode, col.syms[col.mode]
      local prop_h = hcnt / total
      local prop_o = ocnt / nrows
      diff[i] = {name=col.name, type="sym",
                 diff=math.abs(prop_h - prop_o),
                 hmode=hmode, omode=omode}
    end
  end
  return diff
end

local effDiff  = column_differences(highEff)
local defDiff  = column_differences(highDef)

-- ---------- sort and print -----------------------------------------------
local function sort_by_diff(d)
  local arr = {}
  for _,v in pairs(d) do table.insert(arr, v) end
  table.sort(arr, function(a,b)
    return math.abs(a.diff) > math.abs(b.diff) end)
  return arr
end

local function print_section(id, title, diffs, highset, overall, metric_name)
  print("== " .. id .. " ==")
  print(string.format("Identified %d high %s projects out of %d rows.",
         #highset, metric_name, nrows))
  print("Top characteristics most associated (sorted by absolute difference):")
  local sorted = sort_by_diff(diffs)
  local limit = math.min(5, #sorted)
  for i=1,limit do
    local c = sorted[i]
    if c.type == "num" then
      print(string.format("%d. %s: high mean=%.2f, overall mean=%.2f, diff=%.2f",
             i, c.name, round(c.hmean), round(c.omean), round(c.diff)))
    else
      print(string.format("%d. %s: high mode=%s, overall mode=%s, diff=%.2f",
             i, c.name, show(c.hmode), show(c.omode), round(c.diff)))
    end
  end
  print()
end

print_section("NASA-09","High effort", effDiff, highEff, effort_col, "effort")
print_section("NASA-10","High defects", defDiff, highDef, defects_col, "defects")
