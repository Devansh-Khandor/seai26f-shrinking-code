#!/usr/bin/env lua
-- Skill 08 – Alternative vehicle suggestions for poorly rated cars
-- Usage:  lua skill_08.lua <csv_path>

local ezr = require("ezr")
local round = ezr.round
local sorted = ezr.sorted

-- ------------------------------------------------------------------
-- Utility: find a column object by name
local function get_col(tbl, name)
  for _, c in pairs(tbl.cols.all) do
    if c.name == name then return c end
  end
  error("column not found: " .. tostring(name))
end

-- ------------------------------------------------------------------
-- Normalisation helper (0..1)
local function norm(x, xmin, xmax)
  if xmax == xmin then return 0.5 end
  return (x - xmin) / (xmax - xmin)
end

-- ------------------------------------------------------------------
-- Median of a numeric list
local function median(arr)
  if #arr == 0 then return nil end
  local s = sorted(arr)
  local idx = math.floor(#s / 2) + 1
  return s[idx]
end

-- ------------------------------------------------------------------
-- Composite performance score
-- Lower weight, lower acceleration, higher mpg
local function comp_score(row, minW, maxW, minA, maxA, minM, maxM)
  local w = row[weight_col.at]
  local a = row[acc_col.at]
  local m = row[mpg_col.at]
  return (1 - norm(w, minW, maxW)) +
         (1 - norm(a, minA, maxA)) +
         norm(m, minM, maxM)
end

-- ------------------------------------------------------------------
-- Read CSV
local csv_path = arg[1]
if not csv_path then
  io.stderr:write("Error: missing CSV path\n")
  os.exit(1)
end

local tbl = ezr.Tbl(ezr.csv(csv_path))

-- ------------------------------------------------------------------
-- Identify metrics of interest
local weight_col = get_col(tbl, "Lbs-")
local acc_col    = get_col(tbl, "Acc+")
local mpg_col    = get_col(tbl, "Mpg+")
local model_col  = get_col(tbl, "Model")

-- ------------------------------------------------------------------
-- Compute global min/max for normalisation
local minW, maxW = math.huge, -math.huge
local minA, maxA = math.huge, -math.huge
local minM, maxM = math.huge, -math.huge

-- Also keep lists to compute medians later
local w_vals, a_vals, m_vals = {}, {}, {}

for _, row in ipairs(tbl.rows) do
  local w = row[weight_col.at]
  local a = row[acc_col.at]
  local m = row[mpg_col.at]
  if type(w) == "number" then
    minW = math.min(minW, w)
    maxW = math.max(maxW, w)
    table.insert(w_vals, w)
  end
  if type(a) == "number" then
    minA = math.min(minA, a)
    maxA = math.max(maxA, a)
    table.insert(a_vals, a)
  end
  if type(m) == "number" then
    minM = math.min(minM, m)
    maxM = math.max(maxM, m)
    table.insert(m_vals, m)
  end
end

-- ------------------------------------------------------------------
-- Compute score for each row and store it
for _, row in ipairs(tbl.rows) do
  row._score = comp_score(row, minW, maxW, minA, maxA, minM, maxM)
end

-- ------------------------------------------------------------------
-- Find the worst vehicle (lowest score)
local worst
for _, row in ipairs(tbl.rows) do
  if not worst or row._score < worst._score then
    worst = row
  end
end

-- ------------------------------------------------------------------
-- Compute thresholds (median) for each metric
local w_med = median(w_vals)
local a_med = median(a_vals)
local m_med = median(m_vals)

-- ------------------------------------------------------------------
-- CAR-15: Alternatives with similar characteristics but better score
print("== CAR-15 ==")

-- Find alternatives sorted by distance from worst
local alternatives = {}
for _, row in ipairs(tbl.rows) do
  if row ~= worst and row._score > worst._score then
    local d = tbl:distx(worst, row)
    table.insert(alternatives, {row=row, dist=d})
  end
end
table.sort(alternatives, function(a,b) return a.dist < b.dist end)

local n_out = math.min(3, #alternatives)
for i=1,n_out do
  local alt = alternatives[i].row
  print(string.format(
    "%d) Model=%s | Score=%.2f | Weight=%d | Acc=%.1f | MPG=%.2f",
    i, alt[model_col.at], round(alt._score,2),
    alt[weight_col.at], alt[acc_col.at], alt[mpg_col.at]))
  print(string.format(
    "   Weight diff=%.1f kg | Acc diff=%.1f m/s² | MPG diff=%.1f mpg",
    alt[weight_col.at]-worst[weight_col.at],
    alt[acc_col.at]-worst[acc_col.at],
    alt[mpg_col.at]-worst[mpg_col.at]))
  print()
end

-- ------------------------------------------------------------------
-- CAR-16: Find closest alternative that meets more targets
print("== CAR-16 ==")

-- Define target thresholds (median values)
local targets = {
  weight = w_med,
  acc    = a_med,
  mpg    = m_med
}

-- Identify a vehicle that does NOT meet at least one target
local candidate
for _, row in ipairs(tbl.rows) do
  if row ~= worst then
    if row[weight_col.at] > targets.weight or
       row[acc_col.at]   > targets.acc    or
       row[mpg_col.at]   < targets.mpg then
      candidate = row
      break
    end
  end
end

-- If none found (unlikely), fallback to worst
if not candidate then candidate = worst end

-- Find alternative with minimal distance that satisfies at least 2 of 3 targets
local best_alt
for _, row in ipairs(tbl.rows) do
  if row ~= candidate then
    local met = 0
    if row[weight_col.at] <= targets.weight then met = met + 1 end
    if row[acc_col.at]    <= targets.acc    then met = met + 1 end
    if row[mpg_col.at]    >= targets.mpg    then met = met + 1 end
    if met >= 2 then
      local d = tbl:distx(candidate, row)
      if not best_alt or d < best_alt.dist then
        best_alt = {row=row, dist=d, met=met}
      end
    end
  end
end

if best_alt then
  local alt = best_alt.row
  print(string.format(
    "Recommended alternative to %s:",
    candidate[model_col.at]))
  print(string.format(
    "Model=%s | Score=%.2f | Weight=%d | Acc=%.1f | MPG=%.2f",
    alt[model_col.at], round(alt._score,2),
    alt[weight_col.at], alt[acc_col.at], alt[mpg_col.at]))
  print(string.format(
    "Weight diff=%.1f kg | Acc diff=%.1f m/s² | MPG diff=%.1f mpg",
    alt[weight_col.at]-candidate[weight_col.at],
    alt[acc_col.at]-candidate[acc_col.at],
    alt[mpg_col.at]-candidate[mpg_col.at]))
  print("Targets met:", best_alt.met .. "/3")
else
  print("No suitable alternative found that improves on at least two targets.")
end
