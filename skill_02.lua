-- skill_02.lua
-- Estimates missing fuel‑efficiency (Mpg+) and vehicle weight (Lbs-)
-- from the other attributes in the inventory.
--
-- Usage: lua skill_02.lua <path-to-auto93.csv>
--
-- Prints:
--   == CAR-03 ==
--   3+ lines of results for fuel‑efficiency estimates
--   == CAR-04 ==
--   3+ lines of results for weight estimates

local ezr = require("ezr")   -- use the bundled csv reader
local csv = ezr.csv

-- --------------------------------------------------------------
-- Utility matrix operations -----------------------------------
local function mat_transpose(A)
  local n, p = #A, #A[1]
  local B = {}
  for i = 1, p do
    B[i] = {}
    for j = 1, n do
      B[i][j] = A[j][i]
    end
  end
  return B
end

local function mat_mult(A, B)
  local n, m, p = #A, (#B[1] or 1), #B
  local C = {}
  for i = 1, n do
    C[i] = {}
    for j = 1, m do
      local sum = 0
      for k = 1, p do
        sum = sum + A[i][k] * B[k][j]
      end
      C[i][j] = sum
    end
  end
  return C
end

-- Gaussian‑Jordan inverse (for small matrices)
local function mat_inverse(A)
  local n = #A
  local M = {}
  for i = 1, n do
    M[i] = {}
    for j = 1, n do
      M[i][j] = A[i][j]
    end
  end

  -- Augment with identity
  for i = 1, n do
    for j = 1, n do
      M[i][n + j] = (i == j) and 1 or 0
    end
  end

  -- Forward elimination
  for i = 1, n do
    local piv = M[i][i]
    if math.abs(piv) < 1e-12 then
      local swap = nil
      for k = i + 1, n do
        if math.abs(M[k][i]) > 1e-12 then
          swap = k
          break
        end
      end
      if not swap then error("Singular matrix") end
      M[i], M[swap] = M[swap], M[i]
      piv = M[i][i]
    end
    for j = i, 2 * n do
      M[i][j] = M[i][j] / piv
    end
    for k = 1, n do
      if k ~= i then
        local factor = M[k][i]
        if factor ~= 0 then
          for j = i, 2 * n do
            M[k][j] = M[k][j] - factor * M[i][j]
          end
        end
      end
    end
  end

  local Inv = {}
  for i = 1, n do
    Inv[i] = {}
    for j = 1, n do
      Inv[i][j] = M[i][n + j]
    end
  end
  return Inv
end

-- --------------------------------------------------------------
-- Build linear‑regression coefficients
local function build_beta(rows, target_index)
  local X = {}
  local y = {}
  for _, row in ipairs(rows) do
    local tgt = row[target_index]
    if tgt ~= "?" and tgt ~= nil then
      local feat = {1}          -- bias term
      local ok = true
      for j = 1, #row do
        if j ~= target_index then
          local v = row[j]
          if v == "?" or v == nil then
            ok = false
            break
          end
          table.insert(feat, v)
        end
      end
      if ok then
        table.insert(X, feat)
        table.insert(y, tgt)
      end
    end
  end
  if #X < 2 then error("Not enough data to build model") end

  -- Compute (XᵀX)⁻¹ Xᵀy
  local XtX = {}
  local p = #X[1]
  for i = 1, p do
    XtX[i] = {}
    for j = 1, p do
      local sum = 0
      for r = 1, #X do
        sum = sum + X[r][i] * X[r][j]
      end
      XtX[i][j] = sum
    end
  end

  local Xty = {}
  for i = 1, p do
    local sum = 0
    for r = 1, #X do
      sum = sum + X[r][i] * y[r]
    end
    Xty[i] = {sum}
  end

  local invXtX = mat_inverse(XtX)
  local beta = mat_mult(invXtX, Xty)
  local beta_list = {}
  for i = 1, p do
    beta_list[i] = beta[i][1]
  end
  return beta_list
end

-- --------------------------------------------------------------
-- Predict a target value for a row using beta coefficients
local function predict(row, beta, target_index)
  local val = beta[1]            -- bias
  local j = 2
  for i = 1, #row do
    if i ~= target_index then
      local v = row[i]
      if v == "?" or v == nil then return nil end
      val = val + beta[j] * v
      j = j + 1
    end
  end
  return val
end

-- --------------------------------------------------------------
-- Main -----------------------------------------------------------
local function main(csv_path)
  local all_rows = {}
  for r in csv(csv_path) do
    table.insert(all_rows, r)
  end
  if #all_rows < 2 then
    error("CSV must contain a header and at least one data row")
  end

  local header = all_rows[1]
  table.remove(all_rows, 1)  -- remove header

  local idx_mpg, idx_lbs = nil, nil
  for i, name in ipairs(header) do
    if name == "Mpg+" then idx_mpg = i end
    if name == "Lbs-" then idx_lbs = i end
  end
  if not idx_mpg or not idx_lbs then
    error("CSV missing required columns: Mpg+ and Lbs-")
  end

  -- Build regression models
  local beta_mpg = build_beta(all_rows, idx_mpg)
  local beta_lbs = build_beta(all_rows, idx_lbs)

  -- Fuel‑efficiency estimation
  print("== CAR-03 ==")
  for i = 1, math.min(3, #all_rows) do
    local row = all_rows[i]
    local actual = row[idx_mpg]
    local pred = predict(row, beta_mpg, idx_mpg)
    print(string.format("Row %d: actual mpg=%s, predicted mpg=%.3f",
                        i, tostring(actual), pred))
  end

  -- Weight estimation
  print("\n== CAR-04 ==")
  for i = 1, math.min(3, #all_rows) do
    local row = all_rows[i]
    local actual = row[idx_lbs]
    local pred = predict(row, beta_lbs, idx_lbs)
    print(string.format("Row %d: actual lbs=%s, predicted lbs=%.3f",
                        i, tostring(actual), pred))
  end
end

if arg[1] then
  main(arg[1])
else
  io.stderr:write("Usage: lua skill_02.lua <csv_path>\n")
end
