-- Feed two samples to the course's own `same` (vendor/ezr.lua), unchanged.
-- stdin: two lines of space-separated numbers. stdout: true | false.
-- Run with LUA_PATH pointing at harness/vendor (analyze.py does this).
local ezr = require("ezr")
local function nums(line)
  local t = {}
  for x in (line or ""):gmatch("%S+") do t[#t + 1] = tonumber(x) end
  table.sort(t)
  return t
end
local xs, ys = nums(io.read("l")), nums(io.read("l"))
print(tostring(ezr.same(xs, ys)))
