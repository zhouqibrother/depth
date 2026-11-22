/* Compare the browser scorer with a fixture written by test_intraday.py. */
const fs = require("fs");
const vm = require("vm");
const fixture = JSON.parse(fs.readFileSync(process.argv[2], "utf8"));
const context = {};
vm.createContext(context);
vm.runInContext(fs.readFileSync(require("path").join(__dirname, "..", "intraday.js"), "utf8"), context);
for (const row of fixture.cases) {
  const got = context.scoreIntraday(fixture.model, row.price);
  if (Math.abs(got - row.fp) > 1e-6) {
    console.error("score", row.price, got, row.fp);
    process.exit(1);
  }
}
const folded = context.foldIntraday(fixture.model, fixture.fold.price, fixture.fold.day);
const got = context.scoreIntraday(folded, fixture.fold.then);
if (Math.abs(got - fixture.fold.expect) > 1e-6) {
  console.error("fold", got, fixture.fold.expect);
  process.exit(1);
}
const blended = context.blendIntraday(fixture.cases[0].fp, 0.3, 12);
const py = fixture.cases[0].fp * 0.7 + 12 * 0.3;
if (Math.abs(blended - py) > 1e-9) {
  console.error("blend", blended, py);
  process.exit(1);
}
console.log("ok");
