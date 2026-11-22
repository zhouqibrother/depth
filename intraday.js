/* Bitcoin intraday depth. Same one-step update as rebuild/intraday.py.
   Official closes stay on the server; this only scores a hypothetical close. */
function pctIncluding(sorted, x) {
  let lo = 0, hi = sorted.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (sorted[mid] < x) lo = mid + 1;
    else hi = mid;
  }
  const left = lo;
  lo = 0;
  hi = sorted.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (sorted[mid] <= x) lo = mid + 1;
    else hi = mid;
  }
  const right = lo;
  const n = sorted.length + 1;
  return (left + 0.5 * ((right + 1) - left)) / n * 100;
}

function intradayComponents(model, price) {
  const closes = model.closes;
  const alpha = 1 / 14;
  const delta = price - model.prev_close;
  const gain = delta > 0 ? delta : 0;
  const loss = delta < 0 ? -delta : 0;
  const rsiUp = (1 - alpha) * model.rsi_up + alpha * gain;
  const rsiDn = (1 - alpha) * model.rsi_dn + alpha * loss;
  const rsi = rsiDn === 0 ? 100 : 100 - 100 / (1 + rsiUp / rsiDn);
  let maSum = 0;
  const maFrom = closes.length - 199;
  for (let i = maFrom; i < closes.length; i++) maSum += closes[i];
  const ma = (maSum + price) / 200;
  const ratio = (price - ma) / ma;
  const n = model.ratio_n + 1;
  const mean = model.ratio_mean + (ratio - model.ratio_mean) / n;
  const dlt = ratio - model.ratio_mean;
  const m2 = model.ratio_m2 + dlt * (ratio - mean);
  const std = Math.sqrt(m2 / (n - 1));
  const z = (ratio - mean) / std;
  let peak = price;
  const ddFrom = closes.length - 251;
  for (let i = ddFrom; i < closes.length; i++) if (closes[i] > peak) peak = closes[i];
  const dd = price / peak - 1;
  const seq = closes.slice(-20);
  seq.push(price);
  const rets = [];
  for (let i = 1; i < seq.length; i++) rets.push(seq[i] / seq[i - 1] - 1);
  let mu = 0;
  for (let i = 0; i < rets.length; i++) mu += rets[i];
  mu /= rets.length;
  let varSum = 0;
  for (let i = 0; i < rets.length; i++) {
    const d = rets[i] - mu;
    varSum += d * d;
  }
  const vol = Math.sqrt(varSum / (rets.length - 1));
  const mom = price / closes[closes.length - 20] - 1;
  const zp = pctIncluding(model.z, z);
  const dp = pctIncluding(model.dd, dd);
  const vp = 100 - pctIncluding(model.vol, vol);
  const mp = pctIncluding(model.mom, mom);
  const raw = (rsi + zp + dp + vp + mp) / 5;
  const fp = pctIncluding(model.raw, raw);
  return {fp: fp, rsi_up: rsiUp, rsi_dn: rsiDn, ratio: ratio, z: z, dd: dd, vol: vol, mom: mom, raw: raw};
}

function scoreIntraday(model, price) {
  return intradayComponents(model, price).fp;
}

function insortNum(arr, x) {
  let lo = 0, hi = arr.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (arr[mid] < x) lo = mid + 1;
    else hi = mid;
  }
  arr.splice(lo, 0, x);
}

function foldIntraday(model, price, day) {
  const comp = intradayComponents(model, price);
  const closes = model.closes.slice();
  closes.push(price);
  const out = {
    history_through: day,
    prev_close: price,
    closes: closes.length > 252 ? closes.slice(closes.length - 252) : closes,
    rsi_up: comp.rsi_up,
    rsi_dn: comp.rsi_dn,
    ratio_n: model.ratio_n + 1,
    ratio_mean: 0,
    ratio_m2: 0,
    z: model.z.slice(),
    dd: model.dd.slice(),
    vol: model.vol.slice(),
    mom: model.mom.slice(),
    raw: model.raw.slice(),
    min_history: model.min_history
  };
  const n = out.ratio_n;
  out.ratio_mean = model.ratio_mean + (comp.ratio - model.ratio_mean) / n;
  const dlt = comp.ratio - model.ratio_mean;
  out.ratio_m2 = model.ratio_m2 + dlt * (comp.ratio - out.ratio_mean);
  insortNum(out.z, comp.z);
  insortNum(out.dd, comp.dd);
  insortNum(out.vol, comp.vol);
  insortNum(out.mom, comp.mom);
  insortNum(out.raw, comp.raw);
  return out;
}

function blendIntraday(fp, weight, fng) {
  if (fng == null || !(weight > 0)) return fp;
  return fp * (1 - weight) + fng * weight;
}
