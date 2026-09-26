//! Pure checks over a run summary and a training-metrics stream.
//! No I/O here so every rule is unit-testable.
use serde::Deserialize;
use std::collections::BTreeMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Level {
    Pass,
    Warn,
    Fail,
}

#[derive(Debug)]
pub struct Finding {
    pub level: Level,
    pub rule: &'static str,
    pub msg: String,
}

fn f(level: Level, rule: &'static str, msg: String) -> Finding {
    Finding { level, rule, msg }
}

// ---------------------------------------------------------------- config
#[derive(Debug, Deserialize, Clone)]
pub struct RunCfg {
    pub max_fold_gap: f64,
    pub max_fold_std: f64,
    pub too_good: f64,
    pub ceiling_slack: f64,
    pub max_top_feature_share: f64,
    pub max_country_gap: f64,
    pub max_singleton_drift: f64,
    pub max_test_singleton_drift: f64,
    pub links_ratio_min: f64,
    pub links_ratio_max: f64,
    pub champion_tolerance: f64,
    pub banned_feature_patterns: Vec<String>,
}

#[derive(Debug, Deserialize, Clone)]
pub struct WatchCfg {
    pub patience: usize,
    pub max_loss_ratio: f64,
    pub warmup_evals: usize,
    pub poll_ms: u64,
}

#[derive(Debug, Deserialize, Clone)]
pub struct SubCfg {
    pub partition_violation_is_fail: bool,
}

#[derive(Debug, Deserialize, Clone)]
pub struct Config {
    pub run: RunCfg,
    pub watch: WatchCfg,
    pub submission: SubCfg,
}

// ---------------------------------------------------------------- summary
/// Written by src/runlog.py at the end of a training run (schema: docs/master-plan/MLGUARD.md).
#[derive(Debug, Deserialize, Default)]
pub struct Fold {
    pub fold: u32,
    pub train_score: f64,
    pub valid_score: f64,
    #[serde(default)]
    pub best_iter: Option<u32>,
    #[serde(default)]
    pub max_iter: Option<u32>,
}

#[derive(Debug, Deserialize, Default)]
pub struct Summary {
    pub run_id: String,
    pub oof_score: f64,
    pub threshold_source: String,
    pub folds: Vec<Fold>,
    #[serde(default)]
    pub blocking_recall: Option<f64>,
    #[serde(default)]
    pub per_country: BTreeMap<String, f64>,
    #[serde(default)]
    pub oof_pred_singleton_rate: Option<f64>,
    #[serde(default)]
    pub true_singleton_rate: Option<f64>,
    #[serde(default)]
    pub oof_pred_links_per_entity: Option<f64>,
    #[serde(default)]
    pub test_pred_singleton_rate: BTreeMap<String, f64>,
    #[serde(default)]
    pub test_pred_links_per_entity: BTreeMap<String, f64>,
    #[serde(default)]
    pub feature_importance: BTreeMap<String, f64>,
}

/// Best macro F0.5 a perfect matcher can reach when blocking recall is R.
pub fn f05_ceiling(recall: f64) -> f64 {
    1.25 * recall / (0.25 + recall)
}

fn mean_std(xs: &[f64]) -> (f64, f64) {
    let n = xs.len() as f64;
    let m = xs.iter().sum::<f64>() / n;
    let v = xs.iter().map(|x| (x - m).powi(2)).sum::<f64>() / n;
    (m, v.sqrt())
}

pub fn check_run(s: &Summary, c: &RunCfg, champion: Option<&Summary>) -> Vec<Finding> {
    use Level::*;
    let mut out = Vec::new();

    // threshold must be tuned out-of-fold, never on in-sample predictions
    if !matches!(s.threshold_source.as_str(), "oof" | "holdout") {
        out.push(f(Fail, "threshold_source", format!(
            "threshold tuned on '{}' — must be 'oof' or 'holdout'", s.threshold_source)));
    }

    if s.folds.is_empty() {
        out.push(f(Fail, "folds", "no fold results — cannot assess overfitting".into()));
    } else {
        for fd in &s.folds {
            let gap = fd.train_score - fd.valid_score;
            if gap > c.max_fold_gap {
                out.push(f(Fail, "overfit_gap", format!(
                    "fold {}: train {:.4} vs valid {:.4} (gap {:.4} > {})",
                    fd.fold, fd.train_score, fd.valid_score, gap, c.max_fold_gap)));
            }
            if let (Some(b), Some(m)) = (fd.best_iter, fd.max_iter) {
                if b >= m {
                    out.push(f(Warn, "no_early_stop", format!(
                        "fold {}: best_iter {} hit the cap {} — underfit or early stopping not wired", fd.fold, b, m)));
                } else if b < 20 {
                    out.push(f(Warn, "early_collapse", format!(
                        "fold {}: best_iter {} — model stopped almost immediately", fd.fold, b)));
                }
            }
        }
        let v: Vec<f64> = s.folds.iter().map(|x| x.valid_score).collect();
        let (_, sd) = mean_std(&v);
        if sd > c.max_fold_std {
            out.push(f(Fail, "fold_instability", format!(
                "fold valid std {:.4} > {} — split or model unstable", sd, c.max_fold_std)));
        }
    }

    if s.oof_score > c.too_good {
        out.push(f(Fail, "too_good", format!(
            "OOF {:.4} > {} — treat as leakage until explained", s.oof_score, c.too_good)));
    }
    if let Some(r) = s.blocking_recall {
        let ceil = f05_ceiling(r);
        if s.oof_score > ceil + c.ceiling_slack {
            out.push(f(Fail, "beats_ceiling", format!(
                "OOF {:.4} exceeds blocking ceiling {:.4} (recall {:.4}) — impossible without leakage",
                s.oof_score, ceil, r)));
        }
    }

    // features: banned names and single-feature dominance
    for name in s.feature_importance.keys() {
        let low = name.to_lowercase();
        if let Some(p) = c.banned_feature_patterns.iter().find(|p| low.contains(p.as_str())) {
            out.push(f(Fail, "banned_feature", format!("feature '{}' matches banned pattern '{}'", name, p)));
        }
    }
    let total: f64 = s.feature_importance.values().sum();
    if total > 0.0 {
        if let Some((k, v)) = s.feature_importance.iter().max_by(|a, b| a.1.total_cmp(b.1)) {
            if v / total > c.max_top_feature_share {
                out.push(f(Warn, "feature_dominance", format!(
                    "'{}' holds {:.0}% of total gain", k, 100.0 * v / total)));
            }
        }
    }

    for (country, sc) in &s.per_country {
        if s.oof_score - sc > c.max_country_gap {
            out.push(f(Warn, "country_gap", format!(
                "{}: {:.4} vs overall {:.4}", country, sc, s.oof_score)));
        }
    }

    if let (Some(p), Some(t)) = (s.oof_pred_singleton_rate, s.true_singleton_rate) {
        if (p - t).abs() > c.max_singleton_drift {
            out.push(f(Warn, "singleton_rate", format!(
                "OOF predicted singleton rate {:.3} vs true {:.3}", p, t)));
        }
    }
    out.extend(drift_checks(s, c));

    if let Some(ch) = champion {
        if s.oof_score < ch.oof_score - c.champion_tolerance {
            out.push(f(Fail, "champion_regression", format!(
                "OOF {:.4} below champion '{}' {:.4}", s.oof_score, ch.run_id, ch.oof_score)));
        }
    }
    out
}

/// Label-free drift on test: an unseen country (France) collapsing to "no match"
/// or exploding into over-merging shows up here before the leaderboard does.
pub fn drift_checks(s: &Summary, c: &RunCfg) -> Vec<Finding> {
    let mut out = Vec::new();
    if let Some(p) = s.oof_pred_singleton_rate {
        for (country, r) in &s.test_pred_singleton_rate {
            if (r - p).abs() > c.max_test_singleton_drift {
                out.push(f(Level::Fail, "test_singleton_drift", format!(
                    "test {}: predicted singleton rate {:.3} vs OOF {:.3}", country, r, p)));
            }
        }
    }
    if let Some(l) = s.oof_pred_links_per_entity {
        for (country, t) in &s.test_pred_links_per_entity {
            let ratio = t / l;
            if ratio < c.links_ratio_min || ratio > c.links_ratio_max {
                out.push(f(Level::Fail, "test_links_drift", format!(
                    "test {}: {:.2} links/entity vs OOF {:.2} (ratio {:.2})", country, t, l, ratio)));
            }
        }
    }
    out
}

// ---------------------------------------------------------------- watch
/// One line of metrics.jsonl written during training.
#[derive(Debug, Deserialize)]
pub struct Tick {
    #[serde(default)]
    pub fold: u32,
    #[serde(default)]
    pub iter: u32,
    #[serde(default)]
    pub train_loss: Option<f64>,
    #[serde(default)]
    pub valid_loss: Option<f64>,
    #[serde(default)]
    pub event: Option<String>,
}

/// Streaming overfit detector. Feed ticks in order; it reports the first violation per fold.
pub struct Watcher {
    cfg: WatchCfg,
    fold: Option<u32>,
    evals: usize,
    prev: Option<(f64, f64)>,
    best_valid: f64,
    rising: usize,
    tripped: bool,
}

impl Watcher {
    pub fn new(cfg: WatchCfg) -> Self {
        Watcher { cfg, fold: None, evals: 0, prev: None, best_valid: f64::INFINITY, rising: 0, tripped: false }
    }

    pub fn feed(&mut self, t: &Tick) -> Option<Finding> {
        if self.fold != Some(t.fold) {
            // new fold: reset state
            self.fold = Some(t.fold);
            self.evals = 0;
            self.prev = None;
            self.best_valid = f64::INFINITY;
            self.rising = 0;
            self.tripped = false;
        }
        let (tr, va) = match (t.train_loss, t.valid_loss) {
            (Some(a), Some(b)) => (a, b),
            _ => return None,
        };
        if !tr.is_finite() || !va.is_finite() {
            return Some(f(Level::Fail, "nan_loss", format!(
                "fold {} iter {}: non-finite loss (train {}, valid {})", t.fold, t.iter, tr, va)));
        }
        self.evals += 1;
        let mut hit = None;
        if let Some((ptr, pva)) = self.prev {
            if va > pva && tr < ptr {
                self.rising += 1;
            } else {
                self.rising = 0;
            }
            if self.rising >= self.cfg.patience && !self.tripped {
                hit = Some(f(Level::Fail, "diverging_valid", format!(
                    "fold {} iter {}: valid loss rose {} evals in a row while train loss fell",
                    t.fold, t.iter, self.rising)));
            }
        }
        // A train/valid loss ratio alone is not overfitting: in boosting both losses head to
        // zero and the ratio keeps growing while valid still improves (run 003 tripped at
        // 0.035/0.022 = 1.5 with valid F0.5 still rising). Only flag it once valid has also
        // stopped improving — memorization with no generalization gain.
        let stalled = va > self.best_valid * 1.02;
        self.best_valid = self.best_valid.min(va);
        if hit.is_none() && self.evals > self.cfg.warmup_evals && tr > 0.0
            && va / tr > self.cfg.max_loss_ratio && stalled && !self.tripped
        {
            hit = Some(f(Level::Fail, "loss_ratio", format!(
                "fold {} iter {}: valid/train loss {:.2} > {}", t.fold, t.iter, va / tr, self.cfg.max_loss_ratio)));
        }
        self.prev = Some((tr, va));
        if hit.is_some() {
            self.tripped = true;
        }
        hit
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> Config {
        toml::from_str(include_str!("../mlguard.toml")).unwrap()
    }

    fn good() -> Summary {
        Summary {
            run_id: "good".into(),
            oof_score: 0.90,
            threshold_source: "oof".into(),
            folds: (0..5).map(|i| Fold { fold: i, train_score: 0.91, valid_score: 0.90 + 0.001 * i as f64,
                                         best_iter: Some(500), max_iter: Some(2000) }).collect(),
            blocking_recall: Some(0.95),
            oof_pred_singleton_rate: Some(0.07),
            true_singleton_rate: Some(0.056),
            oof_pred_links_per_entity: Some(3.2),
            ..Default::default()
        }
    }

    fn fails(v: &[Finding]) -> Vec<&'static str> {
        v.iter().filter(|x| x.level == Level::Fail).map(|x| x.rule).collect()
    }

    #[test]
    fn clean_run_passes() {
        assert!(fails(&check_run(&good(), &cfg().run, None)).is_empty());
    }

    #[test]
    fn catches_overfit_leak_and_drift() {
        let c = cfg().run;
        let mut s = good();
        s.folds[0].train_score = 0.99;
        assert!(fails(&check_run(&s, &c, None)).contains(&"overfit_gap"));

        let mut s = good();
        s.oof_score = 0.9999;
        let r = fails(&check_run(&s, &c, None));
        assert!(r.contains(&"too_good") && r.contains(&"beats_ceiling"));

        let mut s = good();
        s.threshold_source = "train".into();
        assert!(fails(&check_run(&s, &c, None)).contains(&"threshold_source"));

        let mut s = good();
        s.feature_importance.insert("s1_entity_id_num".into(), 1.0);
        assert!(fails(&check_run(&s, &c, None)).contains(&"banned_feature"));

        let mut s = good();
        s.test_pred_singleton_rate.insert("France".into(), 0.60);
        s.test_pred_links_per_entity.insert("France".into(), 1.0);
        let r = fails(&check_run(&s, &c, None));
        assert!(r.contains(&"test_singleton_drift") && r.contains(&"test_links_drift"));

        let champ = Summary { run_id: "champ".into(), oof_score: 0.93, ..Default::default() };
        assert!(fails(&check_run(&good(), &c, Some(&champ))).contains(&"champion_regression"));
    }

    #[test]
    fn ceiling_formula() {
        assert!((f05_ceiling(0.9499) - 0.990).abs() < 1e-3);
        assert!((f05_ceiling(1.0) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn watcher_trips_on_divergence_and_nan() {
        let mut w = Watcher::new(cfg().watch);
        let mut tripped = None;
        for i in 0..30u32 {
            let (tr, va) = if i < 15 { (1.0 - 0.02 * i as f64, 1.0 - 0.02 * i as f64) }
                           else { (0.7 - 0.01 * i as f64, 0.70 + 0.002 * (i - 15) as f64) };
            if let Some(x) = w.feed(&Tick { fold: 0, iter: i, train_loss: Some(tr), valid_loss: Some(va), event: None }) {
                tripped = Some(x.rule);
                break;
            }
        }
        assert_eq!(tripped, Some("diverging_valid"));

        let mut w = Watcher::new(cfg().watch);
        let x = w.feed(&Tick { fold: 1, iter: 0, train_loss: Some(f64::NAN), valid_loss: Some(0.5), event: None });
        assert_eq!(x.map(|x| x.rule), Some("nan_loss"));
    }

    #[test]
    fn ratio_alone_is_not_overfitting() {
        let mut w = Watcher::new(cfg().watch);
        for i in 0..200u32 {
            let tr = 0.3 / (1.0 + i as f64 * 0.2);      // -> 0.0075
            let va = 0.03 + 0.3 / (1.0 + i as f64 * 0.1); // still improving, ratio grows past 4
            assert!(w.feed(&Tick { fold: 0, iter: i, train_loss: Some(tr), valid_loss: Some(va), event: None }).is_none(), "iter {}", i);
        }
    }

    #[test]
    fn watcher_quiet_on_healthy_curve() {
        let mut w = Watcher::new(cfg().watch);
        for i in 0..200u32 {
            let tr = 0.5 / (1.0 + i as f64 * 0.05);
            let va = tr * 1.1 + 0.01;
            assert!(w.feed(&Tick { fold: 0, iter: i, train_loss: Some(tr), valid_loss: Some(va), event: None }).is_none());
        }
    }
}
