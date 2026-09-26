//! File-level checks: submission TSVs and train/valid split files.
use crate::checks::{Finding, Level};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs;
use std::path::Path;

fn fd(level: Level, rule: &'static str, msg: String) -> Finding {
    Finding { level, rule, msg }
}

fn read(p: &Path) -> Result<String, String> {
    fs::read_to_string(p).map_err(|e| format!("{}: {}", p.display(), e))
}

/// entity_id -> country from a test source file (first and last tab field).
pub fn load_source(p: &Path) -> Result<HashMap<String, String>, String> {
    let text = read(p)?;
    let mut m = HashMap::new();
    for line in text.lines().skip(1) {
        let line = line.trim_end_matches('\r');
        if line.is_empty() {
            continue;
        }
        let id = line.split('\t').next().unwrap_or("").to_string();
        let country = line.rsplit('\t').next().unwrap_or("").to_string();
        m.insert(id, country);
    }
    Ok(m)
}

pub struct Table {
    pub rows: Vec<(String, Vec<String>)>,
}

pub fn parse_table(text: &str, list_col: &str, out: &mut Vec<Finding>, label: &str) -> Table {
    let mut lines = text.lines();
    let header = lines.next().unwrap_or("").trim_end_matches('\r');
    let want = format!("source1_entity_id\t{}", list_col);
    if header != want {
        out.push(fd(Level::Fail, "header", format!("{}: header '{}' != '{}'", label, header, want)));
    }
    let mut rows = Vec::new();
    for (i, line) in lines.enumerate() {
        let line = line.trim_end_matches('\r');
        if line.is_empty() {
            continue;
        }
        let mut parts = line.splitn(3, '\t');
        let id = parts.next().unwrap_or("").to_string();
        let list = parts.next().unwrap_or("");
        if parts.next().is_some() {
            out.push(fd(Level::Fail, "columns", format!("{} line {}: more than 2 columns", label, i + 2)));
        }
        let ids: Vec<String> = if list.trim().is_empty() {
            vec![]
        } else {
            list.split(',').map(|s| s.trim().to_string()).collect()
        };
        rows.push((id, ids));
    }
    Table { rows }
}

pub struct CountryStats {
    pub singleton_rate: BTreeMap<String, f64>,
    pub links_per_entity: BTreeMap<String, f64>,
}

/// Validate one output table against the test sources. Returns per-country stats.
pub fn check_table(
    t: &Table, label: &str, s1: &HashMap<String, String>, others: &HashSet<String>,
    out: &mut Vec<Finding>,
) -> CountryStats {
    let mut seen = HashSet::new();
    let mut bad = 0usize;
    let mut first_bad = String::new();
    let note = |msg: String, bad: &mut usize, first: &mut String| {
        if *bad == 0 {
            *first = msg;
        }
        *bad += 1;
    };
    let mut n: BTreeMap<String, (usize, usize, usize)> = BTreeMap::new(); // entities, empty, links
    for (id, ids) in &t.rows {
        if !seen.insert(id.clone()) {
            note(format!("duplicate row {}", id), &mut bad, &mut first_bad);
        }
        let country = match s1.get(id) {
            Some(c) => c.clone(),
            None => {
                note(format!("{} is not a test Source-1 id", id), &mut bad, &mut first_bad);
                continue;
            }
        };
        let mut inrow = HashSet::new();
        for x in ids {
            if !(x.starts_with("S2-") || x.starts_with("S3-")) {
                note(format!("{}: '{}' is not an S2/S3 id", id, x), &mut bad, &mut first_bad);
            } else if !others.contains(x) {
                note(format!("{}: '{}' not in test S2/S3", id, x), &mut bad, &mut first_bad);
            }
            if !inrow.insert(x) {
                note(format!("{}: duplicate '{}' in list", id, x), &mut bad, &mut first_bad);
            }
        }
        let e = n.entry(country).or_default();
        e.0 += 1;
        e.1 += ids.is_empty() as usize;
        e.2 += ids.len();
    }
    if bad > 0 {
        out.push(fd(Level::Fail, "format", format!("{}: {} problems, first: {}", label, bad, first_bad)));
    }
    let missing = s1.keys().filter(|k| !seen.contains(*k)).count();
    if missing > 0 {
        out.push(fd(Level::Fail, "missing_rows", format!("{}: {} test Source-1 ids have no row", label, missing)));
    }
    CountryStats {
        singleton_rate: n.iter().map(|(k, v)| (k.clone(), v.1 as f64 / v.0 as f64)).collect(),
        links_per_entity: n.iter().map(|(k, v)| (k.clone(), v.2 as f64 / v.0 as f64)).collect(),
    }
}

/// Matches must be a subset of candidates; GT is a partition so no S2/S3 id may serve two S1 rows.
pub fn check_consistency(m: &Table, c: &Table, partition_fail: bool, out: &mut Vec<Finding>) {
    let cand: HashMap<&str, HashSet<&str>> = c.rows.iter()
        .map(|(k, v)| (k.as_str(), v.iter().map(|s| s.as_str()).collect())).collect();
    let mut not_sub = 0usize;
    let mut owner: HashMap<&str, &str> = HashMap::new();
    let mut shared = 0usize;
    for (k, v) in &m.rows {
        let cs = cand.get(k.as_str());
        for x in v {
            if cs.map_or(true, |s| !s.contains(x.as_str())) {
                not_sub += 1;
            }
            if owner.insert(x.as_str(), k.as_str()).is_some() {
                shared += 1;
            }
        }
    }
    if not_sub > 0 {
        out.push(fd(Level::Fail, "not_subset", format!("{} matched ids never appeared as candidates — pipeline bug", not_sub)));
    }
    if shared > 0 {
        let lvl = if partition_fail { Level::Fail } else { Level::Warn };
        out.push(fd(lvl, "partition", format!(
            "{} S2/S3 ids are matched to more than one S1 — ground truth never does this", shared)));
    }
}

/// folds.tsv: `s1_id<TAB>fold`. Every id exactly once; folds roughly balanced.
pub fn check_folds(text: &str, out: &mut Vec<Finding>) {
    let mut seen = HashSet::new();
    let mut sizes: BTreeMap<String, usize> = BTreeMap::new();
    let mut dup = 0usize;
    for line in text.lines() {
        let line = line.trim_end_matches('\r');
        let mut p = line.split('\t');
        let (id, fold) = (p.next().unwrap_or(""), p.next().unwrap_or(""));
        if id.is_empty() || id == "source1_entity_id" || id == "s1_id" {
            continue;
        }
        if !seen.insert(id.to_string()) {
            dup += 1;
        }
        *sizes.entry(fold.to_string()).or_default() += 1;
    }
    if dup > 0 {
        out.push(fd(Level::Fail, "group_leak", format!("{} Source-1 ids appear in more than one fold row", dup)));
    }
    if let (Some(mx), Some(mn)) = (sizes.values().max(), sizes.values().min()) {
        if *mn > 0 && *mx as f64 / *mn as f64 > 1.25 {
            out.push(fd(Level::Warn, "fold_balance", format!("fold sizes unbalanced: {:?}", sizes)));
        }
    }
}

/// Two id lists (one per line) must not intersect.
pub fn check_disjoint(a: &str, b: &str, out: &mut Vec<Finding>) {
    let sa: HashSet<&str> = a.lines().map(|l| l.trim()).filter(|l| !l.is_empty()).collect();
    let n = b.lines().map(|l| l.trim()).filter(|l| sa.contains(l)).count();
    if n > 0 {
        out.push(fd(Level::Fail, "split_overlap", format!("{} ids present in both train and validation", n)));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fails(v: &[Finding]) -> Vec<&'static str> {
        v.iter().filter(|x| x.level == Level::Fail).map(|x| x.rule).collect()
    }

    #[test]
    fn submission_rules() {
        let s1: HashMap<String, String> = [("S1-1", "US"), ("S1-2", "France"), ("S1-3", "US")]
            .iter().map(|(a, b)| (a.to_string(), b.to_string())).collect();
        let others: HashSet<String> = ["S2-1", "S2-2", "S3-1"].iter().map(|s| s.to_string()).collect();

        let mut out = vec![];
        let m = parse_table("source1_entity_id\tmatched_entity_ids\nS1-1\tS2-1,S3-1\nS1-2\t\nS1-3\tS2-2\n",
                            "matched_entity_ids", &mut out, "m");
        let c = parse_table("source1_entity_id\tcandidate_entity_ids\nS1-1\tS2-1,S3-1\nS1-2\tS2-2\nS1-3\tS2-2\n",
                            "candidate_entity_ids", &mut out, "c");
        let st = check_table(&m, "m", &s1, &others, &mut out);
        check_consistency(&m, &c, true, &mut out);
        assert!(fails(&out).is_empty(), "{:?}", out);
        assert_eq!(st.singleton_rate["France"], 1.0);
        assert_eq!(st.links_per_entity["US"], 1.5);

        let mut out = vec![];
        let bad = parse_table("source1_entity_id\tmatched_entity_ids\nS1-1\tS2-1,S2-1,S1-3\nS1-1\tS2-9\n",
                              "matched_entity_ids", &mut out, "m");
        check_table(&bad, "m", &s1, &others, &mut out);
        check_consistency(&bad, &c, true, &mut out);
        let r = fails(&out);
        assert!(r.contains(&"format") && r.contains(&"missing_rows") && r.contains(&"not_subset") && r.contains(&"partition"));
    }

    #[test]
    fn split_rules() {
        let mut out = vec![];
        check_folds("s1_id\tfold\nS1-1\t0\nS1-2\t1\nS1-1\t1\n", &mut out);
        assert!(fails(&out).contains(&"group_leak"));
        let mut out = vec![];
        check_disjoint("S1-1\nS1-2\n", "S1-3\nS1-2\n", &mut out);
        assert!(fails(&out).contains(&"split_overlap"));
    }
}
