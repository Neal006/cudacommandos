//! mlguard — trust checks for the entity-resolution pipeline.
//!
//!   mlguard run <summary.json> [--champion <summary.json>]
//!   mlguard watch <metrics.jsonl> [--once] [--stop-file <path>]
//!   mlguard split --folds <folds.tsv> | --train <ids.txt> --valid <ids.txt>
//!   mlguard submission --matching <tsv> --candidate <tsv> --test-dir <dir> [--summary <summary.json>]
//!
//! Global: --config <mlguard.toml> (defaults to the thresholds compiled in).
//! Exit: 0 = pass (warnings allowed), 1 = a FAIL finding, 2 = usage / IO error.
mod checks;
mod files;

use checks::{Config, Finding, Level, Summary, Tick, Watcher};
use std::collections::{HashMap, HashSet};
use std::io::{BufRead, BufReader, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::{env, fs, process, thread, time::Duration};

const DEFAULT_CFG: &str = include_str!("../mlguard.toml");

fn die(msg: String) -> ! {
    eprintln!("mlguard: {}", msg);
    process::exit(2)
}

struct Args {
    pos: Vec<String>,
    kv: HashMap<String, String>,
    flags: HashSet<String>,
}

fn parse_args(raw: &[String]) -> Args {
    let mut a = Args { pos: vec![], kv: HashMap::new(), flags: HashSet::new() };
    let mut i = 0;
    while i < raw.len() {
        let s = &raw[i];
        if let Some(k) = s.strip_prefix("--") {
            if k == "once" {
                a.flags.insert(k.to_string());
            } else {
                let v = raw.get(i + 1).cloned().unwrap_or_else(|| die(format!("--{} needs a value", k)));
                a.kv.insert(k.to_string(), v);
                i += 1;
            }
        } else {
            a.pos.push(s.clone());
        }
        i += 1;
    }
    a
}

fn load_cfg(a: &Args) -> Config {
    let text = match a.kv.get("config") {
        Some(p) => fs::read_to_string(p).unwrap_or_else(|e| die(format!("{}: {}", p, e))),
        None => DEFAULT_CFG.to_string(),
    };
    toml::from_str(&text).unwrap_or_else(|e| die(format!("config: {}", e)))
}

fn load_summary(p: &str) -> Summary {
    let t = fs::read_to_string(p).unwrap_or_else(|e| die(format!("{}: {}", p, e)));
    serde_json::from_str(&t).unwrap_or_else(|e| die(format!("{}: {}", p, e)))
}

fn need<'a>(a: &'a Args, k: &str) -> &'a str {
    a.kv.get(k).map(|s| s.as_str()).unwrap_or_else(|| die(format!("missing --{}", k)))
}

fn report(title: &str, findings: &[Finding]) -> i32 {
    println!("== mlguard {} ==", title);
    for x in findings {
        let tag = match x.level {
            Level::Fail => "FAIL",
            Level::Warn => "WARN",
            Level::Pass => "PASS",
        };
        println!("[{}] {:<22} {}", tag, x.rule, x.msg);
    }
    let fail = findings.iter().any(|x| x.level == Level::Fail);
    println!("{}", if fail { "RESULT: FAIL" } else { "RESULT: PASS" });
    fail as i32
}

fn cmd_run(a: &Args, cfg: &Config) -> i32 {
    let p = a.pos.get(1).unwrap_or_else(|| die("run <summary.json>".into()));
    let s = load_summary(p);
    let champ = a.kv.get("champion").map(|c| load_summary(c));
    report(&format!("run {}", s.run_id), &checks::check_run(&s, &cfg.run, champ.as_ref()))
}

/// Tail metrics.jsonl while training runs. On a violation, append `<fold><TAB><rule>: <msg>` to
/// the stop file — training stops THAT fold at its next eval (src/runlog.py) — and keep
/// watching the remaining folds. Ends on {"event":"end"} (or EOF with --once) and exits 1
/// if anything was flagged.
fn cmd_watch(a: &Args, cfg: &Config) -> i32 {
    let p = PathBuf::from(a.pos.get(1).unwrap_or_else(|| die("watch <metrics.jsonl>".into())));
    let stop = a.kv.get("stop-file").map(PathBuf::from)
        .unwrap_or_else(|| p.parent().unwrap_or(Path::new(".")).join("MLGUARD_STOP"));
    let once = a.flags.contains("once");
    let mut w = Watcher::new(cfg.watch.clone());
    let mut pos = 0u64;
    let mut n = 0usize;
    let mut found: Vec<Finding> = Vec::new();
    let finish = |found: &[Finding], n: usize| -> i32 {
        if found.is_empty() {
            println!("mlguard watch: {} ticks, no violation", n);
            0
        } else {
            report(&format!("watch {}", p.display()), found)
        }
    };
    loop {
        if let Ok(mut file) = fs::File::open(&p) {
            file.seek(SeekFrom::Start(pos)).ok();
            let mut r = BufReader::new(file);
            let mut line = String::new();
            // only consume complete lines so a half-written line is re-read next poll
            while r.read_line(&mut line).unwrap_or(0) > 0 && line.ends_with('\n') {
                pos += line.len() as u64;
                if let Ok(t) = serde_json::from_str::<Tick>(line.trim()) {
                    n += 1;
                    if t.event.as_deref() == Some("end") {
                        return finish(&found, n);
                    }
                    if let Some(x) = w.feed(&t) {
                        use std::io::Write;
                        if let Ok(mut fh) = fs::OpenOptions::new().create(true).append(true).open(&stop) {
                            writeln!(fh, "{}\t{}: {}", t.fold, x.rule, x.msg).ok();
                        }
                        eprintln!("mlguard watch: fold {} flagged: {}", t.fold, x.msg);
                        found.push(x);
                    }
                }
                line.clear();
            }
        } else if once {
            die(format!("{}: not found", p.display()));
        }
        if once {
            return finish(&found, n);
        }
        thread::sleep(Duration::from_millis(cfg.watch.poll_ms));
    }
}

fn cmd_split(a: &Args) -> i32 {
    let mut out = vec![];
    let read = |k: &str| fs::read_to_string(need(a, k)).unwrap_or_else(|e| die(format!("{}: {}", k, e)));
    if a.kv.contains_key("folds") {
        files::check_folds(&read("folds"), &mut out);
    }
    if a.kv.contains_key("train") {
        files::check_disjoint(&read("train"), &read("valid"), &mut out);
    }
    report("split", &out)
}

fn cmd_submission(a: &Args, cfg: &Config) -> i32 {
    let dir = PathBuf::from(need(a, "test-dir"));
    let s1 = files::load_source(&dir.join("test_source1.tsv")).unwrap_or_else(|e| die(e));
    let mut others: HashSet<String> = HashSet::new();
    for f in ["test_source2.tsv", "test_source3.tsv"] {
        others.extend(files::load_source(&dir.join(f)).unwrap_or_else(|e| die(e)).into_keys());
    }
    let mut out = vec![];
    let rd = |k: &str| fs::read_to_string(need(a, k)).unwrap_or_else(|e| die(format!("{}: {}", k, e)));
    let m = files::parse_table(&rd("matching"), "matched_entity_ids", &mut out, "matching");
    let c = files::parse_table(&rd("candidate"), "candidate_entity_ids", &mut out, "candidate");
    let st = files::check_table(&m, "matching", &s1, &others, &mut out);
    let cst = files::check_table(&c, "candidate", &s1, &others, &mut out);
    files::check_consistency(&m, &c, cfg.submission.partition_violation_is_fail, &mut out);
    for (k, v) in &st.singleton_rate {
        println!("  {:<8} predicted singleton rate {:.3}  links/entity {:.2}  candidates/entity {:.1}",
                 k, v, st.links_per_entity[k], cst.links_per_entity.get(k).copied().unwrap_or(0.0));
    }
    if let Some(sp) = a.kv.get("summary") {
        let mut s = load_summary(sp);
        s.test_pred_singleton_rate = st.singleton_rate;
        s.test_pred_links_per_entity = st.links_per_entity;
        out.extend(checks::drift_checks(&s, &cfg.run));
    }
    report("submission", &out)
}

fn main() {
    let raw: Vec<String> = env::args().skip(1).collect();
    let a = parse_args(&raw);
    let cfg = load_cfg(&a);
    let code = match a.pos.first().map(|s| s.as_str()) {
        Some("run") => cmd_run(&a, &cfg),
        Some("watch") => cmd_watch(&a, &cfg),
        Some("split") => cmd_split(&a),
        Some("submission") => cmd_submission(&a, &cfg),
        _ => die("usage: mlguard <run|watch|split|submission> ... (see src/main.rs header)".into()),
    };
    process::exit(code);
}
