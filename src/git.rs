use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};

static SNAPSHOT_COUNTER: AtomicU64 = AtomicU64::new(0);

pub fn snapshot_tree(cwd: Option<&Path>) -> Option<String> {
    let index = std::env::temp_dir().join(format!(
        "kite-index-{}-{}",
        std::process::id(),
        SNAPSHOT_COUNTER.fetch_add(1, Ordering::Relaxed)
    ));
    let mut add = std::process::Command::new("git");
    if let Some(cwd) = cwd {
        add.current_dir(cwd);
    }
    let add = add.env("GIT_INDEX_FILE", &index).arg("add").arg("-A").output().ok()?;
    if !add.status.success() {
        std::fs::remove_file(&index).ok();
        return None;
    }
    let mut write = std::process::Command::new("git");
    if let Some(cwd) = cwd {
        write.current_dir(cwd);
    }
    let write = write.env("GIT_INDEX_FILE", &index).arg("write-tree").output().ok()?;
    std::fs::remove_file(&index).ok();
    if !write.status.success() {
        return None;
    }
    let stdout = String::from_utf8_lossy(&write.stdout);
    let sha = stdout.trim();
    if sha.is_empty() {
        None
    } else {
        Some(sha.to_string())
    }
}

pub fn head_tree() -> String {
    let output = std::process::Command::new("git")
        .args(["rev-parse", "--verify", "HEAD^{tree}"])
        .output();
    match output {
        Ok(output) if output.status.success() => {
            String::from_utf8_lossy(&output.stdout).trim().to_string()
        }
        _ => {
            let empty = std::process::Command::new("git")
                .arg("mktree")
                .stdin(std::process::Stdio::null())
                .output();
            match empty {
                Ok(output) if output.status.success() => {
                    String::from_utf8_lossy(&output.stdout).trim().to_string()
                }
                _ => String::new(),
            }
        }
    }
}

pub fn diff(from: &str, to: &str) -> String {
    match std::process::Command::new("git")
        .args(["diff", from, to])
        .output()
    {
        Ok(output) if output.status.success() => {
            String::from_utf8_lossy(&output.stdout).into_owned()
        }
        Ok(output) => String::from_utf8_lossy(&output.stderr).into_owned(),
        Err(source) => format!("git failed: {source}"),
    }
}

#[cfg(test)]
mod test {
    use super::snapshot_tree;

    #[test]
    fn snapshot_captures_new_and_modified_files() {
        let dir = std::env::temp_dir().join(format!(
            "kite-git-snapshot-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let git = |args: &[&str]| -> String {
            let out = std::process::Command::new("git")
                .args(args)
                .current_dir(&dir)
                .output()
                .unwrap();
            String::from_utf8_lossy(&out.stdout).into_owned()
        };
        git(&["init", "-q"]);
        std::fs::write(dir.join("a.txt"), "one\n").unwrap();
        let tree1 = snapshot_tree(Some(&dir)).expect("first snapshot");
        std::fs::write(dir.join("a.txt"), "one\ntwo\n").unwrap();
        std::fs::write(dir.join("b.txt"), "new file\n").unwrap();
        let tree2 = snapshot_tree(Some(&dir)).expect("second snapshot");
        let diff = git(&["diff", &tree1, &tree2]);
        assert!(diff.contains("+two"));
        assert!(diff.contains("b.txt"));
        assert!(diff.contains("+new file"));
        std::fs::remove_dir_all(&dir).ok();
    }
}
