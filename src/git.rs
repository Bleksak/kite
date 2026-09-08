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
    let cwd = match std::env::current_dir() {
        Ok(cwd) => cwd,
        Err(_) => return String::new(),
    };
    let repo = match gix::open(&cwd) {
        Ok(repo) => repo,
        Err(_) => return String::new(),
    };
    match repo.head_tree_id_or_empty() {
        Ok(id) => id.to_string(),
        Err(_) => String::new(),
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
    use super::{head_tree, snapshot_tree};

    fn temp_repo(prefix: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "kite-git-{}-{}-{}",
            prefix,
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
        dir
    }

    fn git_empty_tree(dir: &std::path::Path) -> String {
        let out = std::process::Command::new("git")
            .arg("mktree")
            .stdin(std::process::Stdio::null())
            .current_dir(dir)
            .output()
            .unwrap();
        String::from_utf8_lossy(&out.stdout).trim().to_string()
    }

    #[test]
    fn head_tree_of_a_fresh_repo_matches_git_empty_tree() {
        let dir = temp_repo("head-fresh");
        std::fs::write(dir.join("a.txt"), "one\n").unwrap();
        let expected = git_empty_tree(&dir);
        let before = std::env::current_dir().unwrap();
        std::env::set_current_dir(&dir).unwrap();
        let got = head_tree();
        std::env::set_current_dir(before).unwrap();
        assert_eq!(got, expected);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn head_tree_matches_git_after_a_commit() {
        let dir = temp_repo("head-commit");
        std::fs::write(dir.join("a.txt"), "one\n").unwrap();
        let git = |args: &[&str]| -> String {
            let out = std::process::Command::new("git")
                .args(args)
                .current_dir(&dir)
                .env("GIT_AUTHOR_NAME", "t")
                .env("GIT_AUTHOR_EMAIL", "t@t")
                .env("GIT_COMMITTER_NAME", "t")
                .env("GIT_COMMITTER_EMAIL", "t@t")
                .output()
                .unwrap();
            String::from_utf8_lossy(&out.stdout).into_owned()
        };
        git(&["add", "-A"]);
        git(&["commit", "-q", "-m", "c"]);
        let expected = git(&["rev-parse", "HEAD^{tree}"]).trim().to_string();
        let before = std::env::current_dir().unwrap();
        std::env::set_current_dir(&dir).unwrap();
        let got = head_tree();
        std::env::set_current_dir(before).unwrap();
        assert_eq!(got, expected);
        std::fs::remove_dir_all(&dir).ok();
    }

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
