use std::collections::BTreeMap;
use std::path::Path;

type EntryKind = gix::objs::tree::EntryKind;
type ObjectId = gix::hash::ObjectId;

pub fn snapshot_tree(cwd: Option<&Path>) -> Option<String> {
    let base = match cwd {
        Some(p) => p.to_path_buf(),
        None => std::env::current_dir().ok()?,
    };
    let repo = gix::open(&base).ok()?;
    let work_dir = repo.workdir()?.to_path_buf();
    let index = repo.index_or_empty().ok()?;
    let source = gix::worktree::stack::state::ignore::Source::WorktreeThenIdMappingIfNotSkipped;
    let mut excludes = repo.excludes(&***index, None, source).ok()?;
    let mut files: Vec<(String, EntryKind, ObjectId)> = Vec::new();
    walk(&work_dir, &work_dir, &repo, &mut excludes, &mut files);
    build_tree(&repo, &files).map(|id| id.to_string())
}

fn walk(
    root: &Path,
    dir: &Path,
    repo: &gix::Repository,
    excludes: &mut gix::AttributeStack,
    out: &mut Vec<(String, EntryKind, ObjectId)>,
) {
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return,
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let meta = match entry.metadata() {
            Ok(m) => m,
            Err(_) => continue,
        };
        let rel = match path.strip_prefix(root) {
            Ok(r) => r.to_string_lossy().replace('\\', "/"),
            Err(_) => continue,
        };
        if meta.file_type().is_symlink() {
            if is_ignored(excludes, &rel, false) {
                continue;
            }
            let target = std::fs::read_link(&path).unwrap_or_default();
            let data = target.to_string_lossy().into_owned().into_bytes();
            let oid = match blob(repo, &data) {
                Some(o) => o,
                None => continue,
            };
            out.push((rel, EntryKind::Link, oid));
        } else if meta.is_dir() {
            if is_ignored(excludes, &rel, true) {
                continue;
            }
            if path.join(".git").is_file() {
                if let Some(oid) = submodule_head(&path) {
                    out.push((rel, EntryKind::Commit, oid));
                }
            } else {
                walk(root, &path, repo, excludes, out);
            }
        } else if meta.is_file() {
            if is_ignored(excludes, &rel, false) {
                continue;
            }
            let data = match std::fs::read(&path) {
                Ok(d) => d,
                Err(_) => continue,
            };
            let oid = match blob(repo, &data) {
                Some(o) => o,
                None => continue,
            };
            let kind = if is_executable(&meta) {
                EntryKind::BlobExecutable
            } else {
                EntryKind::Blob
            };
            out.push((rel, kind, oid));
        }
    }
}

fn is_ignored(excludes: &mut gix::AttributeStack, rel: &str, is_dir: bool) -> bool {
    let mode = if is_dir { Some(gix::index::entry::Mode::DIR) } else { None };
    match excludes.at_path(rel, mode) {
        Ok(p) => p.is_excluded(),
        Err(_) => false,
    }
}

fn is_executable(meta: &std::fs::Metadata) -> bool {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        meta.permissions().mode() & 0o100 != 0
    }
    #[cfg(not(unix))]
    {
        false
    }
}

fn submodule_head(path: &Path) -> Option<ObjectId> {
    let repo = gix::open(path).ok()?;
    let id = repo.head_id().ok()?;
    Some((*id).to_owned())
}

fn blob(repo: &gix::Repository, data: &[u8]) -> Option<ObjectId> {
    let blob = gix::objs::Blob {
        data: data.to_vec(),
    };
    let id = repo.write_object(blob).ok()?;
    Some((*id).to_owned())
}

fn build_tree(repo: &gix::Repository, files: &[(String, EntryKind, ObjectId)]) -> Option<ObjectId> {
    let mut groups: BTreeMap<String, Vec<&(String, EntryKind, ObjectId)>> = BTreeMap::new();
    for f in files {
        let name = f.0.split('/').next().unwrap_or(&f.0).to_string();
        groups.entry(name).or_default().push(f);
    }
    let mut entries: Vec<gix::objs::tree::Entry> = Vec::new();
    for (name, group) in groups {
        if group.iter().any(|f| f.0.contains('/')) {
            let sub: Vec<(String, EntryKind, ObjectId)> = group
                .iter()
                .map(|f| {
                    let rest = f
                        .0
                        .split_once('/')
                        .map(|(_, r)| r.to_string())
                        .unwrap_or_default();
                    (rest, f.1, f.2)
                })
                .collect();
            let oid = build_tree(repo, &sub)?;
            entries.push(gix::objs::tree::Entry {
                mode: EntryKind::Tree.into(),
                filename: name.into(),
                oid,
            });
        } else {
            let f = &group[0];
            entries.push(gix::objs::tree::Entry {
                mode: f.1.into(),
                filename: name.into(),
                oid: f.2,
            });
        }
    }
    entries.sort();
    let id = repo.write_object(gix::objs::Tree { entries }).ok()?;
    Some((*id).to_owned())
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
    fn snapshot_respects_gitignore() {
        let dir = temp_repo("snapshot-ignore");
        std::fs::write(dir.join(".gitignore"), "*.log\n").unwrap();
        std::fs::write(dir.join("keep.txt"), "keep\n").unwrap();
        std::fs::write(dir.join("skip.log"), "skip\n").unwrap();
        let tree = snapshot_tree(Some(&dir)).expect("snapshot");
        let out = std::process::Command::new("git")
            .args(["ls-tree", "-r", &tree])
            .current_dir(&dir)
            .output()
            .unwrap();
        let ls = String::from_utf8_lossy(&out.stdout);
        assert!(ls.contains("keep.txt"));
        assert!(!ls.contains("skip.log"));
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
