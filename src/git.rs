use std::collections::BTreeMap;
use std::collections::BTreeSet;
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
            if path.file_name() == Some(std::ffi::OsStr::new(".git")) {
                continue;
            }
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

pub fn head_tree(cwd: Option<&Path>) -> String {
    let base = match cwd {
        Some(p) => p.to_path_buf(),
        None => match std::env::current_dir() {
            Ok(cwd) => cwd,
            Err(_) => return String::new(),
        },
    };
    let repo = match gix::open(&base) {
        Ok(repo) => repo,
        Err(_) => return String::new(),
    };
    match repo.head_tree_id_or_empty() {
        Ok(id) => id.to_string(),
        Err(_) => String::new(),
    }
}

pub fn diff(cwd: Option<&Path>, from: &str, to: &str) -> String {
    let base = match cwd {
        Some(p) => p.to_path_buf(),
        None => match std::env::current_dir() {
            Ok(c) => c,
            Err(_) => return String::new(),
        },
    };
    let repo = match gix::open(&base) {
        Ok(r) => r,
        Err(_) => return String::new(),
    };
    let mut old_files: BTreeMap<String, String> = BTreeMap::new();
    let mut new_files: BTreeMap<String, String> = BTreeMap::new();
    read_tree_flat(&repo, from, "", &mut old_files);
    read_tree_flat(&repo, to, "", &mut new_files);
    let mut paths: BTreeSet<String> = BTreeSet::new();
    for p in old_files.keys().chain(new_files.keys()) {
        paths.insert(p.clone());
    }
    let mut out = String::new();
    for path in &paths {
        let (old_oid, new_oid) = (old_files.get(path), new_files.get(path));
        let changed = match (old_oid, new_oid) {
            (None, Some(_)) | (Some(_), None) => true,
            (Some(o), Some(n)) => o != n,
            _ => false,
        };
        if !changed {
            continue;
        }
        let old_data = old_oid.map(|o| read_blob(&repo, o)).unwrap_or_default();
        let new_data = new_oid.map(|o| read_blob(&repo, o)).unwrap_or_default();
        let old_text = String::from_utf8_lossy(&old_data);
        let new_text = String::from_utf8_lossy(&new_data);
        let old_s: &str = &old_text;
        let new_s: &str = &new_text;
        let text_diff = similar::TextDiff::from_lines(old_s, new_s);
        let mut udiff = similar::udiff::UnifiedDiff::from_text_diff(&text_diff);
        udiff.context_radius(3);
        udiff.header(&format!("a/{path}"), &format!("b/{path}"));
        out.push_str(&format!("diff --git a/{path} b/{path}\n"));
        out.push_str(&udiff.to_string());
    }
    out
}

fn read_tree_flat(
    repo: &gix::Repository,
    tree_id: &str,
    prefix: &str,
    out: &mut BTreeMap<String, String>,
) {
    let oid: gix::hash::ObjectId = match tree_id.parse() {
        Ok(o) => o,
        Err(_) => return,
    };
    let tree = match repo.find_tree(oid) {
        Ok(t) => t,
        Err(_) => return,
    };
    for entry in tree.iter().flatten() {
        let name = entry.inner.filename.to_string();
        let full = if prefix.is_empty() {
            name.clone()
        } else {
            format!("{prefix}/{name}")
        };
        let oid = entry.inner.oid.to_string();
        if entry.inner.mode.is_tree() {
            read_tree_flat(repo, &oid, &full, out);
        } else {
            out.insert(full, oid);
        }
    }
}

fn read_blob(repo: &gix::Repository, oid: &str) -> Vec<u8> {
    let oid: gix::hash::ObjectId = match oid.parse() {
        Ok(o) => o,
        Err(_) => return Vec::new(),
    };
    match repo.find_blob(oid) {
        Ok(blob) => blob.data.to_vec(),
        Err(_) => Vec::new(),
    }
}

#[cfg(test)]
mod test {
    use super::{diff, head_tree, is_ignored, snapshot_tree};

    fn temp_repo() -> test_files::TestFiles {
        let dir = test_files::TestFiles::new();
        std::process::Command::new("git")
            .args(["init", "-q"])
            .current_dir(dir.path())
            .output()
            .unwrap();
        dir
    }

    fn git(dir: &std::path::Path, args: &[&str]) -> String {
        let out = std::process::Command::new("git")
            .args(args)
            .current_dir(dir)
            .output()
            .unwrap();
        String::from_utf8_lossy(&out.stdout).into_owned()
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
        let dir = temp_repo();
        dir.file("a.txt", "one\n");
        let expected = git_empty_tree(dir.path());
        let got = head_tree(Some(dir.path()));
        assert_eq!(got, expected);
    }

    #[test]
    fn head_tree_matches_git_after_a_commit() {
        let dir = temp_repo();
        dir.file("a.txt", "one\n");
        let commit = |args: &[&str]| {
            let out = std::process::Command::new("git")
                .args(args)
                .current_dir(dir.path())
                .env("GIT_AUTHOR_NAME", "t")
                .env("GIT_AUTHOR_EMAIL", "t@t")
                .env("GIT_COMMITTER_NAME", "t")
                .env("GIT_COMMITTER_EMAIL", "t@t")
                .output()
                .unwrap();
            String::from_utf8_lossy(&out.stdout).into_owned()
        };
        commit(&["add", "-A"]);
        commit(&["commit", "-q", "-m", "c"]);
        let expected = commit(&["rev-parse", "HEAD^{tree}"]).trim().to_string();
        let got = head_tree(Some(dir.path()));
        assert_eq!(got, expected);
    }

    #[test]
    fn snapshot_respects_gitignore() {
        let dir = temp_repo();
        dir.file(".gitignore", "*.log\n");
        dir.file("keep.txt", "keep\n");
        dir.file("skip.log", "skip\n");
        let tree = snapshot_tree(Some(dir.path())).expect("snapshot");
        let ls = git(dir.path(), &["ls-tree", "-r", &tree]);
        assert!(ls.contains("keep.txt"));
        assert!(!ls.contains("skip.log"));
    }

    #[test]
    fn exclude_stack_ignores_a_gitignored_dir() {
        let dir = temp_repo();
        dir.file(".gitignore", "target/\n");
        std::fs::create_dir_all(dir.path().join("target")).unwrap();
        std::fs::write(dir.path().join("target/a.txt"), "x\n").unwrap();
        let repo = gix::open(dir.path()).unwrap();
        let index = repo.index_or_empty().unwrap();
        let source = gix::worktree::stack::state::ignore::Source::WorktreeThenIdMappingIfNotSkipped;
        let mut excludes = repo.excludes(&***index, None, source).unwrap();
        assert!(is_ignored(&mut excludes, "target", true));
    }

    #[test]
    fn diff_produces_a_unified_diff() {
        let dir = temp_repo();
        dir.file("a.txt", "one\ntwo\nthree\n");
        dir.file("b.txt", "keep\n");
        let commit = |args: &[&str]| {
            let out = std::process::Command::new("git")
                .args(args)
                .current_dir(dir.path())
                .env("GIT_AUTHOR_NAME", "t")
                .env("GIT_AUTHOR_EMAIL", "t@t")
                .env("GIT_COMMITTER_NAME", "t")
                .env("GIT_COMMITTER_EMAIL", "t@t")
                .output()
                .unwrap();
            String::from_utf8_lossy(&out.stdout).into_owned()
        };
        commit(&["add", "-A"]);
        commit(&["commit", "-q", "-m", "c1"]);
        let tree1 = head_tree(Some(dir.path()));
        dir.file("a.txt", "one\nTWO\nthree\nfour\n");
        dir.file("c.txt", "new file\n");
        std::fs::remove_file(dir.path().join("b.txt")).unwrap();
        commit(&["add", "-A"]);
        commit(&["commit", "-q", "-m", "c2"]);
        let tree2 = head_tree(Some(dir.path()));
        let diff = diff(Some(dir.path()), &tree1, &tree2);
        assert!(diff.contains("diff --git a/a.txt b/a.txt"), "missing a.txt header: {diff}");
        assert!(diff.contains("-two"), "missing removed line: {diff}");
        assert!(diff.contains("+TWO"), "missing added line: {diff}");
        assert!(diff.contains("+four"), "missing inserted line: {diff}");
        assert!(diff.contains("diff --git a/c.txt b/c.txt"), "missing c.txt header: {diff}");
        assert!(diff.contains("+new file"), "missing c.txt content: {diff}");
        assert!(diff.contains("diff --git a/b.txt b/b.txt"), "missing b.txt header: {diff}");
        assert!(diff.contains("-keep"), "missing b.txt removed line: {diff}");
        assert!(!diff.contains("@@ -0,0 +1,0 @@"), "empty hunk: {diff}");
    }

    #[test]
    fn diff_matches_git_diff() {
        let dir = temp_repo();
        dir.file("a.txt", "one\ntwo\nthree\n");
        let commit = |args: &[&str]| {
            let out = std::process::Command::new("git")
                .args(args)
                .current_dir(dir.path())
                .env("GIT_AUTHOR_NAME", "t")
                .env("GIT_AUTHOR_EMAIL", "t@t")
                .env("GIT_COMMITTER_NAME", "t")
                .env("GIT_COMMITTER_EMAIL", "t@t")
                .output()
                .unwrap();
            String::from_utf8_lossy(&out.stdout).into_owned()
        };
        commit(&["add", "-A"]);
        commit(&["commit", "-q", "-m", "c1"]);
        let tree1 = head_tree(Some(dir.path()));
        dir.file("a.txt", "one\nTWO\nthree\nfour\n");
        commit(&["add", "-A"]);
        commit(&["commit", "-q", "-m", "c2"]);
        let tree2 = head_tree(Some(dir.path()));
        let expected = git(dir.path(), &["diff", &tree1, &tree2]);
        let got = diff(Some(dir.path()), &tree1, &tree2);
        let norm = |s: &str| -> Vec<String> {
            s.lines()
                .filter(|l| !l.starts_with("index ") && !l.starts_with("diff --git"))
                .map(|l| l.to_string())
                .collect()
        };
        assert_eq!(norm(&got), norm(&expected), "\n--- got ---\n{got}\n--- expected ---\n{expected}");
    }

    #[test]
    fn snapshot_captures_new_and_modified_files() {
        let dir = temp_repo();
        dir.file("a.txt", "one\n");
        let tree1 = snapshot_tree(Some(dir.path())).expect("first snapshot");
        dir.file("a.txt", "one\ntwo\n");
        dir.file("b.txt", "new file\n");
        let tree2 = snapshot_tree(Some(dir.path())).expect("second snapshot");
        let diff = git(dir.path(), &["diff", &tree1, &tree2]);
        assert!(diff.contains("+two"));
        assert!(diff.contains("b.txt"));
        assert!(diff.contains("+new file"));
    }
}
