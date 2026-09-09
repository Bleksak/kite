use std::path::Path;

pub const MAX_FILE_LINES: usize = 500;
pub const MAX_FILE_BYTES: usize = 100 * 1024;
const MAX_WALK_FILES: usize = 200;
const MAX_WALK_DEPTH: usize = 8;
pub const MAX_POPUP_CANDIDATES: usize = 5;

#[derive(Debug, Clone, PartialEq)]
pub struct Suggestion {
    pub insert: String,
    pub display: String,
}

pub fn token_at(input: &str, cursor: usize) -> Option<(usize, String)> {
    let chars: Vec<char> = input.chars().collect();
    let cursor = cursor.min(chars.len());
    let mut start = cursor;
    while start > 0 && !chars[start - 1].is_whitespace() {
        start -= 1;
    }
    let token: String = chars[start..cursor].iter().collect();
    (!token.is_empty()).then_some((start, token))
}

pub fn skill_candidates(prefix: &str, skills: &[crate::skill::Skill]) -> Vec<Suggestion> {
    let mut out = Vec::new();
    for skill in skills {
        if skill.name.starts_with(prefix) && skill.name.len() > prefix.len() {
            out.push(Suggestion {
                insert: format!("/skill:{}", skill.name),
                display: format!("{} — {}", skill.name, skill.description),
            });
        }
    }
    out.truncate(MAX_POPUP_CANDIDATES);
    out
}

pub fn candidates(input: &str, cursor: usize, cwd: &Path) -> Option<Vec<Suggestion>> {
    let (_start, token) = token_at(input, cursor)?;
    // The /skill: branch must be checked before the generic / branch: a
    // /skill:gi token would otherwise fall into the command branch and
    // return an empty list, never reaching the skill suggestions.
    if let Some(prefix) = token.strip_prefix("/skill:") {
        Some(skill_candidates(prefix, &crate::skill::load_skills()))
    } else if let Some(prefix) = token.strip_prefix('/') {
        let mut out = Vec::new();
        for (name, description) in crate::commands::command_list() {
            if name.starts_with(prefix) && name.len() > prefix.len() {
                out.push(Suggestion {
                    insert: format!("/{name}"),
                    display: format!("{name} — {description}"),
                });
            }
        }
        Some(out)
    } else if let Some(prefix) = token.strip_prefix('@') {
        let prefix = prefix.to_lowercase();
        let mut out = Vec::new();
        for file in walk_files(cwd) {
            let lower = file.to_lowercase();
            if lower.starts_with(&prefix) && lower.len() > prefix.len() {
                out.push(Suggestion {
                    insert: format!("@{file}"),
                    display: file,
                });
            }
        }
        out.truncate(MAX_POPUP_CANDIDATES);
        Some(out)
    } else {
        None
    }
}

pub fn apply(input: &str, cursor: usize, s: &Suggestion) -> (String, usize) {
    let (start, _token) = token_at(input, cursor).expect("a token must exist to apply to");
    let mut chars: Vec<char> = input.chars().collect();
    chars.splice(start..cursor, s.insert.chars());
    (chars.into_iter().collect(), start + s.insert.chars().count())
}

fn walk_files(cwd: &Path) -> Vec<String> {
    let mut out = Vec::new();
    walk(cwd, "", 0, &mut out);
    out
}

fn walk(dir: &Path, prefix: &str, depth: usize, out: &mut Vec<String>) {
    if depth > MAX_WALK_DEPTH || out.len() >= MAX_WALK_FILES {
        return;
    }
    let entries = match std::fs::read_dir(dir) {
        Ok(entries) => entries,
        Err(_) => return,
    };
    let mut names: Vec<String> = entries
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.file_name().to_string_lossy().into_owned())
        .collect();
    names.sort();
    for name in names {
        if out.len() >= MAX_WALK_FILES {
            return;
        }
        let path = dir.join(&name);
        let meta = match path.symlink_metadata() {
            Ok(meta) => meta,
            Err(_) => continue,
        };
        if meta.is_dir() {
            if !name.starts_with('.')
                && !matches!(
                    name.as_str(),
                    "node_modules" | "target" | "dist" | "build"
                )
            {
                let rel = if prefix.is_empty() {
                    name.clone()
                } else {
                    format!("{prefix}/{name}")
                };
                walk(&path, &rel, depth + 1, out);
            }
        } else if meta.is_file() {
            out.push(if prefix.is_empty() {
                name
            } else {
                format!("{prefix}/{name}")
            });
        }
    }
}

pub fn expand_file_refs(text: &str, cwd: &Path) -> String {
    let mut out = String::new();
    let mut rest = text;
    while let Some(pos) = rest.find('@') {
        let at_start = pos == 0 || rest[..pos].chars().last().unwrap().is_whitespace();
        if !at_start {
            out.push_str(&rest[..pos + 1]);
            rest = &rest[pos + 1..];
            continue;
        }
        out.push_str(&rest[..pos]);
        let token_end = rest[pos..]
            .find(char::is_whitespace)
            .map(|offset| pos + offset)
            .unwrap_or(rest.len());
        let token = &rest[pos..token_end];
        let path = &token[1..];
        let full = cwd.join(path);
        if !path.is_empty() && full.is_file() {
            out.push_str(&file_block(path, &full));
        } else {
            out.push_str(token);
        }
        rest = &rest[token_end..];
    }
    out.push_str(rest);
    out
}

fn file_block(path: &str, full: &Path) -> String {
    let data = match std::fs::read(full) {
        Ok(data) => data,
        Err(error) => {
            return format!("<file path=\"{path}\">[unreadable: {error}]</file>")
        }
    };
    let byte_truncated = data.len() > MAX_FILE_BYTES;
    let data = &data[..data.len().min(MAX_FILE_BYTES)];
    if data.iter().take(8192).any(|&byte| byte == 0) {
        return format!(
            "<file path=\"{path}\">[binary file ({} bytes)]</file>",
            data.len()
        );
    }
    let text = String::from_utf8_lossy(data);
    let lines: Vec<&str> = text.lines().collect();
    let line_truncated = lines.len() > MAX_FILE_LINES;
    let shown = lines
        .iter()
        .take(MAX_FILE_LINES)
        .cloned()
        .collect::<Vec<_>>()
        .join("\n");
    let mut out = format!("<file path=\"{path}\">\n{shown}");
    if line_truncated || byte_truncated {
        out.push_str("\n… truncated — use read_file to see the rest");
    }
    out.push_str("</file>");
    out
}

#[cfg(test)]
mod test {
    use super::*;
    use std::fs;

    #[test]
    fn token_at_finds_the_token_ending_at_the_cursor() {
        assert_eq!(token_at("hello /help", 11), Some((6, "/help".into())));
        assert_eq!(token_at("hello /help", 9), Some((6, "/he".into())));
        assert_eq!(token_at("hello ", 6), None);
        assert_eq!(token_at("", 0), None);
    }

    #[test]
    fn command_candidates_filter_by_strict_prefix() {
        let cwd = std::env::current_dir().unwrap();
        let out = candidates("/h", 2, &cwd).unwrap();
        assert_eq!(out, vec![Suggestion {
            insert: "/help".into(),
            display: "help — show this message".into(),
        }]);
        assert_eq!(candidates("/help", 5, &cwd), Some(Vec::new()));
        assert_eq!(candidates("help", 4, &cwd), None);
        assert_eq!(candidates("/asdf", 5, &cwd), Some(Vec::new()));
    }

    #[test]
    fn skill_candidates_filter_by_strict_prefix() {
        let skills = vec![
            crate::skill::Skill {
                name: "git-commit".into(),
                description: "write a commit message".into(),
                path: std::path::PathBuf::from("/tmp/skills/git-commit/SKILL.md"),
            },
            crate::skill::Skill {
                name: "git-push".into(),
                description: "push to the remote".into(),
                path: std::path::PathBuf::from("/tmp/skills/git-push/SKILL.md"),
            },
            crate::skill::Skill {
                name: "deploy".into(),
                description: "ship to prod".into(),
                path: std::path::PathBuf::from("/tmp/skills/deploy/SKILL.md"),
            },
        ];
        assert_eq!(
            skill_candidates("gi", &skills),
            vec![
                Suggestion {
                    insert: "/skill:git-commit".into(),
                    display: "git-commit — write a commit message".into(),
                },
                Suggestion {
                    insert: "/skill:git-push".into(),
                    display: "git-push — push to the remote".into(),
                },
            ]
        );
        assert_eq!(skill_candidates("git-commit", &skills), Vec::new());
        assert_eq!(skill_candidates("nope", &skills), Vec::new());
    }

    #[test]
    fn skill_candidates_an_empty_prefix_lists_all_capped() {
        let skills: Vec<crate::skill::Skill> = (0..8)
            .map(|i| crate::skill::Skill {
                name: format!("skill{i:02}"),
                description: format!("d{i}"),
                path: std::path::PathBuf::from(format!("/tmp/skills/skill{i:02}/SKILL.md")),
            })
            .collect();
        let out = skill_candidates("", &skills);
        assert_eq!(out.len(), MAX_POPUP_CANDIDATES);
        assert_eq!(
            out[0],
            Suggestion {
                insert: "/skill:skill00".into(),
                display: "skill00 — d0".into(),
            }
        );
    }

    #[test]
    fn file_candidates_walk_the_working_directory() {
        let dir = std::env::temp_dir().join(format!("kite_autocomplete_{}", std::process::id()));
        let nested = dir.join("src");
        fs::create_dir_all(&nested).unwrap();
        fs::write(dir.join("alpha.txt"), "a").unwrap();
        fs::write(nested.join("main.rs"), "fn main() {}").unwrap();
        fs::write(dir.join("beta.rs"), "b").unwrap();
        let hidden = dir.join(".hidden");
        fs::create_dir_all(&hidden).unwrap();
        fs::write(hidden.join("secret.rs"), "s").unwrap();
        let target = dir.join("target");
        fs::create_dir_all(&target).unwrap();
        fs::write(target.join("skipped.rs"), "t").unwrap();

        let out = candidates("@al", 3, &dir).unwrap();
        assert_eq!(
            out,
            vec![Suggestion {
                insert: "@alpha.txt".into(),
                display: "alpha.txt".into(),
            }]
        );
        let out = candidates("@", 1, &dir).unwrap();
        let displays: Vec<&str> = out.iter().map(|s| s.display.as_str()).collect();
        assert!(displays.contains(&"alpha.txt"));
        assert!(displays.contains(&"src/main.rs"));
        assert!(!displays.contains(&".hidden/secret.rs"));
        assert!(!displays.contains(&"target/skipped.rs"));

        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn apply_replaces_the_token_and_parks_the_cursor() {
        let s = Suggestion {
            insert: "/help".into(),
            display: "help".into(),
        };
        assert_eq!(apply("do /h", 5, &s), ("do /help".into(), 8));
    }

    #[test]
    fn expand_file_refs_inlines_existing_files_only() {
        let dir = std::env::temp_dir().join(format!("kite_expand_{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("a.txt"), "alpha content").unwrap();

        let out = expand_file_refs("read @a.txt and @missing.txt and a@b.com", &dir);
        assert!(out.starts_with("read <file path=\"a.txt\">\nalpha content</file>"));
        assert!(out.contains("and @missing.txt and a@b.com"));

        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn expand_file_refs_truncates_long_files() {
        let dir = std::env::temp_dir().join(format!("kite_truncate_{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let content: String = (0..600).map(|i| format!("line {i}")).collect::<Vec<_>>().join("\n");
        fs::write(dir.join("big.txt"), content).unwrap();

        let out = expand_file_refs("@big.txt", &dir);
        assert!(out.contains("line 499"));
        assert!(!out.contains("line 500"));
        assert!(out.contains("… truncated — use read_file to see the rest"));

        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn expand_file_refs_flags_binary_files() {
        let dir = std::env::temp_dir().join(format!("kite_binary_{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("bin.dat"), vec![0u8, 1, 2, 3]).unwrap();

        let out = expand_file_refs("@bin.dat", &dir);
        assert!(out.contains("[binary file (4 bytes)]"));

        fs::remove_dir_all(&dir).unwrap();
    }

}
