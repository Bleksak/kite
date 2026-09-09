use std::collections::HashSet;
use std::path::{Path, PathBuf};

use crate::paths::CONTEXT_DIR;

pub const MAX_SKILLS: usize = 20;
const MAX_DESCRIPTION_CHARS: usize = 200;

#[derive(Debug, Clone)]
pub struct Skill {
    pub name: String,
    pub description: String,
    pub path: PathBuf,
}

impl Skill {
    fn make(dir_name: &str, path: &Path, content: &str) -> Option<Self> {
        let (meta, _) = frontmatter::parse_and_find_content(content).ok()?;
        let meta = meta?;
        let description = meta["description"].as_str()?.trim().to_string();
        if description.is_empty() {
            return None;
        }
        let name = meta["name"]
            .as_str()
            .map(str::trim)
            .filter(|name| !name.is_empty())
            .unwrap_or(dir_name)
            .to_string();
        Some(Self {
            name,
            description: truncate_chars(&description, MAX_DESCRIPTION_CHARS),
            path: path.to_path_buf(),
        })
    }
}

#[derive(Debug, thiserror::Error)]
pub enum SkillError {
    #[error("skill {0} not found")]
    NotFound(String),

    #[error(transparent)]
    Io(#[from] std::io::Error),
}

pub fn load_skills() -> Vec<Skill> {
    let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let project = cwd.join(CONTEXT_DIR).join("skills");
    let user = dirs::home_dir()
        .map(|home| home.join(CONTEXT_DIR).join("skills"))
        .unwrap_or_default();
    load_skills_from(&project, &user)
}

pub fn load_skills_from(project: &Path, user: &Path) -> Vec<Skill> {
    let mut skills = scan_root(project);
    for skill in scan_root(user) {
        if !skills.iter().any(|existing: &Skill| existing.name == skill.name) {
            skills.push(skill);
        }
    }
    skills.sort_by(|a, b| a.name.cmp(&b.name));
    skills.truncate(MAX_SKILLS);
    skills
}

pub fn skill_section(skills: &[Skill]) -> Option<String> {
    if skills.is_empty() {
        return None;
    }
    Some(
        skills
            .iter()
            .map(|skill| {
                format!(
                    "- {}: {} — use_skill \"{}\" loads it (files: .kite/skills/{}/)",
                    skill.name, skill.description, skill.name, skill.name
                )
            })
            .collect::<Vec<_>>()
            .join("\n"),
    )
}

pub fn load_skill_body(name: &str) -> Result<String, SkillError> {
    let cwd = std::env::current_dir()?;
    let project = cwd.join(CONTEXT_DIR).join("skills");
    let user = dirs::home_dir().map(|home| home.join(CONTEXT_DIR).join("skills"));
    load_skill_body_from(name, &project, user.as_deref())
}

fn load_skill_body_from(
    name: &str,
    project: &Path,
    user: Option<&Path>,
) -> Result<String, SkillError> {
    for root in [Some(project), user] {
        if let Some(root) = root {
            let path = root.join(name).join("SKILL.md");
            if path.is_file() {
                let content = std::fs::read_to_string(path)?;
                return Ok(frontmatter::parse_and_find_content(&content)
                    .map(|(_, body)| body.to_string())
                    .unwrap_or_else(|_| content));
            }
        }
    }
    Err(SkillError::NotFound(name.to_string()))
}

// Mirrors pi's _expandSkillCommand: a leading /skill:name [args] expands to
// the skill's full SKILL.md body inlined in a <skill> block. Unknown skills
// and unreadable files pass through unchanged. No line/byte caps.
pub fn expand_skill_command(text: &str, skills: &[Skill]) -> String {
    let Some(rest) = text.strip_prefix("/skill:") else {
        return text.to_string();
    };
    let (name, args) = match rest.find(char::is_whitespace) {
        Some(pos) => (rest[..pos].to_string(), rest[pos..].trim().to_string()),
        None => (rest.to_string(), String::new()),
    };
    let Some(skill) = skills.iter().find(|skill| skill.name == name) else {
        return text.to_string(); // Unknown skill, pass through
    };
    let content = match std::fs::read_to_string(&skill.path) {
        Ok(content) => content,
        Err(_) => return text.to_string(), // Read error, pass through
    };
    let body = frontmatter::parse_and_find_content(&content)
        .map(|(_, body)| body.to_string())
        .unwrap_or_else(|_| content)
        .trim()
        .to_string();
    let base_dir = skill
        .path
        .parent()
        .map(|parent| parent.display().to_string())
        .unwrap_or_default();
    let block = format!(
        "<skill name=\"{}\" location=\"{}\">\nReferences are relative to {}.\n\n{}\n</skill>",
        skill.name, skill.path.display(), base_dir, body
    );
    if args.is_empty() {
        block
    } else {
        format!("{block}\n\n{args}")
    }
}

fn scan_root(root: &Path) -> Vec<Skill> {
    let entries = match std::fs::read_dir(root) {
        Ok(entries) => entries,
        Err(_) => return Vec::new(),
    };
    let mut dirs: Vec<_> = entries.filter_map(|entry| entry.ok()).collect();
    dirs.sort_by(|a, b| a.file_name().cmp(&b.file_name()));
    let mut seen = HashSet::new();
    let mut skills = Vec::new();
    for entry in dirs {
        if !entry.file_type().is_ok_and(|ty| ty.is_dir()) {
            continue;
        }
        let dir_name = entry.file_name().to_string_lossy().into_owned();
        let path = entry.path().join("SKILL.md");
        let content = match std::fs::read_to_string(&path) {
            Ok(content) => content,
            Err(_) => continue,
        };
        let Some(skill) = Skill::make(&dir_name, &path, &content) else {
            continue;
        };
        if seen.insert(skill.name.clone()) {
            skills.push(skill);
        }
    }
    skills
}

fn truncate_chars(value: &str, max_chars: usize) -> String {
    if value.chars().count() <= max_chars {
        return value.to_string();
    }
    value.chars().take(max_chars).collect()
}

#[cfg(test)]
mod test {
    use super::*;
    use std::fs;
    use tempfile::tempdir;

    fn write_skill(root: &Path, dir_name: &str, content: &str) {
        let dir = root.join(dir_name);
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("SKILL.md"), content).unwrap();
    }

    fn roots() -> (tempfile::TempDir, PathBuf, PathBuf) {
        let tmp = tempdir().unwrap();
        let project = tmp.path().join("project");
        let user = tmp.path().join("user");
        fs::create_dir_all(&project).unwrap();
        fs::create_dir_all(&user).unwrap();
        (tmp, project, user)
    }

    #[test]
    fn frontmatter_parses_name_and_description() {
        let (_tmp, project, user) = roots();
        write_skill(
            &project,
            "deploy",
            "---\nname: deploy\ndescription: deploys the app to prod\n---\n# deploy\n",
        );
        let skills = load_skills_from(&project, &user);
        assert_eq!(skills.len(), 1);
        assert_eq!(skills[0].name, "deploy");
        assert_eq!(skills[0].description, "deploys the app to prod");
        assert_eq!(
            skills[0].path,
            project.join("deploy").join("SKILL.md")
        );
    }

    #[test]
    fn name_falls_back_to_the_directory_name() {
        let (_tmp, project, user) = roots();
        write_skill(
            &project,
            "migrate",
            "---\ndescription: migrates the database\n---\n",
        );
        let skills = load_skills_from(&project, &user);
        assert_eq!(skills.len(), 1);
        assert_eq!(skills[0].name, "migrate");
    }

    #[test]
    fn malformed_files_are_skipped_silently() {
        let (_tmp, project, user) = roots();
        write_skill(&project, "no_frontmatter", "just a markdown file\n");
        write_skill(&project, "unterminated", "---\nname: x\ndescription: y\n");
        write_skill(&project, "missing_description", "---\nname: x\n---\n");
        write_skill(
            &project,
            "empty_description",
            "---\nname: x\ndescription: \"\"\n---\n",
        );
        write_skill(&project, "bad_yaml", "---\nname: [unclosed\ndescription: y\n---\n");
        write_skill(&project, "good", "---\nname: good\ndescription: works\n---\n");
        let skills = load_skills_from(&project, &user);
        assert_eq!(skills.len(), 1);
        assert_eq!(skills[0].name, "good");
    }

    #[test]
    fn a_project_skill_shadows_a_user_skill_with_the_same_name() {
        let (_tmp, project, user) = roots();
        write_skill(
            &project,
            "shared",
            "---\nname: shared\ndescription: from project\n---\n",
        );
        write_skill(
            &user,
            "shared",
            "---\nname: shared\ndescription: from user\n---\n",
        );
        write_skill(
            &user,
            "user_only",
            "---\nname: user_only\ndescription: only in user\n---\n",
        );
        let skills = load_skills_from(&project, &user);
        assert_eq!(skills.len(), 2);
        let shared = skills
            .iter()
            .find(|skill| skill.name == "shared")
            .unwrap();
        assert_eq!(shared.description, "from project");
        assert!(shared.path.starts_with(&project));
        assert_eq!(
            skills.iter().filter(|skill| skill.name == "user_only").count(),
            1
        );
    }

    #[test]
    fn missing_roots_yield_an_empty_list() {
        let tmp = tempdir().unwrap();
        let skills = load_skills_from(
            &tmp.path().join("does-not-exist"),
            &tmp.path().join("nor-this"),
        );
        assert!(skills.is_empty());
    }

    #[test]
    fn skills_are_sorted_by_name() {
        let (_tmp, project, user) = roots();
        write_skill(&project, "zeta", "---\nname: zeta\ndescription: z\n---\n");
        write_skill(&project, "alpha", "---\nname: alpha\ndescription: a\n---\n");
        write_skill(&project, "mid", "---\nname: mid\ndescription: m\n---\n");
        let names: Vec<_> = load_skills_from(&project, &user)
            .into_iter()
            .map(|skill| skill.name)
            .collect();
        assert_eq!(names, vec!["alpha", "mid", "zeta"]);
    }

    #[test]
    fn descriptions_are_truncated_to_200_chars() {
        let (_tmp, project, user) = roots();
        write_skill(
            &project,
            "long",
            &format!(
                "---\nname: long\ndescription: {}\n---\n",
                "x".repeat(300)
            ),
        );
        let skills = load_skills_from(&project, &user);
        assert_eq!(skills[0].description, "x".repeat(200));
    }

    #[test]
    fn truncation_never_splits_a_multibyte_char() {
        let (_tmp, project, user) = roots();
        write_skill(
            &project,
            "long",
            &format!(
                "---\nname: long\ndescription: {}\n---\n",
                "é".repeat(300)
            ),
        );
        let skills = load_skills_from(&project, &user);
        assert_eq!(skills[0].description.chars().count(), 200);
        assert!(skills[0].description.chars().all(|c| c == 'é'));
    }

    #[test]
    fn the_list_is_capped_at_20_skills() {
        let (_tmp, project, user) = roots();
        for i in 0..25 {
            write_skill(
                &project,
                &format!("skill{i:02}"),
                &format!("---\nname: skill{i:02}\ndescription: d{i}\n---\n"),
            );
        }
        let skills = load_skills_from(&project, &user);
        assert_eq!(skills.len(), 20);
        assert_eq!(skills.first().unwrap().name, "skill00");
        assert_eq!(skills.last().unwrap().name, "skill19");
    }

    #[test]
    fn skill_section_renders_one_line_per_skill() {
        let skills = vec![Skill {
            name: "deploy".into(),
            description: "deploys the app".into(),
            path: PathBuf::from("/home/u/.kite/skills/deploy/SKILL.md"),
        }];
        assert_eq!(
            skill_section(&skills),
            Some(
                "- deploy: deploys the app — use_skill \"deploy\" loads it (files: .kite/skills/deploy/)"
                    .into()
            )
        );
    }

    #[test]
    fn skill_section_is_none_without_skills() {
        assert_eq!(skill_section(&[]), None);
    }

    #[test]
    fn body_read_picks_up_edits_without_a_restart() {
        let (_tmp, project, user) = roots();
        write_skill(
            &project,
            "deploy",
            "---\nname: deploy\ndescription: d\n---\nold body",
        );
        assert_eq!(
            load_skill_body_from("deploy", &project, Some(&user)).unwrap(),
            "old body"
        );
        fs::write(
            project.join("deploy").join("SKILL.md"),
            "---\nname: deploy\ndescription: d\n---\nnew body",
        )
        .unwrap();
        assert_eq!(
            load_skill_body_from("deploy", &project, Some(&user)).unwrap(),
            "new body"
        );
    }

    #[test]
    fn body_read_prefers_project_over_user() {
        let (_tmp, project, user) = roots();
        write_skill(
            &project,
            "shared",
            "---\nname: shared\ndescription: d\n---\nproject body",
        );
        write_skill(
            &user,
            "shared",
            "---\nname: shared\ndescription: d\n---\nuser body",
        );
        assert_eq!(
            load_skill_body_from("shared", &project, Some(&user)).unwrap(),
            "project body"
        );
    }

    #[test]
    fn body_read_falls_back_to_user() {
        let (_tmp, project, user) = roots();
        write_skill(
            &user,
            "user_only",
            "---\nname: user_only\ndescription: d\n---\nuser body",
        );
        assert_eq!(
            load_skill_body_from("user_only", &project, Some(&user)).unwrap(),
            "user body"
        );
    }

    #[test]
    fn body_read_of_an_unknown_name_is_not_found() {
        let (_tmp, project, user) = roots();
        let err = load_skill_body_from("missing", &project, Some(&user)).unwrap_err();
        assert!(matches!(err, SkillError::NotFound(name) if name == "missing"));
    }

    fn skill_for_test(tmp: &tempfile::TempDir, name: &str, content: &str) -> Skill {
        let path = tmp.path().join("skills").join(name).join("SKILL.md");
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(&path, content).unwrap();
        Skill {
            name: name.into(),
            description: "d".into(),
            path,
        }
    }

    #[test]
    fn expand_inlines_a_leading_skill_command_with_args() {
        let tmp = tempdir().unwrap();
        let skill = skill_for_test(
            &tmp,
            "deploy",
            "---\nname: deploy\ndescription: d\n---\n# deploy\nstep one\nstep two\n",
        );
        let out = expand_skill_command(
            "/skill:deploy ship to prod",
            &[skill.clone()],
        );
        assert_eq!(
            out,
            format!(
                "<skill name=\"deploy\" location=\"{}\">\n\
                 References are relative to {}.\n\n\
                 # deploy\nstep one\nstep two\n\
                 </skill>\n\n\
                 ship to prod",
                skill.path.display(),
                skill.path.parent().unwrap().display()
            )
        );
    }

    #[test]
    fn expand_passes_an_unknown_skill_through_unchanged() {
        let tmp = tempdir().unwrap();
        let skill = skill_for_test(&tmp, "deploy", "---\nname: deploy\ndescription: d\n---\nbody");
        let out = expand_skill_command(
            "/skill:nope do the thing",
            &[skill],
        );
        assert_eq!(out, "/skill:nope do the thing");
    }

    #[test]
    fn expand_ignores_a_non_leading_skill_token() {
        let tmp = tempdir().unwrap();
        let skill = skill_for_test(&tmp, "deploy", "---\nname: deploy\ndescription: d\n---\nbody");
        let out = expand_skill_command(
            "please use /skill:deploy now",
            &[skill],
        );
        assert_eq!(out, "please use /skill:deploy now");
    }

    #[test]
    fn expand_a_bare_skill_name_yields_just_the_block() {
        let tmp = tempdir().unwrap();
        let skill = skill_for_test(&tmp, "deploy", "---\nname: deploy\ndescription: d\n---\nbody");
        let out = expand_skill_command("/skill:deploy", &[skill.clone()]);
        assert_eq!(
            out,
            format!(
                "<skill name=\"deploy\" location=\"{}\">\n\
                 References are relative to {}.\n\n\
                 body\n\
                 </skill>",
                skill.path.display(),
                skill.path.parent().unwrap().display()
            )
        );
    }

    #[test]
    fn expand_passes_through_when_the_file_is_unreadable() {
        let skill = Skill {
            name: "deploy".into(),
            description: "d".into(),
            path: PathBuf::from("/no/such/skill/SKILL.md"),
        };
        let out = expand_skill_command("/skill:deploy args", &[skill]);
        assert_eq!(out, "/skill:deploy args");
    }

    #[test]
    fn expand_inlines_the_full_body_without_caps() {
        let tmp = tempdir().unwrap();
        let body = "line\n".repeat(10_000);
        let skill = skill_for_test(
            &tmp,
            "big",
            &format!("---\nname: big\ndescription: d\n---\n{body}"),
        );
        let out = expand_skill_command("/skill:big go", &[skill]);
        assert!(out.contains(&"line\n".repeat(10_000)));
        assert_eq!(out.matches("line\n").count(), 10_000);
        assert!(out.ends_with("go"));
    }
}
