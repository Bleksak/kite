use std::path::Path;

use crate::context::Context;

#[derive(serde::Serialize, serde::Deserialize)]
pub struct SessionFile {
    pub label: String,
    pub context: Context,
}

fn sessions_dir(kite_dir: &Path) -> std::path::PathBuf {
    kite_dir.join("sessions")
}

pub fn save_session(kite_dir: &Path, id: u64, label: &str, context: &Context) {
    let dir = sessions_dir(kite_dir);
    if std::fs::create_dir_all(&dir).is_ok() {
        let file = SessionFile {
            label: label.to_string(),
            context: context.clone(),
        };
        if let Ok(json) = serde_json::to_string_pretty(&file) {
            let _ = std::fs::write(dir.join(format!("{id}.json")), json);
        }
    }
}

pub fn remove_session_file(kite_dir: &Path, id: u64) {
    let _ = std::fs::remove_file(sessions_dir(kite_dir).join(format!("{id}.json")));
}

pub fn session_file_exists(kite_dir: &Path, id: u64) -> bool {
    sessions_dir(kite_dir).join(format!("{id}.json")).exists()
}

pub fn load_sessions(kite_dir: &Path) -> Vec<(u64, SessionFile)> {
    let mut loaded = Vec::new();
    let entries = match std::fs::read_dir(sessions_dir(kite_dir)) {
        Ok(entries) => entries,
        Err(_) => return loaded,
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().is_some_and(|ext| ext == "json")
            && let Some(stem) = path.file_stem().and_then(|s| s.to_str())
            && let Ok(id) = stem.parse::<u64>()
            && let Ok(json) = std::fs::read_to_string(&path)
            && let Ok(file) = serde_json::from_str::<SessionFile>(&json)
        {
            loaded.push((id, file));
        }
    }
    loaded.sort_by_key(|(id, _)| *id);
    loaded
}

#[cfg(test)]
mod test {
    use super::{load_sessions, remove_session_file, save_session};
    use crate::context::Context;
    use crate::message::Message;

    fn temp_dir(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("kite-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        dir
    }

    #[test]
    fn session_file_round_trip() {
        let dir = temp_dir("session-test");

        let mut context = Context::new("sys prompt", 100);
        context.messages.push(Message::User { content: "hello".into() });
        context.messages.push(Message::Assistant {
            content: Some("hi".into()),
            tool_calls: vec![],
        });
        context.total_prompt_tokens = 120;
        context.total_completion_tokens = 34;

        save_session(&dir, 7, "fix login", &context);
        let loaded = load_sessions(&dir);

        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].0, 7);
        assert_eq!(loaded[0].1.label, "fix login");
        assert_eq!(loaded[0].1.context.messages.len(), 2);
        assert_eq!(loaded[0].1.context.total_prompt_tokens, 120);
        assert_eq!(loaded[0].1.context.total_completion_tokens, 34);

        remove_session_file(&dir, 7);
        assert!(load_sessions(&dir).is_empty());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn load_sessions_ignores_corrupt_and_unnamed_files() {
        let dir = temp_dir("session-bad");
        let sessions = dir.join("sessions");
        std::fs::create_dir_all(&sessions).unwrap();
        std::fs::write(sessions.join("3.json"), "not json").unwrap();
        std::fs::write(sessions.join("notes.txt"), "{}").unwrap();
        std::fs::write(sessions.join("x.json"), "{}").unwrap();

        assert!(load_sessions(&dir).is_empty());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn load_sessions_missing_dir_is_empty() {
        let dir = temp_dir("session-missing");
        assert!(load_sessions(&dir).is_empty());
    }
}
