use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::{Arc, LazyLock, Mutex};
use std::time::{Duration, Instant};
use tokio::io::AsyncReadExt;

pub static REGISTRY: LazyLock<BgRegistry> = LazyLock::new(BgRegistry::new);

static INSTANCE: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

struct BgState {
    tasks: HashMap<String, BgTask>,
    next_id: u32,
}

pub struct BgRegistry {
    state: Arc<Mutex<BgState>>,
    dir_name: String,
    finished: tokio::sync::broadcast::Sender<String>,
}

impl BgRegistry {
    pub fn new() -> BgRegistry {
        let dir_name = format!(
            "{}-{}",
            std::process::id(),
            INSTANCE.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        );
        let (finished, _) = tokio::sync::broadcast::channel(16);
        BgRegistry {
            state: Arc::new(Mutex::new(BgState {
                tasks: HashMap::new(),
                next_id: 1,
            })),
            dir_name,
            finished,
        }
    }

    pub fn subscribe(&self) -> tokio::sync::broadcast::Receiver<String> {
        self.finished.subscribe()
    }

    pub fn run(&self, command: &str) -> Result<String, Box<dyn std::error::Error>> {
        let mut child = tokio::process::Command::new("bash")
            .arg("-c")
            .arg(command)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()?;
        let (id, output_path) = {
            let mut state = self.state.lock().unwrap();
            let id = state.next_id.to_string();
            state.next_id += 1;
            let dir = Path::new(crate::paths::CONTEXT_DIR).join("tasks").join(&self.dir_name);
            std::fs::create_dir_all(&dir)?;
            let output_path = dir.join(format!("{id}.log"));
            state.tasks.insert(
                id.clone(),
                BgTask {
                    id: id.clone(),
                    command: command.to_string(),
                    pid: child.id(),
                    output_path: output_path.clone(),
                    status: BgStatus::Running,
                    started_at: Instant::now(),
                    finished_at: None,
                },
            );
            (id, output_path)
        };
        let state = self.state.clone();
        let finished = self.finished.clone();
        let return_id = id.clone();
        tokio::spawn(async move {
            let mut stdout = child.stdout.take();
            let mut stderr = child.stderr.take();
            let read = async {
                let mut out = Vec::new();
                let mut err = Vec::new();
                if let Some(stream) = stdout.as_mut() {
                    let _ = stream.read_to_end(&mut out).await;
                }
                if let Some(stream) = stderr.as_mut() {
                    let _ = stream.read_to_end(&mut err).await;
                }
                (out, err)
            };
            tokio::pin!(read);
            let wait = child.wait();
            tokio::pin!(wait);
            let (code, out, err) = tokio::select! {
                status = wait.as_mut() => {
                    let code = status.ok().and_then(|s| s.code());
                    let (out, err) = tokio::time::timeout(
                        Duration::from_secs(2),
                        read.as_mut(),
                    )
                    .await
                    .unwrap_or_default();
                    (code, out, err)
                }
                (out, err) = read.as_mut() => {
                    let code = wait.as_mut().await.ok().and_then(|s| s.code());
                    (code, out, err)
                }
            };
            let mut text = String::from_utf8_lossy(&out).into_owned();
            let stderr_text = String::from_utf8_lossy(&err).into_owned();
            if !stderr_text.is_empty() {
                if !text.is_empty() {
                    text.push('\n');
                }
                text.push_str("--- stderr ---\n");
                text.push_str(&stderr_text);
            }
            let _ = std::fs::write(&output_path, text);
            if let Some(task) = state.lock().unwrap().tasks.get_mut(&id) {
                task.status = BgStatus::Finished(code);
                task.finished_at = Some(Instant::now());
            }
            let _ = finished.send(id);
        });
        Ok(return_id)
    }

    pub fn list(&self) -> Vec<BgTask> {
        let mut tasks: Vec<BgTask> = self.state.lock().unwrap().tasks.values().cloned().collect();
        tasks.sort_by_key(|task| task.id.parse::<u32>().unwrap_or(0));
        tasks
    }

    pub fn kill(&self, id: &str) -> Result<(), String> {
        let state = self.state.lock().unwrap();
        let task = state
            .tasks
            .get(id)
            .ok_or_else(|| format!("unknown task {id}"))?;
        if matches!(task.status, BgStatus::Finished(_)) {
            return Err(format!("task {id} already finished"));
        }
        let pid = task.pid.ok_or_else(|| format!("task {id} has no process"))?;
        drop(state);
        let status = std::process::Command::new("kill")
            .args(["-9", &pid.to_string()])
            .status()
            .map_err(|source| source.to_string())?;
        if !status.success() {
            return Err(format!("kill {pid} failed"));
        }
        Ok(())
    }

    pub fn due_reports(&self, seen: &mut HashSet<String>) -> Vec<BgReport> {
        let state = self.state.lock().unwrap();
        let ids: Vec<String> = state
            .tasks
            .values()
            .filter(|task| matches!(task.status, BgStatus::Finished(_)) && !seen.contains(&task.id))
            .map(|task| task.id.clone())
            .collect();
        ids.into_iter()
            .filter_map(|id| {
                let task = state.tasks.get(&id)?;
                seen.insert(id);
                Some(BgReport {
                    id: task.id.clone(),
                    command: task.command.clone(),
                    code: match &task.status {
                        BgStatus::Finished(code) => *code,
                        BgStatus::Running => None,
                    },
                    output_path: task.output_path.clone(),
                })
            })
            .collect()
    }
}

impl Default for BgRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Clone)]
pub struct BgTask {
    pub id: String,
    pub command: String,
    pub pid: Option<u32>,
    pub output_path: PathBuf,
    pub status: BgStatus,
    pub started_at: Instant,
    pub finished_at: Option<Instant>,
}

#[derive(Clone)]
pub enum BgStatus {
    Running,
    Finished(Option<i32>),
}

pub struct BgReport {
    pub id: String,
    pub command: String,
    pub code: Option<i32>,
    pub output_path: PathBuf,
}

impl BgReport {
    pub fn status_line(&self) -> String {
        match self.code {
            Some(0) => "exit 0".to_string(),
            Some(code) => format!("exit {code}"),
            None => "killed".to_string(),
        }
    }

    pub fn tail(&self, limit: usize) -> String {
        let text = std::fs::read_to_string(&self.output_path).unwrap_or_default();
        if text.len() <= limit {
            return text;
        }
        let mut start = text.len() - limit;
        while !text.is_char_boundary(start) {
            start += 1;
        }
        format!("[truncated]\n{}", &text[start..])
    }
}

pub fn format_duration(duration: Duration) -> String {
    let seconds = duration.as_secs();
    format!("{}:{:02}", seconds / 60, seconds % 60)
}

#[cfg(test)]
mod test {
    use super::*;

    async fn wait_finished(registry: &BgRegistry, id: &str) -> bool {
        for _ in 0..200 {
            if matches!(
                registry.state.lock().unwrap().tasks.get(id).unwrap().status,
                BgStatus::Finished(_)
            ) {
                return true;
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
        false
    }

    #[tokio::test]
    async fn bg_run_completes_and_reports_once() {
        let registry = BgRegistry::new();
        let id = registry.run("echo hello").unwrap();
        let mut seen = HashSet::new();
        assert!(wait_finished(&registry, &id).await);
        let reports = registry.due_reports(&mut seen);
        assert_eq!(reports.len(), 1);
        assert_eq!(reports[0].code, Some(0));
        assert!(reports[0].tail(4096).contains("hello"));
        assert!(registry.due_reports(&mut seen).is_empty());
    }

    #[tokio::test]
    async fn bg_run_reports_nonzero_exit() {
        let registry = BgRegistry::new();
        let id = registry.run("exit 3").unwrap();
        let mut seen = HashSet::new();
        assert!(wait_finished(&registry, &id).await);
        let reports = registry.due_reports(&mut seen);
        assert_eq!(reports[0].code, Some(3));
        assert_eq!(reports[0].status_line(), "exit 3");
    }

    #[tokio::test]
    async fn bg_run_writes_output_file() {
        let registry = BgRegistry::new();
        let id = registry.run("echo to-stdout; echo to-stderr >&2").unwrap();
        let mut seen = HashSet::new();
        assert!(wait_finished(&registry, &id).await);
        let reports = registry.due_reports(&mut seen);
        let tail = reports[0].tail(4096);
        assert!(tail.contains("to-stdout"));
        assert!(tail.contains("to-stderr"));
        assert!(tail.contains("--- stderr ---"));
    }

    #[tokio::test]
    async fn kill_running_task_and_refuse_finished() {
        let registry = BgRegistry::new();
        let id = registry.run("sleep 30").unwrap();
        registry.kill(&id).unwrap();
        let mut seen = HashSet::new();
        assert!(wait_finished(&registry, &id).await);
        let reports = registry.due_reports(&mut seen);
        assert_eq!(reports[0].code, None);
        assert_eq!(reports[0].status_line(), "killed");
        assert!(registry.kill(&id).is_err());
        assert!(registry.kill("nope").is_err());
    }

    #[tokio::test]
    async fn finished_signal_fires_on_completion() {
        let registry = BgRegistry::new();
        let mut rx = registry.subscribe();
        let id = registry.run("echo signal-test").unwrap();
        let received = tokio::time::timeout(Duration::from_secs(10), rx.recv()).await;
        assert!(received.is_ok());
        assert!(wait_finished(&registry, &id).await);
    }

    #[test]
    fn format_duration_formats_minutes_and_seconds() {
        assert_eq!(format_duration(Duration::from_secs(5)), "0:05");
        assert_eq!(format_duration(Duration::from_secs(125)), "2:05");
    }
}
