use crate::agent::Agent;
use crate::context::Context;
use crate::mode::Mode;
use crate::tui::TuiEvent;


fn stage_implement_message(
    stages: &Option<Vec<crate::tool::PlanStage>>,
    index: usize,
) -> String {
    let Some(stages) = stages.as_ref() else {
        return "Execute the plan.".to_string();
    };
    let stage = &stages[index];
    let done = if index > 0 {
        let done = stages[..index]
            .iter()
            .enumerate()
            .map(|(i, s)| format!("Step {}: {}", i + 1, s.title))
            .collect::<Vec<_>>()
            .join("; ");
        format!("Stages already done: {done}.\n")
    } else {
        String::new()
    };
    format!(
        "Execute Step {} of {}: {}\n{}\n{}Do not start later stages.",
        index + 1,
        stages.len(),
        stage.title,
        stage.description,
        done
    )
}

fn stage_reimplement_message(
    stages: &Option<Vec<crate::tool::PlanStage>>,
    index: usize,
    feedback: &str,
) -> String {
    let Some(stages) = stages.as_ref() else {
        return format!(
            "The implementation was rejected. Feedback: {feedback}\n\nRe-implement the stage, addressing the feedback."
        );
    };
    let stage = &stages[index];
    format!(
        "Re-implement Step {} of {}: {}\n{}\n\nThe previous implementation was rejected. Feedback: {feedback}\n\nAddress the feedback, keeping what is already done, and re-implement this stage. Do not start later stages.",
        index + 1,
        stages.len(),
        stage.title,
        stage.description
    )
}

fn stage_replan_message(
    stages: &Option<Vec<crate::tool::PlanStage>>,
    feedback: &str,
) -> String {
    let Some(stages) = stages.as_ref() else {
        return format!(
            "The plan was rejected. Feedback: {feedback}\n\nRevise the plan and call submit_plan."
        );
    };
    format!(
        "The plan was rejected. Feedback: {feedback}\n\nOriginal plan:\n{}\n\nRevise the plan and call submit_plan.",
        crate::tool::plan_text(stages)
    )
}

fn stage_step_replan_message(
    stages: &Option<Vec<crate::tool::PlanStage>>,
    index: usize,
    feedback: &str,
) -> String {
    let Some(stages) = stages.as_ref() else {
        return format!(
            "The step was rejected. Feedback: {feedback}\n\nRe-plan the step and call submit_plan."
        );
    };
    let stage = &stages[index];
    let others = stages
        .iter()
        .enumerate()
        .filter(|(i, _)| *i != index)
        .map(|(i, s)| format!("Step {}: {}\n{}", i + 1, s.title, s.description))
        .collect::<Vec<_>>()
        .join("\n");
    let others_note = if others.is_empty() {
        String::new()
    } else {
        format!("Keep the other stages unchanged:\n{others}\n")
    };
    format!(
        "Step {} of {}: {}\n{}\n\nThe step was rejected. Feedback: {feedback}\n\nRe-plan only this step. {others_note}Call submit_plan with all stages, changing only this step.",
        index + 1,
        stages.len(),
        stage.title,
        stage.description
    )
}

#[derive(Clone, Copy, PartialEq)]
enum GatePhase {
    ReviewPlan,
    ReviewStep,
    ReviewImplementation,
}

async fn handle_outcome(
    agent: &mut Agent,
    new_agent: &std::sync::Arc<dyn Fn() -> Agent + Send + Sync>,
    event_tx: &tokio::sync::mpsc::UnboundedSender<TuiEvent>,
    id: u64,
    gate: &mut Option<(String, GatePhase)>,
    stages: &mut Option<Vec<crate::tool::PlanStage>>,
    stage_index: &mut usize,
    from_gate: Option<GatePhase>,
    mut outcome: crate::agent::ChatOutcome,
    cancel: &std::sync::Arc<tokio::sync::watch::Receiver<u64>>,
    steering: &crate::agent::Steering,
) {
    loop {
        match &outcome {
            crate::agent::ChatOutcome::Terminated {
                tool: crate::tool::Tool::SubmitPlan(parsed),
            } => {
                *stages = Some(parsed.clone());
                let per_step = matches!(
                    from_gate,
                    Some(GatePhase::ReviewStep) | Some(GatePhase::ReviewImplementation)
                );
                if per_step {
                    *stage_index = (*stage_index).min(parsed.len().saturating_sub(1));
                } else {
                    *stage_index = 0;
                }
                let (count, title) = (
                    parsed.len(),
                    parsed
                        .get(*stage_index)
                        .map(|stage| stage.title.clone())
                        .unwrap_or_default(),
                );
                let (message, phase) = if per_step {
                    (
                        format!(
                            "📋 review Step {}: {title} — Enter to implement, type feedback to re-plan",
                            *stage_index + 1
                        ),
                        GatePhase::ReviewStep,
                    )
                } else {
                    (
                        format!(
                            "📋 plan ready ({count} stages) — Enter to review Step 1: {title}, type feedback to re-plan"
                        ),
                        GatePhase::ReviewPlan,
                    )
                };
                *gate = Some((message, phase));
                let context = agent.context.clone();
                let _ = event_tx.send(TuiEvent::TurnDone {
                    session: id,
                    context,
                });
                let _ = event_tx.send(TuiEvent::PlanUpdated {
                    session: id,
                    plan: Some((stages.clone().unwrap(), *stage_index)),
                });
                let _ = event_tx.send(TuiEvent::GatePending {
                    session: id,
                    message: gate.clone().unwrap().0,
                });
                return;
            }
            crate::agent::ChatOutcome::Terminated {
                tool: crate::tool::Tool::Escalate(findings),
            } => {
                *agent = new_agent()
                    .with_pinned_mode(Mode::Plan)
                    .with_cancel(cancel.clone())
                    .with_steering(steering.clone());
                let _ = event_tx.send(TuiEvent::StageChanged {
                    session: id,
                    mode: Some(Mode::Plan),
                });
                let message = if let Some(s) = stages.as_ref() {
                    format!(
                        "The implementation hit a blocker at Step {} of {}: {}\n\nRevise the remaining stages (from the current one on), keeping what is already done, and call submit_plan with all of them.",
                        *stage_index + 1,
                        s.len(),
                        findings
                    )
                } else {
                    format!(
                        "The implementation hit a blocker: {findings}\n\nRevise the plan, keeping what is already done, and call submit_plan."
                    )
                };
                let tx = event_tx.clone();
                match agent
                    .chat(&message, &mut |event| {
                        let _ = tx.send(TuiEvent::Agent { session: id, event });
                    })
                    .await
                {
                    Ok(crate::agent::ChatOutcome::Cancelled) => {
                        let context = agent.context.clone();
                        let _ = event_tx.send(TuiEvent::TurnDone {
                            session: id,
                            context,
                        });
                        return;
                    }
                    Ok(next) => {
                        outcome = next;
                        continue;
                    }
                    Err(error) => {
                        let _ = event_tx.send(TuiEvent::TurnError {
                            session: id,
                            message: error.to_string(),
                        });
                        return;
                    }
                }
            }
            _ => {
                let review = agent.stage_mode() == Some(Mode::Implement)
                    && stages
                        .as_ref()
                        .map(|s| *stage_index + 1 < s.len())
                        .unwrap_or(false);
                if review {
                    *stage_index += 1;
                    *gate = Some((
                        format!(
                            "📋 Step {} implemented — review the implementation — Enter to continue, type feedback to re-implement",
                            *stage_index
                        ),
                        GatePhase::ReviewImplementation,
                    ));
                    let context = agent.context.clone();
                    let _ = event_tx.send(TuiEvent::TurnDone {
                        session: id,
                        context,
                    });
                    let _ = event_tx.send(TuiEvent::PlanUpdated {
                        session: id,
                        plan: Some((stages.clone().unwrap(), *stage_index)),
                    });
                    let _ = event_tx.send(TuiEvent::GatePending {
                        session: id,
                        message: gate.clone().unwrap().0,
                    });
                    return;
                }
                if agent.stage_mode() == Some(Mode::Implement) {
                    *agent = new_agent().with_cancel(cancel.clone()).with_steering(steering.clone());
                    let _ = event_tx.send(TuiEvent::StageChanged {
                        session: id,
                        mode: None,
                    });
                    let _ = event_tx.send(TuiEvent::PlanUpdated {
                        session: id,
                        plan: None,
                    });
                    let _ = event_tx.send(TuiEvent::ReviewBaseline {
                        session: id,
                        baseline: None,
                    });
                }
                let context = agent.context.clone();
                let _ = event_tx.send(TuiEvent::TurnDone {
                    session: id,
                    context,
                });
                return;
            }
        }
    }
}

pub fn spawn_agent(
    id: u64,
    new_agent: std::sync::Arc<dyn Fn() -> Agent + Send + Sync>,
    restored: Option<Context>,
    event_tx: tokio::sync::mpsc::UnboundedSender<TuiEvent>,
    cwd: Option<std::path::PathBuf>,
) -> (
    tokio::sync::mpsc::UnboundedSender<String>,
    tokio::task::AbortHandle,
    tokio::sync::watch::Sender<u64>,
    crate::agent::Steering,
) {
    let (input_tx, mut input_rx) = tokio::sync::mpsc::unbounded_channel::<String>();
    let (cancel_tx, cancel_rx) = tokio::sync::watch::channel(0u64);
    let cancel_rx = std::sync::Arc::new(cancel_rx);
    let steering = crate::agent::Steering::new();
    let steering_task = steering.clone();
    let task = tokio::spawn(async move {
        let mut agent = new_agent()
            .with_cancel(cancel_rx.clone())
            .with_steering(steering_task.clone());
        if let Some(context) = restored {
            agent.context = context;
        }
        let mut gate: Option<(String, GatePhase)> = None;
        let mut stages: Option<Vec<crate::tool::PlanStage>> = None;
        let mut stage_index: usize = 0;
        let mut bg_rx = crate::bg::REGISTRY.subscribe();
        loop {
            tokio::select! {
                input = input_rx.recv() => {
                    let Some(mut input) = input else { break; };
                    let from_gate = gate.take().map(|(_, phase)| phase);
                    if let Some(phase) = from_gate {
                        match (phase, input.is_empty()) {
                            (GatePhase::ReviewImplementation, true) => {
                                let title = stages
                                    .as_ref()
                                    .map(|s| s[stage_index].title.clone())
                                    .unwrap_or_default();
                                let message = format!(
                                    "📋 review Step {}: {title} — Enter to implement, type feedback to re-plan",
                                    stage_index + 1
                                );
                                agent = new_agent().with_pinned_mode(Mode::Plan).with_cancel(cancel_rx.clone()).with_steering(steering_task.clone());
                                gate = Some((message.clone(), GatePhase::ReviewStep));
                                let _ = event_tx.send(TuiEvent::StageChanged { session: id, mode: Some(Mode::Plan) });
                                let _ = event_tx.send(TuiEvent::GatePending { session: id, message });
                                continue;
                            }
                            (GatePhase::ReviewImplementation, false) => {
                                stage_index = stage_index.saturating_sub(1);
                                input = stage_reimplement_message(
                                    &stages,
                                    stage_index,
                                    &input,
                                );
                            }
                            (GatePhase::ReviewPlan, true) => {
                                let title = stages
                                    .as_ref()
                                    .and_then(|s| s.first())
                                    .map(|s| s.title.clone())
                                    .unwrap_or_default();
                                let message = format!(
                                    "📋 review Step 1: {title} — Enter to implement, type feedback to re-plan"
                                );
                                agent = new_agent().with_pinned_mode(Mode::Plan).with_cancel(cancel_rx.clone()).with_steering(steering_task.clone());
                                gate = Some((message.clone(), GatePhase::ReviewStep));
                                let _ = event_tx.send(TuiEvent::StageChanged { session: id, mode: Some(Mode::Plan) });
                                let _ = event_tx.send(TuiEvent::GatePending { session: id, message });
                                continue;
                            }
                            (GatePhase::ReviewPlan, false) => {
                                input = stage_replan_message(&stages, &input);
                            }
                            (GatePhase::ReviewStep, true) => {
                                let baseline = crate::git::snapshot_tree(cwd.as_deref());
                                let _ = event_tx.send(TuiEvent::ReviewBaseline {
                                    session: id,
                                    baseline,
                                });
                                agent = new_agent().with_pinned_mode(Mode::Implement).with_cancel(cancel_rx.clone()).with_steering(steering_task.clone());
                                let _ = event_tx.send(TuiEvent::StageChanged { session: id, mode: Some(Mode::Implement) });
                                input = stage_implement_message(&stages, stage_index);
                            }
                            (GatePhase::ReviewStep, false) => {
                                input = stage_step_replan_message(&stages, stage_index, &input);
                            }
                        }
                    }
                    let tx = event_tx.clone();
                    let result = agent
                        .chat(
                            &input,
                            &mut |event| {
                                let _ = tx.send(TuiEvent::Agent { session: id, event });
                            },
                        )
                        .await
                        .map_err(|error| error.to_string());
                    match result {
                        Ok(crate::agent::ChatOutcome::Cancelled) => {
                            let context = agent.context.clone();
                            let _ = event_tx.send(TuiEvent::TurnDone {
                                session: id,
                                context,
                            });
                        }
                        Ok(outcome) => {
                            handle_outcome(
                                &mut agent,
                                &new_agent,
                                &event_tx,
                                id,
                                &mut gate,
                                &mut stages,
                                &mut stage_index,
                                from_gate,
                                outcome,
                                &cancel_rx,
                                &steering_task,
                            )
                            .await;
                        }
                        Err(message) => {
                            let _ = event_tx.send(TuiEvent::TurnError {
                                session: id,
                                message,
                            });
                        }
                    }
                }
                signal = bg_rx.recv() => {
                    if gate.is_none()
                        && let Ok(task_id) = signal
                        && agent.owns_and_unseen(&task_id)
                    {
                        let tx = event_tx.clone();
                        let result = agent
                            .bg_turn(&mut |event| {
                                let _ = tx.send(TuiEvent::Agent { session: id, event });
                            })
                            .await
                            .map_err(|error| error.to_string());
                        match result {
                            Ok(crate::agent::ChatOutcome::Cancelled) => {
                                let context = agent.context.clone();
                                let _ = tx.send(TuiEvent::TurnDone {
                                    session: id,
                                    context,
                                });
                            }
                            Ok(outcome) => {
                                handle_outcome(
                                    &mut agent,
                                    &new_agent,
                                    &event_tx,
                                    id,
                                    &mut gate,
                                    &mut stages,
                                    &mut stage_index,
                                    None,
                                    outcome,
                                    &cancel_rx,
                                    &steering_task,
                                )
                                .await;
                            }
                            Err(message) => {
                                let _ = tx.send(TuiEvent::TurnError {
                                    session: id,
                                    message,
                                });
                            }
                        }
                    }
                }
            }
        }
    });
    (input_tx, task.abort_handle(), cancel_tx, steering)
}

#[cfg(test)]
mod test {
    use super::*;
    use openai_oxide::client::OpenAI;
    use std::sync::{Arc, Mutex};
    use std::time::Duration;
    use tokio::io::AsyncReadExt;
    use tokio::io::AsyncWriteExt;

    fn git_repo() -> test_files::TestFiles {
        let dir = test_files::TestFiles::new();
        std::process::Command::new("git")
            .args(["init", "-q"])
            .current_dir(dir.path())
            .output()
            .unwrap();
        dir.file("a.txt", "one\n");
        dir
    }


    fn complete_request(data: &[u8]) -> Option<String> {
        let header_end = data.windows(4).position(|w| w == b"\r\n\r\n")?;
        let headers = std::str::from_utf8(&data[..header_end]).unwrap();
        let length = headers.lines().find_map(|line| {
            let (name, value) = line.split_once(':')?;
            if name.trim().eq_ignore_ascii_case("content-length") {
                value.trim().parse::<usize>().ok()
            } else {
                None
            }
        })?;
        let body_start = header_end + 4;
        if data.len() < body_start + length {
            return None;
        }
        Some(String::from_utf8_lossy(&data[body_start..body_start + length]).into_owned())
    }

    async fn wait_for_event(
        rx: &mut tokio::sync::mpsc::UnboundedReceiver<TuiEvent>,
        events: &mut Vec<TuiEvent>,
        predicate: impl Fn(&TuiEvent) -> bool,
    ) -> bool {
        let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(10);
        while tokio::time::Instant::now() < deadline {
            if let Ok(Some(event)) =
                tokio::time::timeout(std::time::Duration::from_secs(1), rx.recv()).await
            {
                if predicate(&event) {
                    events.push(event);
                    return true;
                }
                events.push(event);
            }
        }
        false
    }

    #[tokio::test]
    async fn plan_gate_approval_runs_the_implement_stage_then_resets() {
        let repo = git_repo();
        let (tx_req, mut rx_req) = tokio::sync::mpsc::unbounded_channel::<String>();
        let responses = Arc::new(Mutex::new(std::collections::VecDeque::from([
            "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"c1\",\"type\":\"function\",\"function\":{\"name\":\"submit_plan\",\"arguments\":\"{\\\"stages\\\":[{\\\"title\\\":\\\"step one\\\",\\\"description\\\":\\\"do it\\\"}]}\"}}]}}]}\n\ndata: [DONE]\n\n".to_string(),
            "data: {\"choices\":[{\"delta\":{\"content\":\"implemented\"}}]}\n\ndata: [DONE]\n\n".to_string(),
        ])));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            while let Ok((mut socket, _)) = listener.accept().await {
                let tx = tx_req.clone();
                let responses = responses.clone();
                tokio::spawn(async move {
                    let mut data = Vec::new();
                    let mut buf = [0u8; 8192];
                    loop {
                        match socket.read(&mut buf).await {
                            Ok(0) | Err(_) => break,
                            Ok(n) => {
                                data.extend_from_slice(&buf[..n]);
                                if let Some(body) = complete_request(&data) {
                                    let _ = tx.send(body);
                                    let sse = match responses.lock().unwrap().pop_front() {
                                        Some(response) => response,
                                        None => "data: {\"choices\":[{\"delta\":{\"content\":\"ok\"}}]}\n\ndata: [DONE]\n\n".to_string(),
                                    };
                                    let response = format!(
                                        "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\nconnection: close\r\n\r\n{sse}"
                                    );
                                    if socket.write_all(response.as_bytes()).await.is_err() {
                                        break;
                                    }
                                    break;
                                }
                            }
                        }
                    }
                });
            }
        });

        let client = OpenAI::with_config(
            openai_oxide::ClientConfig::new("local").base_url(format!("http://{addr}")),
        );
        let shared = Arc::new(Mutex::new(Mode::Yolo));
        let factory = Arc::new(move || {
            Agent::new(
                client.clone(),
                "test-model",
                shared.clone(),
                10000,
                Duration::from_secs(30),
            )
            .with_pinned_mode(Mode::Plan)
        });
        let (event_tx, mut event_rx) = tokio::sync::mpsc::unbounded_channel::<TuiEvent>();
        let (input_tx, handle, _cancel_tx, _steer) = spawn_agent(1, factory, None, event_tx, Some(repo.path().to_path_buf()));

        input_tx.send("plan me a feature".to_string()).unwrap();
        let mut events = vec![];
        assert!(
            wait_for_event(&mut event_rx, &mut events, |e| matches!(
                e,
                TuiEvent::GatePending { .. }
            ))
            .await
        );
        input_tx.send(String::new()).unwrap();
        assert!(
            wait_for_event(&mut event_rx, &mut events, |e| matches!(
                e,
                TuiEvent::GatePending { message, .. } if message.contains("review Step 1")
            ))
            .await
        );
        input_tx.send(String::new()).unwrap();
        assert!(
            wait_for_event(&mut event_rx, &mut events, |e| matches!(
                e,
                TuiEvent::StageChanged { mode: None, .. }
            ))
            .await
        );
        handle.abort();

        let mut requests = vec![];
        while let Ok(body) = rx_req.try_recv() {
            requests.push(body);
        }
        let implement_request = requests
            .into_iter()
            .find(|body| body.contains("Execute Step 1 of 1"))
            .expect("the implement stage request carries the plan");
        assert!(implement_request.contains("step one"));
        assert!(events.iter().any(|e| matches!(
            e,
            TuiEvent::StageChanged {
                mode: Some(Mode::Implement),
                ..
            }
        )));
        assert!(
            events
                .iter()
                .any(|e| matches!(e, TuiEvent::TurnDone { .. }))
        );
    }

    async fn staged_mock_server(responses: Vec<String>) -> (String, tokio::sync::mpsc::UnboundedReceiver<String>) {
        let (tx_req, rx_req) = tokio::sync::mpsc::unbounded_channel::<String>();
        let responses = Arc::new(Mutex::new(std::collections::VecDeque::from(responses)));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            while let Ok((mut socket, _)) = listener.accept().await {
                let tx = tx_req.clone();
                let responses = responses.clone();
                tokio::spawn(async move {
                    let mut data = Vec::new();
                    let mut buf = [0u8; 8192];
                    loop {
                        match socket.read(&mut buf).await {
                            Ok(0) | Err(_) => break,
                            Ok(n) => {
                                data.extend_from_slice(&buf[..n]);
                                if let Some(body) = complete_request(&data) {
                                    let _ = tx.send(body);
                                    let sse = match responses.lock().unwrap().pop_front() {
                                        Some(response) => response,
                                        None => "data: {\"choices\":[{\"delta\":{\"content\":\"ok\"}}]}\n\ndata: [DONE]\n\n".to_string(),
                                    };
                                    let response = format!(
                                        "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\nconnection: close\r\n\r\n{sse}"
                                    );
                                    if socket.write_all(response.as_bytes()).await.is_err() {
                                        break;
                                    }
                                    break;
                                }
                            }
                        }
                    }
                });
            }
        });
        (format!("http://{addr}"), rx_req)
    }

    fn plan_sse(id: &str, stages: &str) -> String {
        "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"__ID__\",\"type\":\"function\",\"function\":{\"name\":\"submit_plan\",\"arguments\":\"__STAGES__\"}}]}}]}\n\ndata: [DONE]\n\n"
            .replace("__ID__", id)
            .replace("__STAGES__", stages)
    }

    #[tokio::test]
    async fn staged_plan_walks_through_every_stage_with_a_review_gate() {
        let repo = git_repo();
        let stages = r#"{\"stages\":[{\"title\":\"data\",\"description\":\"entity\"},{\"title\":\"api\",\"description\":\"controller\"}]}"#;
        let (base_url, mut rx_req) = staged_mock_server(vec![
            plan_sse("c1", stages),
            "data: {\"choices\":[{\"delta\":{\"content\":\"stage one done\"}}]}\n\ndata: [DONE]\n\n".to_string(),
            "data: {\"choices\":[{\"delta\":{\"content\":\"stage two done\"}}]}\n\ndata: [DONE]\n\n".to_string(),
        ]).await;
        let client = OpenAI::with_config(
            openai_oxide::ClientConfig::new("local").base_url(base_url),
        );
        let shared = Arc::new(Mutex::new(Mode::Yolo));
        let factory = Arc::new(move || {
            Agent::new(
                client.clone(),
                "test-model",
                shared.clone(),
                10000,
                Duration::from_secs(30),
            )
            .with_pinned_mode(Mode::Plan)
        });
        let (event_tx, mut event_rx) = tokio::sync::mpsc::unbounded_channel::<TuiEvent>();
        let (input_tx, handle, _cancel_tx, _steer) = spawn_agent(1, factory, None, event_tx, Some(repo.path().to_path_buf()));
        let mut events = vec![];

        input_tx.send("plan me".to_string()).unwrap();
        assert!(
            wait_for_event(&mut event_rx, &mut events, |e| matches!(
                e,
                TuiEvent::GatePending { message, .. } if message.contains("plan ready (2 stages)")
                    && message.contains("Step 1: data")
            ))
            .await
        );

        input_tx.send(String::new()).unwrap();
        assert!(
            wait_for_event(&mut event_rx, &mut events, |e| matches!(
                e,
                TuiEvent::GatePending { message, .. } if message.contains("review Step 1: data")
            ))
            .await
        );

        input_tx.send(String::new()).unwrap();
        assert!(
            wait_for_event(&mut event_rx, &mut events, |e| matches!(
                e,
                TuiEvent::GatePending { message, .. } if message.contains("Step 1 implemented")
                    && message.contains("review the implementation")
            ))
            .await
        );
        assert!(
            !events
                .iter()
                .rev()
                .take(3)
                .any(|e| matches!(e, TuiEvent::StageChanged { mode: Some(Mode::Plan), .. }))
        );

        input_tx.send(String::new()).unwrap();
        assert!(
            wait_for_event(&mut event_rx, &mut events, |e| matches!(
                e,
                TuiEvent::GatePending { message, .. } if message.contains("review Step 2: api")
            ))
            .await
        );

        input_tx.send(String::new()).unwrap();
        assert!(
            wait_for_event(&mut event_rx, &mut events, |e| matches!(
                e,
                TuiEvent::StageChanged { mode: None, .. }
            ))
            .await
        );
        handle.abort();

        let mut requests = vec![];
        while let Ok(body) = rx_req.try_recv() {
            requests.push(body);
        }
        assert!(requests.iter().any(|body| body.contains("Execute Step 1 of 2: data")));
        assert!(requests.iter().any(|body| body.contains("Do not start later stages")));
        assert!(requests.iter().any(|body| body.contains("Execute Step 2 of 2: api")));
        assert!(requests.iter().any(|body| body.contains("Stages already done: Step 1: data")));
    }

    #[tokio::test]
    async fn review_gate_feedback_replans_only_the_rejected_step() {
        let repo = git_repo();
        let stages = r#"{\"stages\":[{\"title\":\"data\",\"description\":\"entity\"},{\"title\":\"api\",\"description\":\"controller\"}]}"#;
        let (base_url, mut rx_req) = staged_mock_server(vec![
            plan_sse("c1", stages),
            "data: {\"choices\":[{\"delta\":{\"content\":\"stage one done\"}}]}\n\ndata: [DONE]\n\n".to_string(),
        ]).await;
        let client = OpenAI::with_config(
            openai_oxide::ClientConfig::new("local").base_url(base_url),
        );
        let shared = Arc::new(Mutex::new(Mode::Yolo));
        let factory = Arc::new(move || {
            Agent::new(
                client.clone(),
                "test-model",
                shared.clone(),
                10000,
                Duration::from_secs(30),
            )
            .with_pinned_mode(Mode::Plan)
        });
        let (event_tx, mut event_rx) = tokio::sync::mpsc::unbounded_channel::<TuiEvent>();
        let (input_tx, handle, _cancel_tx, _steer) = spawn_agent(1, factory, None, event_tx, Some(repo.path().to_path_buf()));
        let mut events = vec![];

        input_tx.send("plan me".to_string()).unwrap();
        assert!(
            wait_for_event(&mut event_rx, &mut events, |e| matches!(
                e,
                TuiEvent::GatePending { .. }
            ))
            .await
        );
        input_tx.send(String::new()).unwrap();
        assert!(
            wait_for_event(&mut event_rx, &mut events, |e| matches!(
                e,
                TuiEvent::GatePending { message, .. } if message.contains("review Step 1: data")
            ))
            .await
        );
        input_tx.send(String::new()).unwrap();
        assert!(
            wait_for_event(&mut event_rx, &mut events, |e| matches!(
                e,
                TuiEvent::GatePending { message, .. } if message.contains("Step 1 implemented")
            ))
            .await
        );
        input_tx.send(String::new()).unwrap();
        assert!(
            wait_for_event(&mut event_rx, &mut events, |e| matches!(
                e,
                TuiEvent::GatePending { message, .. } if message.contains("review Step 2: api")
            ))
            .await
        );
        input_tx.send("api should be rest".to_string()).unwrap();
        tokio::time::sleep(std::time::Duration::from_millis(300)).await;
        handle.abort();

        let mut requests = vec![];
        while let Ok(body) = rx_req.try_recv() {
            requests.push(body);
        }
        let replan = requests
            .iter()
            .find(|body| body.contains("Re-plan only this step"))
            .expect("the review gate feedback re-plans only the rejected step");
        assert!(replan.contains("api should be rest"));
        assert!(replan.contains("Step 2 of 2"));
        assert!(replan.contains("Keep the other stages unchanged"));
    }

    #[tokio::test]
    async fn implementation_gate_feedback_reimplements_the_current_stage() {
        let repo = git_repo();
        let stages = r#"{\"stages\":[{\"title\":\"data\",\"description\":\"entity\"},{\"title\":\"api\",\"description\":\"controller\"}]}"#;
        let (base_url, mut rx_req) = staged_mock_server(vec![
            plan_sse("c1", stages),
            "data: {\"choices\":[{\"delta\":{\"content\":\"stage one done\"}}]}\n\ndata: [DONE]\n\n".to_string(),
            "data: {\"choices\":[{\"delta\":{\"content\":\"re-implemented\"}}]}\n\ndata: [DONE]\n\n".to_string(),
        ]).await;
        let client = OpenAI::with_config(
            openai_oxide::ClientConfig::new("local").base_url(base_url),
        );
        let shared = Arc::new(Mutex::new(Mode::Yolo));
        let factory = Arc::new(move || {
            Agent::new(
                client.clone(),
                "test-model",
                shared.clone(),
                10000,
                Duration::from_secs(30),
            )
            .with_pinned_mode(Mode::Plan)
        });
        let (event_tx, mut event_rx) = tokio::sync::mpsc::unbounded_channel::<TuiEvent>();
        let (input_tx, handle, _cancel_tx, _steer) = spawn_agent(1, factory, None, event_tx, Some(repo.path().to_path_buf()));
        let mut events = vec![];

        input_tx.send("plan me".to_string()).unwrap();
        assert!(
            wait_for_event(&mut event_rx, &mut events, |e| matches!(
                e,
                TuiEvent::GatePending { .. }
            ))
            .await
        );
        input_tx.send(String::new()).unwrap();
        assert!(
            wait_for_event(&mut event_rx, &mut events, |e| matches!(
                e,
                TuiEvent::GatePending { message, .. } if message.contains("review Step 1: data")
            ))
            .await
        );
        input_tx.send(String::new()).unwrap();
        assert!(
            wait_for_event(&mut event_rx, &mut events, |e| matches!(
                e,
                TuiEvent::GatePending { message, .. } if message.contains("review the implementation")
            ))
            .await
        );
        input_tx.send("the entity is wrong".to_string()).unwrap();
        tokio::time::sleep(std::time::Duration::from_millis(300)).await;
        handle.abort();

        let mut requests = vec![];
        while let Ok(body) = rx_req.try_recv() {
            requests.push(body);
        }
        let reimplement = requests
            .iter()
            .find(|body| body.contains("Re-implement Step 1 of 2"))
            .expect("the implementation gate feedback re-implements the current stage");
        assert!(reimplement.contains("the entity is wrong"));
        assert!(reimplement.contains("The previous implementation was rejected"));
    }
}
