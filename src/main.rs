mod agent;
mod context;
mod message;
mod tool;

use std::io::{IsTerminal, Write};

use agent::{Agent, AgentEvent, AnswerGate, ThinkingMode};
use openai_oxide::client::OpenAI;
use tokio::io::AsyncBufReadExt;

const SYSTEM_PROMPT: &str = "You are a coding agent. Use the tools to accomplish tasks.";

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let api_url = std::env::var("API_URL")
        .unwrap_or_else(|_| "http://localhost:11434/v1".to_string());
    let model = std::env::var("KITE_MODEL").unwrap_or_else(|_| "llama3.2".to_string());
    let max_tokens = std::env::var("KITE_CONTEXT_WINDOW")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(24000);

    let client = OpenAI::with_config(openai_oxide::ClientConfig::new("local").base_url(api_url));
    let mut agent = Agent::new(client, model, SYSTEM_PROMPT, max_tokens);
    if let Ok(value) = std::env::var("KITE_THINKING") {
        let enabled = value != "0";
        agent = agent.with_extra_body(serde_json::json!({
            "chat_template_kwargs": { "enable_thinking": enabled }
        }));
    }

    let tty = std::io::stdout().is_terminal();
    let mut stdin = tokio::io::BufReader::new(tokio::io::stdin());
    let mut line = String::new();

    loop {
        print!("> ");
        std::io::stdout().flush()?;
        if stdin.read_line(&mut line).await? == 0 {
            break;
        }
        let input = line.trim().to_string();
        line.clear();
        if input.is_empty() {
            continue;
        }

        let mut gate = AnswerGate::new();
        let mut thinking_open = false;
        match agent
            .chat(
                &input,
                &mut |event: AgentEvent| match event {
                    AgentEvent::Tokens(chunk) => {
                        let (mode, text) = gate.on_chunk(&chunk);

                        if mode == ThinkingMode::Live
                            && let Some(t) = &chunk.thinking
                        {
                            if !thinking_open {
                                eprint!("{}", if tty { "\x1b[2m" } else { "[thinking] " });
                                thinking_open = true;
                            }
                            eprint!("{t}");
                            let _ = std::io::stderr().flush();
                        }

                        if let Some(text) = text {
                            if thinking_open {
                                eprint!("{}", if tty { "\x1b[0m\n" } else { "\n" });
                                thinking_open = false;
                            }
                            print!("{text}");
                            let _ = std::io::stdout().flush();
                        }
                    }
                    AgentEvent::ToolStarted { header, body } => {
                        if thinking_open {
                            eprint!("{}", if tty { "\x1b[0m\n" } else { "\n" });
                            thinking_open = false;
                        }
                        print!(
                            "{}",
                            if tty {
                                format!("\x1b[2m⚙ {header}\x1b[0m\n")
                            } else {
                                format!("[tool] {header}\n")
                            }
                        );
                        if let Some(body) = body {
                            print!("{body}\n");
                        }
                        let _ = std::io::stdout().flush();
                    }
                    AgentEvent::ToolResult(body) => {
                        print!("{body}\n");
                        let _ = std::io::stdout().flush();
                    }
                },
            )
            .await
        {
            Ok(_) => {
                if let Some(remaining) = gate.finish() {
                    if thinking_open {
                        eprint!("{}", if tty { "\x1b[0m\n" } else { "\n" });
                        thinking_open = false;
                    }
                    print!("{remaining}");
                }
                if thinking_open {
                    eprint!("{}", if tty { "\x1b[0m\n" } else { "\n" });
                }
                println!();
            }
            Err(error) => eprintln!("error: {error}"),
        }
    }

    println!(
        "session over: {} prompt / {} completion tokens",
        agent.context.total_prompt_tokens,
        agent.context.total_completion_tokens
    );

    Ok(())
}
