mod agent;
mod bg;
mod context;
mod message;
mod render;
mod tool;
mod tui;

use std::io::{IsTerminal, Write};

use agent::Agent;
use clap::{Parser, ValueEnum};
use openai_oxide::client::OpenAI;
use render::Renderer;
use tokio::io::AsyncBufReadExt;

const SYSTEM_PROMPT: &str = "You are a coding agent. Use the tools to accomplish tasks. For long-running commands (tests, builds, dev servers), use bg_run instead of bash; its result is reported automatically when the task finishes. Your configuration and session history live in .kite/: previous sessions are stored as JSON transcripts in .kite/sessions/ and background task logs in .kite/tasks/ — read them when the user refers to previous work. Before quoting or summarizing any file's content, re-read it. Never answer from remembered file content — files may have changed since you last saw them.";

#[derive(Parser)]
struct Cli {
    #[arg(long, default_value = "http://localhost:11434/v1")]
    api_url: String,

    #[arg(long, default_value = "llama3.2")]
    model: String,

    #[arg(long, default_value_t = 24000)]
    context_window: u64,

    #[arg(long, value_enum, default_value = "auto")]
    thinking: Thinking,

    #[arg(long, default_value_t = 300)]
    bash_timeout: u64,

    #[arg(long, value_enum)]
    ui: Option<Ui>,
}

#[derive(Copy, Clone, PartialEq, Eq, ValueEnum)]
enum Thinking {
    Auto,
    On,
    Off,
}

#[derive(Copy, Clone, PartialEq, Eq, ValueEnum)]
enum Ui {
    Plain,
    Tui,
}

fn build_agent(
    client: &OpenAI,
    model: &str,
    thinking: Thinking,
    context_window: u64,
    bash_timeout: u64,
) -> Agent {
    let mut agent = Agent::new(
        client.clone(),
        model,
        SYSTEM_PROMPT,
        context_window,
        std::time::Duration::from_secs(bash_timeout),
    );
    if thinking != Thinking::Auto {
        let enabled = matches!(thinking, Thinking::On);
        agent = agent.with_extra_body(serde_json::json!({
            "chat_template_kwargs": { "enable_thinking": enabled }
        }));
    }
    agent
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    let client = OpenAI::with_config(openai_oxide::ClientConfig::new("local").base_url(cli.api_url));
    let model = cli.model.clone();

    let use_tui = match cli.ui {
        Some(Ui::Tui) => true,
        Some(Ui::Plain) => false,
        None => std::io::stdin().is_terminal(),
    };

    if use_tui {
        let client = client.clone();
        tui::run(
            {
                let model = model.clone();
                move || {
                    build_agent(
                        &client,
                        &model,
                        cli.thinking,
                        cli.context_window,
                        cli.bash_timeout,
                    )
                }
            },
            model,
        )
        .await
    } else {
        run_plain(build_agent(
            &client,
            &model,
            cli.thinking,
            cli.context_window,
            cli.bash_timeout,
        ))
        .await
    }
}

async fn run_plain(mut agent: Agent) -> Result<(), Box<dyn std::error::Error>> {
    let out_tty = std::io::stdout().is_terminal();
    let err_tty = std::io::stderr().is_terminal();
    let mut stdin = tokio::io::BufReader::new(tokio::io::stdin());
    let mut line = String::new();
    let mut bg_rx = crate::bg::REGISTRY.subscribe();

    loop {
        print!("> ");
        std::io::stdout().flush()?;
        tokio::select! {
            result = stdin.read_line(&mut line) => {
                if result? == 0 {
                    break;
                }
                let input = line.trim().to_string();
                line.clear();
                if input.is_empty() {
                    continue;
                }
                let mut renderer = Renderer::new(out_tty, err_tty, std::io::stdout(), std::io::stderr());
                match agent.chat(&input, &mut |event| renderer.on_event(event)).await {
                    Ok(_) => renderer.finish(),
                    Err(error) => {
                        renderer.finish();
                        eprintln!("error: {error}");
                    }
                }
            }
            signal = bg_rx.recv() => {
                if let Ok(id) = signal
                    && agent.owns_and_unseen(&id)
                {
                    let mut renderer = Renderer::new(out_tty, err_tty, std::io::stdout(), std::io::stderr());
                    match agent.bg_turn(&mut |event| renderer.on_event(event)).await {
                        Ok(_) => renderer.finish(),
                        Err(error) => {
                            renderer.finish();
                            eprintln!("error: {error}");
                        }
                    }
                }
            }
        }
    }

    println!(
        "session over: {} prompt / {} completion tokens",
        agent.context.total_prompt_tokens,
        agent.context.total_completion_tokens
    );

    Ok(())
}
