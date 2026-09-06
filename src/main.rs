mod agent;
mod bg;
mod context;
mod message;
mod paths;
mod session;
mod session_store;
mod stream;
mod transcript;
mod tool;
mod tui;


use agent::Agent;
use clap::Parser;
use openai_oxide::client::OpenAI;

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
}

#[derive(Copy, Clone, PartialEq, Eq, clap::ValueEnum)]
enum Thinking {
    Auto,
    On,
    Off,
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
}
