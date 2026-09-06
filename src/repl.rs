use std::io::{IsTerminal, Write};
use tokio::io::AsyncBufReadExt;

use crate::agent::Agent;
use crate::render::Renderer;

pub async fn run_plain(mut agent: Agent) -> Result<(), Box<dyn std::error::Error>> {

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
