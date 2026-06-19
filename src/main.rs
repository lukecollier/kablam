mod agent;
mod runtime;

use std::collections::VecDeque;
use std::env;
use std::fs::{self, OpenOptions};
use std::io as std_io;
use std::time::Duration;

use agent::{ToolParameter, ToolParameterKind, ToolSpec};
use mistralrs::{RequestBuilder, RequestLike, TextMessageRole, TextMessages};
use runtime::{RuntimeBuilder, RuntimeResponseEvent};
use tokio::io::{self, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::sync::mpsc;
use tracing_subscriber::EnvFilter;
use tracing_subscriber::filter::LevelFilter;

const SYSTEM_PROMPT: &str = "You are a concise assistant. If a user asks for information that an available tool can provide, call the tool instead of answering directly.";
const LOG_FILE: &str = "logs/kablam.log";
const MAX_GENERATION_TOKENS: usize = 512;

#[derive(Debug, Clone)]
enum ChatEntry {
    User(String),
    Assistant(String),
}

#[tokio::main(flavor = "multi_thread")]
async fn main() {
    if let Err(err) = init_logging() {
        eprintln!("failed to initialize file logging at {LOG_FILE}: {err}");
    }

    let startup_model = match parse_startup_model() {
        Ok(model) => model,
        Err(err) => {
            eprintln!("{err}");
            print_usage();
            return;
        }
    };

    tracing::info!(
        backend = runtime::runtime_backend(),
        "runtime backend selected"
    );

    let runtime = RuntimeBuilder::new().build();
    let tools = default_tools();
    let mut active_model = startup_model;
    let mut active_connection = match runtime.open_connection(&active_model).await {
        Ok(connection) => connection,
        Err(err) => {
            eprintln!("failed to open initial model connection: {err}");
            return;
        }
    };
    let mut history: VecDeque<ChatEntry> = VecDeque::new();

    let mut stdout = io::stdout();

    println!("registered runtimes:");
    for config in runtime.list_configs() {
        println!("- {}", config.id);
    }
    println!("active model: {active_model}");
    println!("type /model <id> to switch, /quit to exit");

    let mut stdin = BufReader::new(io::stdin());

    loop {
        if let Err(err) = stdout.write_all(b"> ").await {
            eprintln!("failed to write prompt: {err}");
            break;
        }
        if let Err(err) = stdout.flush().await {
            eprintln!("failed to flush stdout: {err}");
            break;
        }

        let ctrl_c = tokio::signal::ctrl_c();
        tokio::pin!(ctrl_c);
        let line = tokio::select! {
            _ = &mut ctrl_c => {
                clear_status_line().await;
                eprintln!("interrupted");
                runtime.request_shutdown();
                std::process::exit(130);
            }
            line = next_line(&mut stdin) => {
                match line {
                    Some(line) => line,
                    None => break,
                }
            }
        };
        let input = line.trim();

        if input.is_empty() {
            continue;
        }

        if input == "/quit" || input == "/exit" {
            runtime.request_shutdown();
            return;
        }

        if let Some(model_id) = input.strip_prefix("/model ") {
            let model_id = model_id.trim();
            if model_id.is_empty() {
                println!("usage: /model <model-id>");
                continue;
            }

            match runtime.open_connection(model_id).await {
                Ok(connection) => {
                    if let Err(err) = active_connection.end_connection().await {
                        eprintln!("failed to close previous model connection: {err}");
                    }
                    active_connection = connection;
                    active_model.clear();
                    active_model.push_str(model_id);
                    println!("active model: {active_model}");
                }
                Err(err) => {
                    println!("{err}");
                    println!("available models:");
                    for config in runtime.list_configs() {
                        println!("- {}", config.id);
                    }
                }
            }
            continue;
        }

        if input == "/models" {
            println!("available models:");
            for config in runtime.list_configs() {
                println!("- {}", config.id);
            }
            continue;
        }

        history.push_back(ChatEntry::User(input.to_string()));

        let Some(config) = runtime.config(&active_model) else {
            eprintln!("active model disappeared from registry: {active_model}");
            break;
        };

        let request = build_request(
            &history,
            config
                .model
                .tool_format()
                .system_prompt(SYSTEM_PROMPT, &tools),
        );
        let tool_format = config.model.tool_format();
        let (status_tx, mut status_rx) = mpsc::unbounded_channel();
        let mut response_stream = match active_connection
            .stream_with_status(request, Some(status_tx))
            .await
        {
            Ok(stream) => stream,
            Err(err) => {
                eprintln!("runtime error: {err}");
                continue;
            }
        };

        let mut response = String::new();
        let mut completed = false;
        let mut stdout = io::stdout();
        let mut status_label = format!("sending to {active_model}");
        let frames = ["|", "/", "-", "\\"];
        let mut frame_index = 0usize;
        let mut tick = tokio::time::interval(Duration::from_millis(100));
        let mut spinner_enabled = true;
        render_status_line(frames[frame_index], &status_label).await;

        loop {
            let ctrl_c = tokio::signal::ctrl_c();
            tokio::pin!(ctrl_c);

            tokio::select! {
                _ = &mut ctrl_c => {
                    clear_status_line().await;
                    eprintln!("interrupted");
                    runtime.request_shutdown();
                    std::process::exit(130);
                }
                status = status_rx.recv() => {
                    if spinner_enabled {
                        if let Some(status) = status {
                            status_label = status.message();
                            render_status_line(frames[frame_index % frames.len()], &status_label).await;
                        }
                    } else if status.is_some() {
                        // Drain status updates so the channel cannot back up after output starts.
                    }
                }
                _ = tick.tick() => {
                    if spinner_enabled {
                        frame_index = frame_index.wrapping_add(1);
                        render_status_line(frames[frame_index % frames.len()], &status_label).await;
                    }
                }
                event = response_stream.next() => {
                    match event {
                        Some(RuntimeResponseEvent::Chunk { content, .. }) => {
                            if spinner_enabled {
                                spinner_enabled = false;
                                clear_status_line().await;
                            }
                            response.push_str(&content);
                            if let Err(err) = stdout.write_all(content.as_bytes()).await {
                                eprintln!("failed to write response chunk: {err}");
                                break;
                            }
                            if let Err(err) = stdout.flush().await {
                                eprintln!("failed to flush response chunk: {err}");
                                break;
                            }
                        }
                        Some(RuntimeResponseEvent::Complete { .. }) => {
                            clear_status_line().await;
                            if !response.ends_with('\n') {
                                let _ = stdout.write_all(b"\n").await;
                                let _ = stdout.flush().await;
                            }
                            completed = true;
                            break;
                        }
                        Some(RuntimeResponseEvent::Error { error, .. }) => {
                            clear_status_line().await;
                            eprintln!("runtime error: {error}");
                            break;
                        }
                        None => {
                            clear_status_line().await;
                            break;
                        }
                    }
                }
            }
        }

        if completed {
            let tool_calls = tool_format.parse(&response);
            if !tool_calls.is_empty() {
                print_tool_calls(&tool_calls);
            }
            history.push_back(ChatEntry::Assistant(response));
        }
    }

    if let Err(err) = active_connection.end_connection().await {
        eprintln!("failed to close model connection: {err}");
    }
}

fn init_logging() -> std_io::Result<()> {
    fs::create_dir_all("logs")?;
    let log_path = String::from(LOG_FILE);
    let writer = move || {
        OpenOptions::new()
            .create(true)
            .append(true)
            .open(&log_path)
            .expect("failed to open log file")
    };
    let filter = EnvFilter::builder()
        .with_default_directive(LevelFilter::INFO.into())
        .from_env_lossy()
        .add_directive("mistralrs_core=debug".parse().unwrap());

    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(false)
        .with_ansi(false)
        .with_writer(writer)
        .try_init()
        .map_err(|err| std_io::Error::new(std_io::ErrorKind::Other, err.to_string()))
}

async fn next_line(stdin: &mut BufReader<io::Stdin>) -> Option<String> {
    let mut line = Vec::new();

    loop {
        let byte = match stdin.read_u8().await {
            Ok(byte) => byte,
            Err(err) => {
                eprintln!("failed to read input: {err}");
                return None;
            }
        };

        match byte {
            b'\n' => break,
            b'\r' => {
                // Some macOS and embedded terminals submit Enter as CR.
                if matches!(stdin.buffer().first(), Some(b'\n')) {
                    let _ = stdin.read_u8().await;
                }
                break;
            }
            byte => line.push(byte),
        }
    }

    Some(String::from_utf8_lossy(&line).into_owned())
}

async fn render_status_line(frame: &str, label: &str) {
    let mut stderr = io::stderr();
    let _ = stderr
        .write_all(format!("\r\x1b[2K{frame} {label}").as_bytes())
        .await;
    let _ = stderr.flush().await;
}

async fn clear_status_line() {
    let mut stderr = io::stderr();
    let _ = stderr.write_all(b"\r\x1b[2K").await;
    let _ = stderr.flush().await;
}

fn parse_startup_model() -> Result<String, String> {
    let mut args = env::args().skip(1);
    let mut model = String::from("smollm2-360m");

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--model" | "-m" => {
                let Some(value) = args.next() else {
                    return Err(String::from("missing value for --model"));
                };
                model = value;
            }
            _ if arg.starts_with("--model=") => {
                model = arg["--model=".len()..].to_string();
            }
            "--help" | "-h" => {
                print_usage();
                std::process::exit(0);
            }
            _ => {
                return Err(format!("unrecognized argument: {arg}"));
            }
        }
    }

    if model.is_empty() {
        return Err(String::from("startup model cannot be empty"));
    }

    Ok(model)
}

fn print_usage() {
    println!("usage: kablam [--model <model-id>]");
    println!("example: kablam --model qwen3.5-q2k");
    println!("run /models in the app to list all registered variants");
}

fn default_tools() -> Vec<ToolSpec> {
    vec![
        ToolSpec::new(
            "search_docs",
            "Search local project documentation for relevant passages.",
            vec![
                ToolParameter::required("query", "The search query.", ToolParameterKind::String),
                ToolParameter::optional(
                    "limit",
                    "Maximum number of results to return.",
                    ToolParameterKind::Integer,
                ),
            ],
        ),
        ToolSpec::new(
            "read_file",
            "Read a local project file by relative path.",
            vec![ToolParameter::required(
                "path",
                "The project-relative file path to read.",
                ToolParameterKind::String,
            )],
        ),
        ToolSpec::new(
            "get_weather",
            "Get fake weather for a city. This is a test tool and does not call a real API.",
            vec![
                ToolParameter::required(
                    "city",
                    "The city to get fake weather for.",
                    ToolParameterKind::String,
                ),
                ToolParameter::optional(
                    "include_forecast",
                    "Whether to include a fake multi-day forecast.",
                    ToolParameterKind::Boolean,
                ),
            ],
        ),
    ]
}

fn print_tool_calls(tool_calls: &[agent::ToolCall]) {
    println!("parsed tool calls:");
    for (index, call) in tool_calls.iter().enumerate() {
        let number = index + 1;
        let arguments = serde_json::to_string_pretty(&call.arguments)
            .unwrap_or_else(|_| call.arguments.to_string());

        println!("{number}. {}", call.name);
        println!("{arguments}");
    }
}

fn build_request(history: &VecDeque<ChatEntry>, system_prompt: String) -> RequestBuilder {
    let mut messages = TextMessages::new().add_message(TextMessageRole::System, system_prompt);

    for entry in history {
        match entry {
            ChatEntry::User(content) => {
                messages = messages.add_message(TextMessageRole::User, content);
            }
            ChatEntry::Assistant(content) => {
                messages = messages.add_message(TextMessageRole::Assistant, content);
            }
        }
    }

    tracing::info!(
        max_generation_tokens = MAX_GENERATION_TOKENS,
        message_count = messages.messages_ref().len(),
        "building chat generation request"
    );

    RequestBuilder::from(messages).set_sampler_max_len(MAX_GENERATION_TOKENS)
}
