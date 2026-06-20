mod agent;
mod command_palette;
mod runtime;
mod terminal;

use std::collections::HashSet;
use std::env;
use std::fs::{self, OpenOptions};
use std::io::{self as std_io, ErrorKind};
use std::thread;
use std::time::{Duration, Instant};

use agent::{ToolParameter, ToolParameterKind, ToolSpec};
use arboard::Clipboard;
use command_palette::{
    CommandCommit, CommandDispatch, CommandPaletteState, CommandPaletteView, CommandParseError,
};
use crossterm::event::{self, Event, KeyCode, MouseEventKind};
use mistralrs::{RequestBuilder, RequestLike, TextMessageRole, TextMessages};
use runtime::{RuntimeBuilder, RuntimeConnection, RuntimeResponseEvent, RuntimeStatus};
use terminal::{
    AssistantStatus, Mode, RenderState, ScrollAnchor, SystemStatus, TerminalUi, TranscriptEntry,
    chat_entry_positions, clipboard_text, selected_transcript_index,
};
use tokio::sync::mpsc;
use tracing_subscriber::EnvFilter;
use tracing_subscriber::filter::LevelFilter;

const SYSTEM_PROMPT: &str = "You are a concise assistant. If a user asks for information that an available tool can provide, call the tool instead of answering directly.";
const LOG_FILE: &str = "logs/kablam.log";
const MAX_GENERATION_TOKENS: usize = 512;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ComposeTarget {
    InsertAfterSelection,
    AppendBottom,
}

#[tokio::main(flavor = "multi_thread")]
async fn main() {
    if let Err(err) = init_logging() {
        eprintln!("failed to initialize file logging at {LOG_FILE}: {err}");
    }

    let startup = match parse_startup_options() {
        Ok(startup) => startup,
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
    let mut tools_enabled = startup.tools_enabled;
    let mut active_model = startup.model;
    let mut active_connection = match runtime.open_connection(&active_model).await {
        Ok(connection) => connection,
        Err(err) => {
            eprintln!("failed to open initial model connection: {err}");
            return;
        }
    };

    let mut transcript: Vec<TranscriptEntry> = Vec::new();
    push_system_message(&mut transcript, format!("active model: {active_model}"));
    let mut command_palette = CommandPaletteState::new(
        runtime
            .list_configs()
            .iter()
            .map(|config| config.id.clone())
            .collect(),
    );

    let mut ui = match TerminalUi::new() {
        Ok(ui) => ui,
        Err(err) => {
            eprintln!("failed to initialize terminal UI: {err}");
            return;
        }
    };

    let (event_tx, mut event_rx) = mpsc::unbounded_channel();
    let _event_thread = start_event_reader(event_tx);
    let mut mode = Mode::Insert;
    let mut selected_chat_entry: Option<usize> = None;
    let mut chat_draft = String::new();
    let mut compose_target = ComposeTarget::AppendBottom;
    let mut scroll_anchor = ScrollAnchor::Bottom;

    if let Err(err) = redraw(
        &mut ui,
        &transcript,
        selected_chat_entry,
        mode,
        &chat_draft,
        compose_target,
        scroll_anchor,
        matches!(mode, Mode::Command).then(|| command_palette.view()),
    ) {
        eprintln!("failed to render UI: {err}");
        return;
    }

    let ctrl_c = tokio::signal::ctrl_c();
    tokio::pin!(ctrl_c);

    loop {
        tokio::select! {
            _ = &mut ctrl_c => {
                runtime.request_shutdown();
                break;
            }
            event = event_rx.recv() => {
                let Some(event) = event else {
                    break;
                };

        let outcome = match event {
                    Event::Key(key) => {
                        handle_key_event(
                            key,
                            &mut mode,
                            &mut selected_chat_entry,
                            &mut chat_draft,
                            &mut compose_target,
                            &mut scroll_anchor,
                            &mut command_palette,
                            &mut transcript,
                            &runtime,
                            &mut active_connection,
                            &mut active_model,
                            &mut tools_enabled,
                            &tools,
                            &mut ui,
                        )
                        .await
                    }
                    Event::Mouse(mouse) => {
                        handle_mouse_event(
                            mouse.kind,
                            &mut mode,
                            &mut selected_chat_entry,
                            &mut chat_draft,
                            &mut compose_target,
                            &mut scroll_anchor,
                            &transcript,
                        );
                        Ok(AppAction::Continue)
                    }
                    Event::Resize(_, _) => Ok(AppAction::Continue),
                    _ => Ok(AppAction::Continue),
                };

                match outcome {
                    Ok(AppAction::Continue) => {
                        if matches!(mode, Mode::Normal) {
                            normalize_selection(&transcript, &mut selected_chat_entry);
                        }
                        if let Err(err) = redraw(
                            &mut ui,
                            &transcript,
                            selected_chat_entry,
                            mode,
                            &chat_draft,
                            compose_target,
                            scroll_anchor,
                            matches!(mode, Mode::Command).then(|| command_palette.view()),
                        ) {
                            eprintln!("failed to redraw UI: {err}");
                            runtime.request_shutdown();
                            break;
                        }
                    }
                    Ok(AppAction::Quit) => {
                        runtime.request_shutdown();
                        break;
                    }
                    Err(err) => {
                        if err.kind() == ErrorKind::Interrupted {
                            break;
                        }
                        eprintln!("input handling failed: {err}");
                        runtime.request_shutdown();
                        break;
                    }
                }
            }
        }
    }

    if let Err(err) = active_connection.end_connection().await {
        eprintln!("failed to close model connection: {err}");
    }
}

async fn handle_key_event(
    key: crossterm::event::KeyEvent,
    mode: &mut Mode,
    selected_chat_entry: &mut Option<usize>,
    chat_draft: &mut String,
    compose_target: &mut ComposeTarget,
    scroll_anchor: &mut ScrollAnchor,
    command_palette: &mut CommandPaletteState,
    transcript: &mut Vec<TranscriptEntry>,
    runtime: &runtime::Runtime,
    active_connection: &mut RuntimeConnection,
    active_model: &mut String,
    tools_enabled: &mut bool,
    tools: &[ToolSpec],
    ui: &mut TerminalUi,
) -> std_io::Result<AppAction> {
    command_palette.set_model_ids(
        runtime
            .list_configs()
            .iter()
            .map(|config| config.id.clone()),
    );

    match *mode {
        Mode::Insert => {
            handle_chat_key_event(
                key,
                mode,
                selected_chat_entry,
                chat_draft,
                compose_target,
                scroll_anchor,
                transcript,
                runtime,
                active_connection,
                active_model,
                tools_enabled,
                tools,
                ui,
            )
            .await
        }
        Mode::Normal => handle_history_key_event(
            key,
            mode,
            selected_chat_entry,
            chat_draft,
            compose_target,
            scroll_anchor,
            command_palette,
            transcript,
        ),
        Mode::Command => {
            handle_command_key_event(
                key,
                mode,
                command_palette,
                selected_chat_entry,
                compose_target,
                chat_draft,
                transcript,
                runtime,
                active_connection,
                active_model,
                tools_enabled,
            )
            .await
        }
    }
}

async fn handle_chat_key_event(
    key: crossterm::event::KeyEvent,
    mode: &mut Mode,
    selected_chat_entry: &mut Option<usize>,
    chat_draft: &mut String,
    compose_target: &mut ComposeTarget,
    scroll_anchor: &mut ScrollAnchor,
    transcript: &mut Vec<TranscriptEntry>,
    runtime: &runtime::Runtime,
    active_connection: &mut RuntimeConnection,
    active_model: &mut String,
    tools_enabled: &mut bool,
    tools: &[ToolSpec],
    ui: &mut TerminalUi,
) -> std_io::Result<AppAction> {
    match key.code {
        KeyCode::Esc => {
            exit_insert_mode(mode, selected_chat_entry, *compose_target, transcript);
        }
        KeyCode::Enter => {
            let input = chat_draft.trim().to_string();
            chat_draft.clear();
            if input.is_empty() {
                return Ok(AppAction::Continue);
            }

            submit_chat_turn(
                input,
                mode,
                selected_chat_entry,
                compose_target,
                scroll_anchor,
                transcript,
                runtime,
                active_connection,
                active_model,
                *tools_enabled,
                tools,
                ui,
            )
            .await?;
        }
        KeyCode::Backspace => {
            chat_draft.pop();
        }
        KeyCode::Char(ch) if key.modifiers.is_empty() => {
            chat_draft.push(ch);
        }
        _ => {}
    }

    Ok(AppAction::Continue)
}

fn handle_history_key_event(
    key: crossterm::event::KeyEvent,
    mode: &mut Mode,
    selected_chat_entry: &mut Option<usize>,
    chat_draft: &mut String,
    compose_target: &mut ComposeTarget,
    scroll_anchor: &mut ScrollAnchor,
    command_palette: &mut CommandPaletteState,
    transcript: &mut Vec<TranscriptEntry>,
) -> std_io::Result<AppAction> {
    match key.code {
        KeyCode::Esc => {}
        KeyCode::Enter => {
            open_footer_compose(mode, selected_chat_entry, compose_target, chat_draft);
        }
        KeyCode::Char(':') => {
            *mode = Mode::Command;
            command_palette.open();
        }
        KeyCode::Char('i') => {
            open_inline_compose_for_selection(
                transcript,
                selected_chat_entry,
                compose_target,
                mode,
                chat_draft,
            );
        }
        KeyCode::Char('o') => {
            open_compose_after_selection(
                transcript,
                selected_chat_entry,
                compose_target,
                mode,
                chat_draft,
            );
        }
        KeyCode::Char('j') | KeyCode::Down => {
            *scroll_anchor = ScrollAnchor::Bottom;
            *selected_chat_entry = move_selection(transcript, *selected_chat_entry, 1);
        }
        KeyCode::Char('k') | KeyCode::Up => {
            *scroll_anchor = ScrollAnchor::Top;
            *selected_chat_entry = move_selection(transcript, *selected_chat_entry, -1);
        }
        KeyCode::Char('y') => {
            if let Some(index) = selected_transcript_index(transcript, *selected_chat_entry) {
                if let Some(entry) = transcript.get(index) {
                    if let Err(err) = yank_entry(entry) {
                        push_system_message(transcript, format!("clipboard error: {err}"));
                    }
                }
            }
        }
        _ => {}
    }

    Ok(AppAction::Continue)
}

async fn handle_command_key_event(
    key: crossterm::event::KeyEvent,
    mode: &mut Mode,
    command_palette: &mut CommandPaletteState,
    selected_chat_entry: &mut Option<usize>,
    compose_target: &mut ComposeTarget,
    chat_draft: &mut String,
    transcript: &mut Vec<TranscriptEntry>,
    runtime: &runtime::Runtime,
    active_connection: &mut RuntimeConnection,
    active_model: &mut String,
    tools_enabled: &mut bool,
) -> std_io::Result<AppAction> {
    match key.code {
        KeyCode::Esc => {
            command_palette.close();
            *mode = Mode::Normal;
        }
        KeyCode::Tab => {
            let _returned_to_editor = command_palette.cycle_selection();
        }
        KeyCode::Enter => match command_palette.commit() {
            Ok(CommandCommit::StayOpen) => {}
            Ok(CommandCommit::Execute(dispatch)) => {
                return execute_command_dispatch(
                    dispatch,
                    mode,
                    command_palette,
                    selected_chat_entry,
                    compose_target,
                    chat_draft,
                    transcript,
                    runtime,
                    active_connection,
                    active_model,
                    tools_enabled,
                )
                .await;
            }
            Err(err) => {
                push_command_error(transcript, err);
            }
        },
        KeyCode::Backspace => {
            command_palette.backspace();
        }
        KeyCode::Char(ch) if key.modifiers.is_empty() => {
            command_palette.input_char(ch);
        }
        _ => {}
    }

    Ok(AppAction::Continue)
}

async fn execute_command_dispatch(
    dispatch: CommandDispatch,
    mode: &mut Mode,
    command_palette: &mut CommandPaletteState,
    selected_chat_entry: &mut Option<usize>,
    compose_target: &mut ComposeTarget,
    chat_draft: &mut String,
    transcript: &mut Vec<TranscriptEntry>,
    runtime: &runtime::Runtime,
    active_connection: &mut RuntimeConnection,
    active_model: &mut String,
    tools_enabled: &mut bool,
) -> std_io::Result<AppAction> {
    match dispatch {
        CommandDispatch::Quit => Ok(AppAction::Quit),
        CommandDispatch::SwitchModel(model_id) => {
            tracing::info!(
                from = %active_model,
                to = %model_id,
                "switching runtime model"
            );

            match runtime.open_connection(&model_id).await {
                Ok(connection) => {
                    if let Err(err) = active_connection.end_connection().await {
                        push_system_message(
                            transcript,
                            format!("failed to close previous model connection: {err}"),
                        );
                    }

                    *active_connection = connection;
                    active_model.clear();
                    active_model.push_str(&model_id);
                    push_system_message(transcript, format!("active model: {active_model}"));
                }
                Err(err) => {
                    push_system_message(transcript, err.to_string());
                }
            }

            command_palette.close();
            open_footer_compose(mode, selected_chat_entry, compose_target, chat_draft);

            Ok(AppAction::Continue)
        }
        CommandDispatch::SetToolsEnabled(enabled) => {
            let changed = *tools_enabled != enabled;
            *tools_enabled = enabled;
            let status = if enabled { "enabled" } else { "disabled" };
            push_system_message(transcript, format!("tools prompt is now {status}"));
            if changed {
                tracing::info!(enabled, "tool prompt toggle requested");
            }

            command_palette.close();
            open_footer_compose(mode, selected_chat_entry, compose_target, chat_draft);
            Ok(AppAction::Continue)
        }
        CommandDispatch::OpenModelPrefix | CommandDispatch::OpenToolsPrefix => {
            Ok(AppAction::Continue)
        }
    }
}

fn push_command_error(transcript: &mut Vec<TranscriptEntry>, err: CommandParseError) {
    match err {
        CommandParseError::Empty => {}
        CommandParseError::Incomplete(message) | CommandParseError::Invalid(message) => {
            push_system_message(transcript, message);
        }
    }
}

fn handle_mouse_event(
    kind: MouseEventKind,
    mode: &mut Mode,
    selected_chat_entry: &mut Option<usize>,
    draft: &mut String,
    compose_target: &mut ComposeTarget,
    scroll_anchor: &mut ScrollAnchor,
    transcript: &[TranscriptEntry],
) {
    if matches!(*mode, Mode::Command) {
        return;
    }

    let delta = match kind {
        MouseEventKind::ScrollUp => Some(-1),
        MouseEventKind::ScrollDown => Some(1),
        _ => None,
    };

    let Some(delta) = delta else {
        return;
    };

    if matches!(*mode, Mode::Insert) {
        if delta > 0 {
            return;
        }
        *mode = Mode::Normal;
        *scroll_anchor = ScrollAnchor::Top;
        *selected_chat_entry = move_selection(transcript, *selected_chat_entry, delta);
        return;
    }

    if delta > 0 && is_at_bottom(transcript, *selected_chat_entry) {
        *selected_chat_entry = None;
        *compose_target = ComposeTarget::AppendBottom;
        *scroll_anchor = ScrollAnchor::Bottom;
        enter_insert_mode(mode, draft);
        return;
    }

    *scroll_anchor = if delta > 0 {
        ScrollAnchor::Bottom
    } else {
        ScrollAnchor::Top
    };
    *selected_chat_entry = move_selection(transcript, *selected_chat_entry, delta);
}

async fn submit_chat_turn(
    input: String,
    mode: &mut Mode,
    selected_chat_entry: &mut Option<usize>,
    compose_target: &mut ComposeTarget,
    scroll_anchor: &mut ScrollAnchor,
    transcript: &mut Vec<TranscriptEntry>,
    runtime: &runtime::Runtime,
    active_connection: &mut RuntimeConnection,
    active_model: &mut String,
    tools_enabled: bool,
    tools: &[ToolSpec],
    ui: &mut TerminalUi,
) -> std_io::Result<()> {
    match *compose_target {
        ComposeTarget::InsertAfterSelection => {
            replace_selected_chat_entry(transcript, *selected_chat_entry, input);
        }
        ComposeTarget::AppendBottom => {
            transcript.push(TranscriptEntry::User(input));
        }
    }
    transcript.push(TranscriptEntry::System {
        content: format!("model {active_model} finished loading"),
        status: Some(SystemStatus::Loading {
            model_id: active_model.clone(),
            started_at: Instant::now(),
        }),
    });
    transcript.push(TranscriptEntry::Assistant {
        model_id: active_model.clone(),
        content: String::new(),
        callouts: Vec::new(),
        status: Some(AssistantStatus::Loading {
            started_at: Instant::now(),
        }),
    });

    *selected_chat_entry = None;
    *compose_target = ComposeTarget::AppendBottom;
    *scroll_anchor = ScrollAnchor::Bottom;

    let Some(config) = runtime.config(active_model) else {
        pop_pending_assistant(transcript);
        push_system_message(
            transcript,
            format!("active model disappeared from registry: {active_model}"),
        );
        return Ok(());
    };

    let request = build_request(
        transcript,
        config
            .model
            .tool_format()
            .system_prompt(SYSTEM_PROMPT, if tools_enabled { tools } else { &[] }),
        tools_enabled,
    );
    let tool_format = config.model.tool_format();
    let (status_tx, mut status_rx) = mpsc::unbounded_channel();
    let mut response_stream = match active_connection
        .stream_with_status(request, Some(status_tx))
        .await
    {
        Ok(stream) => stream,
        Err(err) => {
            pop_pending_assistant(transcript);
            push_system_message(transcript, format!("runtime error: {err}"));
            return Ok(());
        }
    };

    let mut response = String::new();
    let mut completed = false;
    let mut tick = tokio::time::interval(Duration::from_millis(100));
    let ctrl_c = tokio::signal::ctrl_c();
    tokio::pin!(ctrl_c);

    if let Err(err) = redraw(
        ui,
        transcript,
        *selected_chat_entry,
        *mode,
        "",
        *compose_target,
        *scroll_anchor,
        None,
    ) {
        push_system_message(transcript, format!("failed to render loading state: {err}"));
    }

    loop {
        tokio::select! {
            _ = &mut ctrl_c => {
                runtime.request_shutdown();
                return Err(std_io::Error::new(ErrorKind::Interrupted, "ctrl-c"));
            }
            _ = tick.tick() => {
                if let Err(err) = redraw(
                    ui,
                    transcript,
                    *selected_chat_entry,
                    *mode,
                    "",
                    *compose_target,
                    *scroll_anchor,
                    None,
                ) {
                    push_system_message(transcript, format!("failed to refresh spinner: {err}"));
                    break;
                }
            }
            status = status_rx.recv() => {
                if let Some(status) = status {
                    apply_runtime_status(transcript, status);
                    if let Err(err) = redraw(
                        ui,
                        transcript,
                        *selected_chat_entry,
                        *mode,
                        "",
                        *compose_target,
                        *scroll_anchor,
                        None,
                    ) {
                        push_system_message(transcript, format!("failed to render status: {err}"));
                        break;
                    }
                }
            }
            event = response_stream.next() => {
                match event {
                    Some(RuntimeResponseEvent::Chunk { content, .. }) => {
                        response.push_str(&content);
                        append_assistant_chunk(transcript, &content);
                        if let Err(err) = redraw(
                            ui,
                            transcript,
                            *selected_chat_entry,
                            *mode,
                            "",
                            *compose_target,
                            *scroll_anchor,
                            None,
                        ) {
                            push_system_message(transcript, format!("failed to render response chunk: {err}"));
                            break;
                        }
                    }
                    Some(RuntimeResponseEvent::Complete { .. }) => {
                        completed = true;
                        clear_pending_status(transcript);
                        if let Err(err) = redraw(
                            ui,
                            transcript,
                            *selected_chat_entry,
                            *mode,
                            "",
                            *compose_target,
                            *scroll_anchor,
                            None,
                        ) {
                            push_system_message(transcript, format!("failed to finalize response: {err}"));
                        }
                        break;
                    }
                    Some(RuntimeResponseEvent::Error { error, .. }) => {
                        clear_pending_status(transcript);
                        push_system_message(transcript, format!("runtime error: {error}"));
                        if let Err(err) = redraw(
                            ui,
                            transcript,
                            *selected_chat_entry,
                            *mode,
                            "",
                            *compose_target,
                            *scroll_anchor,
                            None,
                        ) {
                            push_system_message(transcript, format!("failed to render runtime error: {err}"));
                        }
                        break;
                    }
                    None => {
                        clear_pending_status(transcript);
                        if let Err(err) = redraw(
                            ui,
                            transcript,
                            *selected_chat_entry,
                            *mode,
                            "",
                            *compose_target,
                            *scroll_anchor,
                            None,
                        ) {
                            push_system_message(transcript, format!("failed to render closed stream: {err}"));
                        }
                        break;
                    }
                }
            }
        }
    }

    if completed {
        let tool_calls = tool_format.parse(&response);
        if tools_enabled && !tool_calls.is_empty() {
            append_tool_calls(transcript, &tool_calls, tools);
        }
    }

    *mode = Mode::Insert;
    *compose_target = ComposeTarget::AppendBottom;
    if let Err(err) = redraw(
        ui,
        transcript,
        *selected_chat_entry,
        *mode,
        "",
        *compose_target,
        *scroll_anchor,
        None,
    ) {
        push_system_message(
            transcript,
            format!("failed to redraw after response: {err}"),
        );
    }

    Ok(())
}

fn apply_runtime_status(transcript: &mut [TranscriptEntry], status: RuntimeStatus) {
    match status {
        RuntimeStatus::Queued { config_id } => {
            if let Some(TranscriptEntry::Assistant {
                model_id, status, ..
            }) = pending_assistant_entry_mut(transcript)
            {
                *status = Some(AssistantStatus::Loading {
                    started_at: Instant::now(),
                });
                tracing::info!(model = %model_id, queued = true, "generation queued");
            }

            tracing::debug!(model = %config_id, "runtime request queued");
        }
        RuntimeStatus::Loading { config_id, .. } => {
            if let Some(TranscriptEntry::System { content, status }) =
                pending_loading_system_entry_mut(transcript)
            {
                *content = format!("model {config_id} finished loading");
                *status = Some(SystemStatus::Loading {
                    model_id: config_id.clone(),
                    started_at: Instant::now(),
                });
            }

            if let Some(TranscriptEntry::Assistant { status, .. }) =
                pending_assistant_entry_mut(transcript)
            {
                *status = Some(AssistantStatus::Loading {
                    started_at: Instant::now(),
                });
            }
        }
        RuntimeStatus::Generating { config_id } => {
            clear_pending_loading_system_status(transcript, &config_id);

            if let Some(TranscriptEntry::Assistant {
                model_id,
                content,
                callouts: _,
                status,
            }) = pending_assistant_entry_mut(transcript)
            {
                *status = Some(AssistantStatus::Generating {
                    started_at: Instant::now(),
                });

                if content.is_empty() {
                    tracing::debug!(model = %model_id, "generation spinner active");
                }
            }
        }
    }
}

fn clear_pending_status(transcript: &mut [TranscriptEntry]) {
    if let Some(TranscriptEntry::Assistant { status, .. }) = pending_assistant_entry_mut(transcript)
    {
        *status = None;
    }
    clear_latest_loading_system_status(transcript);
}

fn pop_pending_assistant(transcript: &mut Vec<TranscriptEntry>) {
    if matches!(
        transcript.last(),
        Some(TranscriptEntry::Assistant {
            status: Some(_),
            ..
        })
    ) {
        transcript.pop();
    }
}

fn append_assistant_chunk(transcript: &mut [TranscriptEntry], content: &str) {
    if let Some(TranscriptEntry::Assistant {
        content: assistant_content,
        callouts: _,
        status,
        ..
    }) = pending_assistant_entry_mut(transcript)
    {
        *status = None;
        assistant_content.push_str(content);
    }
}

fn pending_assistant_entry_mut(transcript: &mut [TranscriptEntry]) -> Option<&mut TranscriptEntry> {
    transcript
        .iter_mut()
        .rev()
        .find(|entry| matches!(entry, TranscriptEntry::Assistant { .. }))
}

fn pending_loading_system_entry_mut(
    transcript: &mut [TranscriptEntry],
) -> Option<&mut TranscriptEntry> {
    transcript.iter_mut().rev().find(|entry| {
        matches!(
            entry,
            TranscriptEntry::System {
                status: Some(SystemStatus::Loading { .. }),
                ..
            }
        )
    })
}

fn clear_pending_loading_system_status(transcript: &mut [TranscriptEntry], model_id: &str) {
    if let Some(TranscriptEntry::System { status, .. }) =
        transcript.iter_mut().rev().find(|entry| {
            matches!(
                entry,
                TranscriptEntry::System {
                    status: Some(SystemStatus::Loading { model_id: entry_model_id, .. }),
                    ..
                } if entry_model_id == model_id
            )
        })
    {
        *status = None;
    }
}

fn clear_latest_loading_system_status(transcript: &mut [TranscriptEntry]) {
    if let Some(TranscriptEntry::System { status, .. }) =
        pending_loading_system_entry_mut(transcript)
    {
        *status = None;
    }
}

fn move_selection(
    transcript: &[TranscriptEntry],
    selected_chat_entry: Option<usize>,
    delta: isize,
) -> Option<usize> {
    let chat_positions = chat_entry_positions(transcript);
    if chat_positions.is_empty() {
        return None;
    }

    let current = selected_chat_entry.unwrap_or(chat_positions.len() - 1) as isize;
    let next = (current + delta).clamp(0, chat_positions.len().saturating_sub(1) as isize);
    Some(next as usize)
}

fn is_at_bottom(transcript: &[TranscriptEntry], selected_chat_entry: Option<usize>) -> bool {
    let chat_positions = chat_entry_positions(transcript);
    match (chat_positions.len(), selected_chat_entry) {
        (0, _) => true,
        (len, Some(selected)) => selected + 1 >= len,
        (_, None) => true,
    }
}

fn enter_insert_mode(mode: &mut Mode, draft: &mut String) {
    *mode = Mode::Insert;
    draft.clear();
}

fn exit_insert_mode(
    mode: &mut Mode,
    selected_chat_entry: &mut Option<usize>,
    compose_target: ComposeTarget,
    transcript: &[TranscriptEntry],
) {
    *mode = Mode::Normal;
    match compose_target {
        ComposeTarget::InsertAfterSelection => {}
        ComposeTarget::AppendBottom => {
            *selected_chat_entry = move_selection(transcript, None, 0);
        }
    }
}

fn open_footer_compose(
    mode: &mut Mode,
    selected_chat_entry: &mut Option<usize>,
    compose_target: &mut ComposeTarget,
    draft: &mut String,
) {
    *selected_chat_entry = None;
    *compose_target = ComposeTarget::AppendBottom;
    enter_insert_mode(mode, draft);
}

fn open_compose_after_selection(
    transcript: &[TranscriptEntry],
    selected_chat_entry: &mut Option<usize>,
    compose_target: &mut ComposeTarget,
    mode: &mut Mode,
    draft: &mut String,
) {
    let editable_selection = next_editable_selection_after(transcript, *selected_chat_entry);
    let editable_positions = editable_entry_positions(transcript);
    let can_insert_after_selection = editable_selection
        .and_then(|selection| {
            editable_positions
                .iter()
                .position(|candidate| *candidate == selection)
        })
        .is_some_and(|position| position + 1 < editable_positions.len());

    if can_insert_after_selection {
        *selected_chat_entry = editable_selection;
        *compose_target = ComposeTarget::InsertAfterSelection;
    } else {
        *compose_target = ComposeTarget::AppendBottom;
        *selected_chat_entry = None;
    }

    enter_insert_mode(mode, draft);
}

fn next_editable_selection_after(
    transcript: &[TranscriptEntry],
    selected_chat_entry: Option<usize>,
) -> Option<usize> {
    let selected = selected_transcript_index(transcript, selected_chat_entry)?;
    editable_entry_positions(transcript)
        .into_iter()
        .find(|position| *position > selected)
}

fn open_inline_compose_for_selection(
    transcript: &[TranscriptEntry],
    selected_chat_entry: &mut Option<usize>,
    compose_target: &mut ComposeTarget,
    mode: &mut Mode,
    draft: &mut String,
) {
    if let Some(editable_selection) =
        selected_editable_transcript_index(transcript, *selected_chat_entry)
    {
        *selected_chat_entry = Some(editable_selection);
        *compose_target = ComposeTarget::InsertAfterSelection;
        *mode = Mode::Insert;
        draft.clear();
        draft.push_str(&editable_entry_content(&transcript[editable_selection]));
    }
}

fn normalize_selection(transcript: &[TranscriptEntry], selected_chat_entry: &mut Option<usize>) {
    let chat_positions = chat_entry_positions(transcript);
    match (*selected_chat_entry, chat_positions.len()) {
        (_, 0) => *selected_chat_entry = None,
        (Some(selected), len) if selected >= len => *selected_chat_entry = Some(len - 1),
        (None, len) => *selected_chat_entry = Some(len - 1),
        _ => {}
    }
}

fn yank_entry(entry: &TranscriptEntry) -> std_io::Result<()> {
    let mut clipboard = Clipboard::new().map_err(|err| std_io::Error::other(err.to_string()))?;
    clipboard
        .set_text(clipboard_text(entry))
        .map_err(|err| std_io::Error::other(err.to_string()))
}

fn editable_entry_content(entry: &TranscriptEntry) -> String {
    match entry {
        TranscriptEntry::User(content) => content.clone(),
        TranscriptEntry::Assistant { content, .. } => content.clone(),
        TranscriptEntry::System { .. } => String::new(),
    }
}

fn redraw(
    ui: &mut TerminalUi,
    transcript: &[TranscriptEntry],
    selected_chat_entry: Option<usize>,
    mode: Mode,
    draft: &str,
    compose_target: ComposeTarget,
    scroll_anchor: ScrollAnchor,
    command_palette: Option<CommandPaletteView<'_>>,
) -> std_io::Result<()> {
    ui.draw(RenderState {
        entries: transcript,
        selected_chat_entry,
        mode,
        draft,
        prompt_inline: matches!(mode, Mode::Insert)
            && matches!(compose_target, ComposeTarget::InsertAfterSelection),
        scroll_anchor,
        command_palette,
    })
}

fn start_event_reader(tx: mpsc::UnboundedSender<Event>) -> thread::JoinHandle<()> {
    thread::spawn(move || {
        loop {
            match event::read() {
                Ok(event) => {
                    if tx.send(event).is_err() {
                        break;
                    }
                }
                Err(err) => {
                    tracing::error!(error = %err, "failed to read terminal event");
                    break;
                }
            }
        }
    })
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
        .map_err(|err| std_io::Error::other(err.to_string()))
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct StartupOptions {
    model: String,
    tools_enabled: bool,
}

fn parse_startup_options() -> Result<StartupOptions, String> {
    parse_startup_args(env::args().skip(1))
}

fn parse_startup_args<I>(args: I) -> Result<StartupOptions, String>
where
    I: IntoIterator<Item = String>,
{
    let mut args = args.into_iter();
    let mut model = String::from("smollm2-360m");
    let mut tools_enabled = true;

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
            "--tools" => {
                let Some(value) = args.next() else {
                    return Err(String::from("missing value for --tools"));
                };
                tools_enabled = parse_tools_flag(&value)?
            }
            _ if arg.starts_with("--tools=") => {
                tools_enabled = parse_tools_flag(&arg["--tools=".len()..])?;
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

    Ok(StartupOptions {
        model,
        tools_enabled,
    })
}

fn parse_tools_flag(value: &str) -> Result<bool, String> {
    match value.trim().to_ascii_lowercase().as_str() {
        "enable" => Ok(true),
        "disable" => Ok(false),
        _ => Err(String::from("tools must be enable or disable")),
    }
}

fn print_usage() {
    println!("usage: kablam [--model <model-id>] [--tools enable|disable]");
    println!("example: kablam --model qwen3.5-q2k");
    println!("example: kablam --tools disable");
}

fn default_tools() -> Vec<ToolSpec> {
    vec![
        ToolSpec::new(
            "search_docs",
            "Search local project documentation when the user asks about this project or its files.",
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

fn append_tool_calls(
    transcript: &mut Vec<TranscriptEntry>,
    tool_calls: &[agent::ToolCall],
    tools: &[ToolSpec],
) {
    let allowed_tool_names = tools
        .iter()
        .map(|tool| tool.name.as_str())
        .collect::<HashSet<_>>();
    let mut supported_calls = Vec::new();
    let mut unsupported_calls = Vec::new();

    for (index, call) in tool_calls.iter().enumerate() {
        if allowed_tool_names.contains(call.name.as_str()) {
            supported_calls.push((index + 1, call));
        } else {
            unsupported_calls.push((index + 1, call));
        }
    }

    if supported_calls.is_empty() && unsupported_calls.is_empty() {
        return;
    }

    let mut callouts = Vec::new();

    if !supported_calls.is_empty() {
        callouts.push(String::from("parsed tool calls:"));
        for (number, call) in supported_calls {
            let arguments = serde_json::to_string_pretty(&call.arguments)
                .unwrap_or_else(|_| call.arguments.to_string());

            callouts.push(format!("{number}. {}", call.name));
            callouts.push(arguments);
        }
    }

    if !unsupported_calls.is_empty() {
        callouts.push(String::from("ignored unsupported tool calls:"));
        for (number, call) in unsupported_calls {
            let arguments = serde_json::to_string_pretty(&call.arguments)
                .unwrap_or_else(|_| call.arguments.to_string());

            callouts.push(format!("{number}. {}", call.name));
            callouts.push(arguments);
        }
    }

    if let Some(TranscriptEntry::Assistant {
        callouts: entry_callouts,
        ..
    }) = transcript.last_mut()
    {
        entry_callouts.extend(callouts);
    }
}

fn build_request(
    transcript: &[TranscriptEntry],
    system_prompt: String,
    tools_enabled: bool,
) -> RequestBuilder {
    let system_prompt_log = system_prompt.clone();
    let mut messages = TextMessages::new().add_message(TextMessageRole::System, system_prompt);

    for entry in transcript {
        match entry {
            TranscriptEntry::User(content) => {
                messages = messages.add_message(TextMessageRole::User, content);
            }
            TranscriptEntry::Assistant {
                content, status, ..
            } if status.is_none() => {
                messages = messages.add_message(TextMessageRole::Assistant, content);
            }
            TranscriptEntry::Assistant { .. } | TranscriptEntry::System { .. } => {}
        }
    }

    tracing::info!(
        max_generation_tokens = MAX_GENERATION_TOKENS,
        message_count = messages.messages_ref().len(),
        tools_enabled,
        "building chat generation request"
    );
    tracing::info!(
        "chat prompt sent to model:\n{}",
        render_chat_log(transcript, &system_prompt_log)
    );

    RequestBuilder::from(messages).set_sampler_max_len(MAX_GENERATION_TOKENS)
}

fn render_chat_log(transcript: &[TranscriptEntry], system_prompt: &str) -> String {
    let mut output = String::new();

    output.push_str("system:\n");
    output.push_str(system_prompt);

    for entry in transcript {
        match entry {
            TranscriptEntry::User(content) => {
                output.push_str("\n\n");
                output.push_str("user:\n");
                output.push_str(content);
            }
            TranscriptEntry::Assistant {
                model_id,
                content,
                status,
                ..
            } if status.is_none() => {
                output.push_str("\n\n");
                output.push_str("assistant:\n");
                output.push_str(model_id);
                output.push('\n');
                output.push_str(content);
            }
            TranscriptEntry::Assistant { .. } | TranscriptEntry::System { .. } => {}
        }
    }

    output
}

fn push_system_message(transcript: &mut Vec<TranscriptEntry>, message: String) {
    transcript.push(TranscriptEntry::System {
        content: message,
        status: None,
    });
}

fn is_editable_entry(entry: &TranscriptEntry) -> bool {
    matches!(
        entry,
        TranscriptEntry::User(_) | TranscriptEntry::Assistant { .. }
    )
}

fn editable_entry_positions(transcript: &[TranscriptEntry]) -> Vec<usize> {
    transcript
        .iter()
        .enumerate()
        .filter_map(|(index, entry)| is_editable_entry(entry).then_some(index))
        .collect()
}

fn selected_editable_transcript_index(
    transcript: &[TranscriptEntry],
    selected_chat_entry: Option<usize>,
) -> Option<usize> {
    selected_transcript_index(transcript, selected_chat_entry)
        .filter(|index| transcript.get(*index).is_some_and(is_editable_entry))
}

fn truncate_history_for_insertion(
    transcript: &mut Vec<TranscriptEntry>,
    selected_chat_entry: Option<usize>,
) {
    let insertion_index = selected_editable_transcript_index(transcript, selected_chat_entry)
        .map(|index| index + 1)
        .unwrap_or(transcript.len());
    transcript.truncate(insertion_index);
}

fn replace_selected_chat_entry(
    transcript: &mut Vec<TranscriptEntry>,
    selected_chat_entry: Option<usize>,
    input: String,
) {
    match selected_editable_transcript_index(transcript, selected_chat_entry) {
        Some(selected_index) => {
            transcript.truncate(selected_index + 1);
            transcript[selected_index] = TranscriptEntry::User(input);
        }
        None => {
            transcript.push(TranscriptEntry::User(input));
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AppAction {
    Continue,
    Quit,
}

#[cfg(test)]
mod tests {
    use super::{
        ComposeTarget, Mode, StartupOptions, TranscriptEntry, exit_insert_mode,
        open_compose_after_selection, open_footer_compose, open_inline_compose_for_selection,
        parse_startup_args, replace_selected_chat_entry, truncate_history_for_insertion,
    };

    #[test]
    fn truncating_for_insertion_removes_tail_after_selected_chat() {
        let mut transcript = vec![
            TranscriptEntry::User("first".to_string()),
            TranscriptEntry::Assistant {
                model_id: "m".to_string(),
                content: "second".to_string(),
                callouts: vec![],
                status: None,
            },
            TranscriptEntry::User("third".to_string()),
        ];

        truncate_history_for_insertion(&mut transcript, Some(0));

        assert_eq!(transcript.len(), 1);
        assert!(matches!(transcript[0], TranscriptEntry::User(_)));
    }

    #[test]
    fn replacing_selected_chat_entry_overwrites_in_place_and_drops_tail() {
        let mut transcript = vec![
            TranscriptEntry::User("first".to_string()),
            TranscriptEntry::Assistant {
                model_id: "m".to_string(),
                content: "second".to_string(),
                callouts: vec![],
                status: None,
            },
            TranscriptEntry::User("third".to_string()),
        ];

        replace_selected_chat_entry(&mut transcript, Some(1), "replacement".to_string());

        assert_eq!(transcript.len(), 2);
        assert!(
            matches!(transcript[1], TranscriptEntry::User(ref value) if value == "replacement")
        );
    }

    #[test]
    fn open_compose_after_selection_targets_the_next_message_when_available() {
        let transcript = vec![
            TranscriptEntry::User("first".to_string()),
            TranscriptEntry::Assistant {
                model_id: "m".to_string(),
                content: "second".to_string(),
                callouts: vec![],
                status: None,
            },
            TranscriptEntry::User("third".to_string()),
        ];
        let mut selected_chat_entry = Some(0);
        let mut compose_target = ComposeTarget::AppendBottom;
        let mut mode = Mode::Normal;
        let mut draft = String::from("old");

        open_compose_after_selection(
            &transcript,
            &mut selected_chat_entry,
            &mut compose_target,
            &mut mode,
            &mut draft,
        );

        assert_eq!(compose_target, ComposeTarget::InsertAfterSelection);
        assert_eq!(selected_chat_entry, Some(1));
        assert_eq!(mode, Mode::Insert);
        assert!(draft.is_empty());
    }

    #[test]
    fn open_compose_after_selection_appends_at_bottom_when_at_tail() {
        let transcript = vec![
            TranscriptEntry::User("first".to_string()),
            TranscriptEntry::Assistant {
                model_id: "m".to_string(),
                content: "second".to_string(),
                callouts: vec![],
                status: None,
            },
        ];
        let mut selected_chat_entry = Some(1);
        let mut compose_target = ComposeTarget::InsertAfterSelection;
        let mut mode = Mode::Normal;
        let mut draft = String::from("old");

        open_compose_after_selection(
            &transcript,
            &mut selected_chat_entry,
            &mut compose_target,
            &mut mode,
            &mut draft,
        );

        assert_eq!(compose_target, ComposeTarget::AppendBottom);
        assert_eq!(selected_chat_entry, None);
        assert_eq!(mode, Mode::Insert);
        assert!(draft.is_empty());
    }

    #[test]
    fn open_compose_after_selection_moves_system_selection_to_next_editable_message() {
        let transcript = vec![
            TranscriptEntry::System {
                content: "system".to_string(),
                status: None,
            },
            TranscriptEntry::Assistant {
                model_id: "m".to_string(),
                content: "editable".to_string(),
                callouts: vec![],
                status: None,
            },
            TranscriptEntry::User("tail".to_string()),
        ];
        let mut selected_chat_entry = Some(0);
        let mut compose_target = ComposeTarget::AppendBottom;
        let mut mode = Mode::Normal;
        let mut draft = String::from("old");

        open_compose_after_selection(
            &transcript,
            &mut selected_chat_entry,
            &mut compose_target,
            &mut mode,
            &mut draft,
        );

        assert_eq!(selected_chat_entry, Some(1));
        assert_eq!(compose_target, ComposeTarget::InsertAfterSelection);
        assert_eq!(mode, Mode::Insert);
        assert!(draft.is_empty());
    }

    #[test]
    fn open_compose_after_selection_appends_at_bottom_when_there_is_no_next_message() {
        let transcript = vec![
            TranscriptEntry::User("first".to_string()),
            TranscriptEntry::Assistant {
                model_id: "m".to_string(),
                content: "second".to_string(),
                callouts: vec![],
                status: None,
            },
        ];
        let mut selected_chat_entry = Some(1);
        let mut compose_target = ComposeTarget::InsertAfterSelection;
        let mut mode = Mode::Normal;
        let mut draft = String::from("old");

        open_compose_after_selection(
            &transcript,
            &mut selected_chat_entry,
            &mut compose_target,
            &mut mode,
            &mut draft,
        );

        assert_eq!(compose_target, ComposeTarget::AppendBottom);
        assert_eq!(selected_chat_entry, None);
        assert_eq!(mode, Mode::Insert);
        assert!(draft.is_empty());
    }

    #[test]
    fn open_inline_compose_for_selection_prefills_the_selected_message() {
        let transcript = vec![
            TranscriptEntry::User("first".to_string()),
            TranscriptEntry::Assistant {
                model_id: "m".to_string(),
                content: "second".to_string(),
                callouts: vec![],
                status: None,
            },
        ];
        let mut selected_chat_entry = Some(1);
        let mut compose_target = ComposeTarget::AppendBottom;
        let mut mode = Mode::Normal;
        let mut draft = String::from("old");

        open_inline_compose_for_selection(
            &transcript,
            &mut selected_chat_entry,
            &mut compose_target,
            &mut mode,
            &mut draft,
        );

        assert_eq!(selected_chat_entry, Some(1));
        assert_eq!(compose_target, ComposeTarget::InsertAfterSelection);
        assert_eq!(mode, Mode::Insert);
        assert_eq!(draft, "second");
    }

    #[test]
    fn open_footer_compose_clears_selection_and_opens_append_bottom_prompt() {
        let mut selected_chat_entry = Some(3);
        let mut compose_target = ComposeTarget::InsertAfterSelection;
        let mut mode = Mode::Normal;
        let mut draft = String::from("old");

        open_footer_compose(
            &mut mode,
            &mut selected_chat_entry,
            &mut compose_target,
            &mut draft,
        );

        assert_eq!(selected_chat_entry, None);
        assert_eq!(compose_target, ComposeTarget::AppendBottom);
        assert_eq!(mode, Mode::Insert);
        assert!(draft.is_empty());
    }

    #[test]
    fn exiting_inline_insert_mode_keeps_the_same_history_selection() {
        let transcript = vec![
            TranscriptEntry::User("first".to_string()),
            TranscriptEntry::Assistant {
                model_id: "m".to_string(),
                content: "second".to_string(),
                callouts: vec![],
                status: None,
            },
        ];
        let mut selected_chat_entry = Some(1);
        let mut mode = Mode::Insert;

        exit_insert_mode(
            &mut mode,
            &mut selected_chat_entry,
            ComposeTarget::InsertAfterSelection,
            &transcript,
        );

        assert_eq!(mode, Mode::Normal);
        assert_eq!(selected_chat_entry, Some(1));
    }

    #[test]
    fn exiting_footer_insert_mode_selects_the_last_history_entry() {
        let transcript = vec![
            TranscriptEntry::User("first".to_string()),
            TranscriptEntry::Assistant {
                model_id: "m".to_string(),
                content: "second".to_string(),
                callouts: vec![],
                status: None,
            },
        ];
        let mut selected_chat_entry = None;
        let mut mode = Mode::Insert;

        exit_insert_mode(
            &mut mode,
            &mut selected_chat_entry,
            ComposeTarget::AppendBottom,
            &transcript,
        );

        assert_eq!(mode, Mode::Normal);
        assert_eq!(selected_chat_entry, Some(1));
    }

    #[test]
    fn parse_startup_args_accepts_tools_enable_and_disable() {
        let startup = parse_startup_args(vec![
            String::from("--tools"),
            String::from("disable"),
            String::from("--model"),
            String::from("qwen3.5-q2k"),
        ])
        .expect("startup args should parse");

        assert_eq!(
            startup,
            StartupOptions {
                model: String::from("qwen3.5-q2k"),
                tools_enabled: false,
            }
        );
    }

    #[test]
    fn replace_selected_chat_entry_does_not_overwrite_system_messages() {
        let mut transcript = vec![TranscriptEntry::System {
            content: "system".to_string(),
            status: None,
        }];

        replace_selected_chat_entry(&mut transcript, Some(0), "replacement".to_string());

        assert_eq!(transcript.len(), 2);
        assert!(matches!(transcript[0], TranscriptEntry::System { .. }));
        assert!(
            matches!(transcript[1], TranscriptEntry::User(ref value) if value == "replacement")
        );
    }
}
