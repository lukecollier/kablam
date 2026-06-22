mod agent;
mod command_palette;
mod runtime;
mod terminal;

use std::collections::{HashMap, HashSet, VecDeque};
use std::env;
use std::fs::{self, OpenOptions};
use std::io::{self as std_io, ErrorKind};
use std::thread;
use std::time::{Duration, Instant};

use agent::{
    MarkdownAccumulator, ParsedMarkdown, ToolParameter, ToolParameterKind, ToolSpec,
    blocks_to_raw_text,
};
use arboard::Clipboard;
use command_palette::{
    CommandCommit, CommandDispatch, CommandPaletteState, CommandParseError, TabTarget,
};
use crossterm::event::{self, Event, KeyCode, KeyModifiers, MouseEventKind};
use mdstream::Block as MarkdownBlock;
use mistralrs::{RequestBuilder, RequestLike, TextMessageRole, TextMessages};
use runtime::{RuntimeBuilder, RuntimeConnection, RuntimeResponseEvent, RuntimeStatus};
use terminal::{
    AssistantStatus, HistoryEntry, Mode, ModelLoadStatus, RenderState, ScrollAnchor, TabRenderInfo,
    TerminalUi, chat_entry_positions, clipboard_text, selected_transcript_index,
};
use tokio::sync::mpsc;
use tracing_subscriber::EnvFilter;
use tracing_subscriber::filter::LevelFilter;

const SYSTEM_PROMPT: &str = "You are a concise assistant. If a user asks for information that an available tool can provide, call the tool instead of answering directly.";
const LOG_FILE: &str = "logs/kablam.log";
const MAX_GENERATION_TOKENS: usize = 2048;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ComposeTarget {
    EditEntry(u64),
    AppendBottom,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum InlineComposePosition {
    Before,
    After,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum InlineCommandPosition {
    Before,
    After,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum HistoryMeta {
    None,
    Command(CommandKind),
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum SelectionEditAction {
    Prompt(String),
    Command(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum CommandKind {
    Model(String),
    Tools(bool),
    ModelList,
}

impl CommandKind {
    fn family(&self) -> Option<CommandFamily> {
        match self {
            Self::Model(_) => Some(CommandFamily::Model),
            Self::Tools(_) => Some(CommandFamily::Tools),
            Self::ModelList => None,
        }
    }

    fn matches_semantics(&self, other: &Self) -> bool {
        self == other
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CommandFamily {
    Model,
    Tools,
}

#[derive(Debug, Clone)]
struct HistoryItem {
    id: u64,
    entry: HistoryEntry,
    meta: HistoryMeta,
}

#[derive(Debug, Clone)]
struct ThreadConfig {
    model_id: String,
    tools_enabled: bool,
}

#[derive(Debug, Clone)]
struct ThreadJob {
    assistant_entry_id: u64,
}

struct ThreadState {
    tab_id: usize,
    base_config: ThreadConfig,
    current_config: ThreadConfig,
    name_override: Option<String>,
    history: Vec<HistoryItem>,
    selected_chat_entry: Option<usize>,
    compose_target: ComposeTarget,
    draft: String,
    scroll_anchor: ScrollAnchor,
    queue: VecDeque<ThreadJob>,
    running_job: bool,
    running_assistant_id: Option<u64>,
    loaded_models: HashSet<String>,
    connections: HashMap<String, RuntimeConnection>,
    has_unseen_changes: bool,
}

struct AppState {
    threads: Vec<ThreadState>,
    active_thread_idx: usize,
    mode: Mode,
    command_palette: CommandPaletteState,
    next_history_id: u64,
    next_tab_id: usize,
    command_edit_target: Option<u64>,
    inline_command_placeholder: Option<(u64, usize)>,
    delete_target_id: Option<u64>,
}

enum AppAction {
    Continue,
    Quit,
}

enum AppEvent {
    Input(Event),
    Tick,
    Job(JobEvent),
}

enum JobEvent {
    Status {
        tab_id: usize,
        assistant_entry_id: u64,
        status: RuntimeStatus,
    },
    Chunk {
        tab_id: usize,
        assistant_entry_id: u64,
        markdown: ParsedMarkdown,
    },
    Complete {
        tab_id: usize,
        assistant_entry_id: u64,
        markdown: ParsedMarkdown,
        callouts: Vec<String>,
    },
    Interrupted {
        tab_id: usize,
        assistant_entry_id: u64,
        markdown: ParsedMarkdown,
    },
    Error {
        tab_id: usize,
        assistant_entry_id: u64,
        error: String,
    },
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

    let initial_connection = match runtime.open_connection(&startup.model).await {
        Ok(connection) => connection,
        Err(err) => {
            eprintln!("failed to open initial model connection: {err}");
            return;
        }
    };

    let initial_thread = ThreadState {
        tab_id: 1,
        base_config: ThreadConfig {
            model_id: startup.model.clone(),
            tools_enabled: startup.tools_enabled,
        },
        current_config: ThreadConfig {
            model_id: startup.model.clone(),
            tools_enabled: startup.tools_enabled,
        },
        name_override: None,
        history: Vec::new(),
        selected_chat_entry: None,
        compose_target: ComposeTarget::AppendBottom,
        draft: String::new(),
        scroll_anchor: ScrollAnchor::Bottom,
        queue: VecDeque::new(),
        running_job: false,
        running_assistant_id: None,
        loaded_models: HashSet::new(),
        connections: HashMap::from([(startup.model.clone(), initial_connection)]),
        has_unseen_changes: false,
    };

    let mut app = AppState {
        threads: vec![initial_thread],
        active_thread_idx: 0,
        mode: Mode::Insert,
        command_palette: CommandPaletteState::new(
            runtime
                .list_configs()
                .iter()
                .map(|config| config.id.clone())
                .collect(),
        ),
        next_history_id: 1,
        next_tab_id: 2,
        command_edit_target: None,
        inline_command_placeholder: None,
        delete_target_id: None,
    };

    let mut ui = match TerminalUi::new() {
        Ok(ui) => ui,
        Err(err) => {
            eprintln!("failed to initialize terminal UI: {err}");
            return;
        }
    };

    let (event_tx, mut event_rx) = mpsc::unbounded_channel();
    let _event_thread = start_event_reader(event_tx.clone());
    let _tick_task = start_tick_loop(event_tx.clone());

    if let Err(err) = redraw(&mut ui, &app) {
        eprintln!("failed to render UI: {err}");
        return;
    }

    loop {
        let Some(event) = event_rx.recv().await else {
            break;
        };

        let outcome = match event {
            AppEvent::Input(event) => {
                handle_input_event(event, &mut app, &runtime, &tools, event_tx.clone()).await
            }
            AppEvent::Tick => Ok(AppAction::Continue),
            AppEvent::Job(event) => {
                handle_job_event(event, &mut app, &runtime, &tools, event_tx.clone()).await;
                Ok(AppAction::Continue)
            }
        };

        match outcome {
            Ok(AppAction::Continue) => {
                normalize_selection(active_thread_mut(&mut app));
                if let Err(err) = redraw(&mut ui, &app) {
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

    shutdown_connections(&mut app).await;
}

async fn handle_input_event(
    event: Event,
    app: &mut AppState,
    runtime: &runtime::Runtime,
    tools: &[ToolSpec],
    app_tx: mpsc::UnboundedSender<AppEvent>,
) -> std_io::Result<AppAction> {
    app.command_palette.set_model_ids(
        runtime
            .list_configs()
            .iter()
            .map(|config| config.id.clone()),
    );
    app.command_palette.set_tab_labels(tab_labels(&app.threads));

    match event {
        Event::Key(key) => {
            if key.code == KeyCode::Char('c') && key.modifiers.contains(KeyModifiers::CONTROL) {
                return handle_ctrl_c(app).await;
            }
            if matches!(key.code, KeyCode::Left | KeyCode::Right)
                && !matches!(app.mode, Mode::Command | Mode::ConfirmDelete)
            {
                cycle_active_tab_relative(app, if key.code == KeyCode::Left { -1 } else { 1 });
                return Ok(AppAction::Continue);
            }

            match app.mode {
                Mode::Insert => handle_insert_key_event(key, app, runtime, tools, app_tx).await,
                Mode::Normal => handle_normal_key_event(key, app),
                Mode::Command => handle_command_key_event(key, app, runtime, tools, app_tx).await,
                Mode::ConfirmDelete => {
                    handle_confirm_delete_key_event(key, app, runtime, tools, app_tx).await
                }
            }
        }
        Event::Mouse(mouse) => {
            handle_mouse_event(mouse.kind, app);
            Ok(AppAction::Continue)
        }
        Event::Resize(_, _) => Ok(AppAction::Continue),
        _ => Ok(AppAction::Continue),
    }
}

async fn handle_insert_key_event(
    key: crossterm::event::KeyEvent,
    app: &mut AppState,
    runtime: &runtime::Runtime,
    tools: &[ToolSpec],
    app_tx: mpsc::UnboundedSender<AppEvent>,
) -> std_io::Result<AppAction> {
    let thread = active_thread_mut(app);
    match key.code {
        KeyCode::Esc => {
            exit_insert_mode(thread);
            app.mode = Mode::Normal;
        }
        KeyCode::Enter => {
            let input = thread.draft.trim().to_string();
            thread.draft.clear();
            if input.is_empty() {
                return Ok(AppAction::Continue);
            }
            submit_chat_turn(app, input, runtime, tools, app_tx).await?;
        }
        KeyCode::Backspace => {
            thread.draft.pop();
        }
        KeyCode::Char(ch)
            if !key
                .modifiers
                .intersects(KeyModifiers::CONTROL | KeyModifiers::ALT | KeyModifiers::SUPER) =>
        {
            thread.draft.push(ch);
        }
        _ => {}
    }

    Ok(AppAction::Continue)
}

async fn handle_ctrl_c(app: &mut AppState) -> std_io::Result<AppAction> {
    let Some((thread_index, assistant_id, sequence_number)) = selected_interrupt_target(app) else {
        return Ok(AppAction::Quit);
    };

    let model_id = app.threads[thread_index].current_config.model_id.clone();
    let Some(connection) = app.threads[thread_index]
        .connections
        .get(&model_id)
        .cloned()
    else {
        return Ok(AppAction::Continue);
    };

    if let Some(sequence_number) = sequence_number {
        connection
            .interrupt_sequence(sequence_number)
            .await
            .map_err(|err| std_io::Error::other(err.to_string()))?;
    } else {
        cancel_queued_assistant(app, thread_index, assistant_id);
    }
    tracing::info!(
        tab_id = app.threads[thread_index].tab_id,
        assistant_entry_id = assistant_id,
        "interrupt requested for running assistant"
    );

    Ok(AppAction::Continue)
}

fn handle_normal_key_event(
    key: crossterm::event::KeyEvent,
    app: &mut AppState,
) -> std_io::Result<AppAction> {
    match key.code {
        KeyCode::Enter => open_footer_compose(app),
        KeyCode::Char(':') => {
            app.mode = Mode::Command;
            app.command_edit_target = None;
            app.command_palette.open();
        }
        KeyCode::Char('i') => {
            let selected = {
                let thread = active_thread(app);
                selected_transcript_index(&history_entries(thread), thread.selected_chat_entry)
            };
            if let Some(index) = selected {
                let action = {
                    let thread = active_thread(app);
                    selection_edit_action(&thread.history[index].entry)
                        .map(|action| (thread.history[index].id, action))
                };
                if let Some((entry_id, action)) = action {
                    match action {
                        SelectionEditAction::Prompt(draft) => {
                            let thread = active_thread_mut(app);
                            thread.selected_chat_entry = Some(index);
                            thread.compose_target = ComposeTarget::EditEntry(entry_id);
                            thread.draft = draft;
                            app.mode = Mode::Insert;
                        }
                        SelectionEditAction::Command(draft) => {
                            app.mode = Mode::Command;
                            app.command_edit_target = Some(entry_id);
                            app.command_palette.open_with_draft(draft);
                        }
                    }
                }
            }
        }
        KeyCode::Char('o') => {
            open_inline_compose_relative(app, InlineComposePosition::After);
        }
        KeyCode::Char('O') => {
            open_inline_compose_relative(app, InlineComposePosition::Before);
        }
        KeyCode::Char('j') | KeyCode::Down => {
            let thread = active_thread_mut(app);
            thread.scroll_anchor = ScrollAnchor::Bottom;
            thread.selected_chat_entry = move_selection(thread, 1);
        }
        KeyCode::Char('k') | KeyCode::Up => {
            let thread = active_thread_mut(app);
            thread.scroll_anchor = ScrollAnchor::Top;
            thread.selected_chat_entry = move_selection(thread, -1);
        }
        KeyCode::Char('y') => {
            let selected = {
                let thread = active_thread(app);
                selected_transcript_index(&history_entries(thread), thread.selected_chat_entry)
            };
            if let Some(index) = selected {
                let result = {
                    let thread = active_thread(app);
                    let entries = history_entries(thread);
                    yank_entry(&entries, index)
                };
                if let Err(err) = result {
                    push_notice_current(app, format!("clipboard error: {err}"));
                }
            }
        }
        KeyCode::Char('d') => {
            let thread = active_thread(app);
            if let Some(index) =
                selected_transcript_index(&history_entries(thread), thread.selected_chat_entry)
            {
                app.delete_target_id = Some(thread.history[index].id);
                app.mode = Mode::ConfirmDelete;
            }
        }
        KeyCode::Left => cycle_active_tab_relative(app, -1),
        KeyCode::Right => cycle_active_tab_relative(app, 1),
        KeyCode::Tab => cycle_active_tab(app),
        _ => {}
    }

    Ok(AppAction::Continue)
}

async fn handle_command_key_event(
    key: crossterm::event::KeyEvent,
    app: &mut AppState,
    runtime: &runtime::Runtime,
    tools: &[ToolSpec],
    app_tx: mpsc::UnboundedSender<AppEvent>,
) -> std_io::Result<AppAction> {
    match key.code {
        KeyCode::Esc => {
            cancel_inline_command_insert(app);
            app.command_palette.close();
            app.command_edit_target = None;
            app.mode = Mode::Normal;
        }
        KeyCode::Tab => {
            let _ = app.command_palette.cycle_selection();
        }
        KeyCode::Enter => match app.command_palette.commit() {
            Ok(CommandCommit::StayOpen) => {}
            Ok(CommandCommit::Execute(dispatch)) => {
                return execute_command_dispatch(dispatch, app, runtime, tools, app_tx).await;
            }
            Err(err) => {
                push_command_error(&mut app.command_palette, err);
            }
        },
        KeyCode::Backspace => app.command_palette.backspace(),
        KeyCode::Char(ch)
            if !key
                .modifiers
                .intersects(KeyModifiers::CONTROL | KeyModifiers::ALT | KeyModifiers::SUPER) =>
        {
            app.command_palette.input_char(ch)
        }
        _ => {}
    }
    Ok(AppAction::Continue)
}

async fn handle_confirm_delete_key_event(
    key: crossterm::event::KeyEvent,
    app: &mut AppState,
    runtime: &runtime::Runtime,
    tools: &[ToolSpec],
    app_tx: mpsc::UnboundedSender<AppEvent>,
) -> std_io::Result<AppAction> {
    match key.code {
        KeyCode::Char('n') | KeyCode::Esc => {
            app.delete_target_id = None;
            app.mode = Mode::Normal;
        }
        KeyCode::Char('y') => {
            if let Some(target_id) = app.delete_target_id.take() {
                delete_history_entry(app, target_id, runtime, tools, app_tx).await?;
            }
            app.mode = Mode::Normal;
        }
        _ => {}
    }

    Ok(AppAction::Continue)
}

async fn execute_command_dispatch(
    dispatch: CommandDispatch,
    app: &mut AppState,
    runtime: &runtime::Runtime,
    tools: &[ToolSpec],
    app_tx: mpsc::UnboundedSender<AppEvent>,
) -> std_io::Result<AppAction> {
    match dispatch {
        CommandDispatch::Quit => Ok(AppAction::Quit),
        CommandDispatch::OpenModelPrefix
        | CommandDispatch::OpenToolsPrefix
        | CommandDispatch::OpenTabPrefix => Ok(AppAction::Continue),
        CommandDispatch::OpenCommandAfter => {
            open_inline_command_relative(app, InlineCommandPosition::After);
            Ok(AppAction::Continue)
        }
        CommandDispatch::OpenCommandBefore => {
            open_inline_command_relative(app, InlineCommandPosition::Before);
            Ok(AppAction::Continue)
        }
        CommandDispatch::ListModels => {
            let models = runtime
                .list_configs()
                .iter()
                .map(|config| config.id.clone())
                .collect::<Vec<_>>()
                .join("\n");
            apply_history_command_edit_or_insert(
                app,
                runtime,
                tools,
                app_tx,
                "model ls".to_string(),
                HistoryEntry::Command {
                    raw: "model ls".to_string(),
                    result: models,
                },
                HistoryMeta::Command(CommandKind::ModelList),
                false,
            )
            .await?;
            Ok(AppAction::Continue)
        }
        CommandDispatch::SwitchModel(model_id) => {
            if runtime.config(&model_id).is_none() {
                app.command_palette.set_error(format!(
                    "{model_id} is not a recognised model, use :model ls to show available models"
                ));
                return Ok(AppAction::Continue);
            }
            ensure_thread_connection(active_thread_mut(app), runtime, &model_id).await?;
            apply_history_command_edit_or_insert(
                app,
                runtime,
                tools,
                app_tx,
                format!("model {model_id}"),
                HistoryEntry::Command {
                    raw: format!("model {model_id}"),
                    result: format!("active model: {model_id}"),
                },
                HistoryMeta::Command(CommandKind::Model(model_id)),
                true,
            )
            .await?;
            Ok(AppAction::Continue)
        }
        CommandDispatch::SetToolsEnabled(enabled) => {
            let raw = if enabled {
                "tools enable"
            } else {
                "tools disable"
            };
            let result = if enabled {
                "tools prompt is now enabled"
            } else {
                "tools prompt is now disabled"
            };
            apply_history_command_edit_or_insert(
                app,
                runtime,
                tools,
                app_tx,
                raw.to_string(),
                HistoryEntry::Command {
                    raw: raw.to_string(),
                    result: result.to_string(),
                },
                HistoryMeta::Command(CommandKind::Tools(enabled)),
                true,
            )
            .await?;
            Ok(AppAction::Continue)
        }
        CommandDispatch::Break => {
            let editing = app.command_edit_target.take();
            if let Some(entry_id) = editing {
                replace_history_entry(
                    active_thread_mut(app),
                    entry_id,
                    HistoryEntry::Break,
                    HistoryMeta::None,
                );
                recompute_thread_config(active_thread_mut(app));
                let thread = active_thread_mut(app);
                if let Some(index) = find_history_index(thread, entry_id) {
                    enqueue_replay_from_index(app, index, None, runtime, tools, app_tx).await?;
                }
            } else {
                let history_id = next_history_id(app);
                push_history_entry(
                    active_thread_mut(app),
                    history_id,
                    HistoryEntry::Break,
                    HistoryMeta::None,
                );
            }
            close_command_mode(app);
            Ok(AppAction::Continue)
        }
        CommandDispatch::ActivateTab(target) => {
            if let Some(index) = resolve_tab_target(&app.threads, &target) {
                activate_thread_index(app, index);
                close_command_mode(app);
            } else {
                app.command_palette.set_error(format!(
                    "unknown tab target: {}",
                    render_tab_target(&target)
                ));
            }
            Ok(AppAction::Continue)
        }
        CommandDispatch::NewTab(model_id) => {
            if runtime.config(&model_id).is_none() {
                app.command_palette.set_error(format!(
                    "{model_id} is not a recognised model, use :model ls to show available models"
                ));
                return Ok(AppAction::Continue);
            }
            let connection = runtime
                .open_connection(&model_id)
                .await
                .map_err(|err| std_io::Error::other(err.to_string()))?;
            let tools_enabled = active_thread(app).current_config.tools_enabled;
            let tab_id = app.next_tab_id;
            app.next_tab_id += 1;
            app.threads.push(ThreadState {
                tab_id,
                base_config: ThreadConfig {
                    model_id: model_id.clone(),
                    tools_enabled,
                },
                current_config: ThreadConfig {
                    model_id: model_id.clone(),
                    tools_enabled,
                },
                name_override: None,
                history: Vec::new(),
                selected_chat_entry: None,
                compose_target: ComposeTarget::AppendBottom,
                draft: String::new(),
                scroll_anchor: ScrollAnchor::Bottom,
                queue: VecDeque::new(),
                running_job: false,
                running_assistant_id: None,
                loaded_models: HashSet::new(),
                connections: HashMap::from([(model_id, connection)]),
                has_unseen_changes: false,
            });
            activate_thread_index(app, app.threads.len() - 1);
            close_command_mode(app);
            Ok(AppAction::Continue)
        }
        CommandDispatch::RenameTab { target, new_name } => {
            let trimmed = new_name.trim();
            if trimmed.is_empty() {
                app.command_palette.set_error("tab name cannot be empty");
                return Ok(AppAction::Continue);
            }
            if tab_labels(&app.threads).iter().any(|name| name == trimmed) {
                app.command_palette
                    .set_error(format!("tab name already exists: {trimmed}"));
                return Ok(AppAction::Continue);
            }
            if let Some(index) = resolve_tab_target(&app.threads, &target) {
                app.threads[index].name_override = Some(trimmed.to_string());
                close_command_mode(app);
            } else {
                app.command_palette.set_error(format!(
                    "unknown tab target: {}",
                    render_tab_target(&target)
                ));
            }
            Ok(AppAction::Continue)
        }
        CommandDispatch::KillTab(target) => {
            if app.threads.len() == 1 {
                app.command_palette
                    .set_error("cannot kill the final remaining tab");
                return Ok(AppAction::Continue);
            }
            if let Some(index) = resolve_tab_target(&app.threads, &target) {
                let mut thread = app.threads.remove(index);
                close_thread_connections(&mut thread).await;
                let next_index = app
                    .active_thread_idx
                    .min(app.threads.len().saturating_sub(1));
                activate_thread_index(app, next_index);
                close_command_mode(app);
            } else {
                app.command_palette.set_error(format!(
                    "unknown tab target: {}",
                    render_tab_target(&target)
                ));
            }
            Ok(AppAction::Continue)
        }
    }
}

async fn apply_history_command_edit_or_insert(
    app: &mut AppState,
    runtime: &runtime::Runtime,
    tools: &[ToolSpec],
    app_tx: mpsc::UnboundedSender<AppEvent>,
    raw: String,
    entry: HistoryEntry,
    meta: HistoryMeta,
    replay_affecting: bool,
) -> std_io::Result<()> {
    let editing = app.command_edit_target.take();
    if let Some(entry_id) = editing {
        let thread = active_thread_mut(app);
        replace_history_entry(thread, entry_id, entry, meta.clone());
        recompute_thread_config(thread);
        let index = find_history_index(thread, entry_id);
        if replay_affecting {
            if let Some(index) = index {
                replay_edited_command(app, index, meta, runtime, tools, app_tx).await?;
            }
        }
    } else {
        let history_id = next_history_id(app);
        let thread = active_thread_mut(app);
        push_history_entry(thread, history_id, entry, meta);
        recompute_thread_config(thread);
    }
    if raw.is_empty() {
        let _ = raw;
    }
    close_command_mode(app);
    Ok(())
}

async fn replay_edited_command(
    app: &mut AppState,
    command_index: usize,
    meta: HistoryMeta,
    runtime: &runtime::Runtime,
    tools: &[ToolSpec],
    app_tx: mpsc::UnboundedSender<AppEvent>,
) -> std_io::Result<()> {
    let HistoryMeta::Command(new_kind) = meta else {
        return Ok(());
    };

    let mut replay_end = None;
    let mut compacted = false;

    {
        let thread = active_thread_mut(app);
        if let Some(family) = new_kind.family() {
            let next_same = thread
                .history
                .iter()
                .enumerate()
                .skip(command_index + 1)
                .find_map(|(index, item)| match &item.meta {
                    HistoryMeta::Command(other) if other.family() == Some(family) => {
                        Some((index, other.clone()))
                    }
                    _ => None,
                });

            if let Some((index, other_kind)) = next_same {
                if new_kind.matches_semantics(&other_kind) {
                    thread.history.remove(index);
                    compacted = true;
                } else {
                    replay_end = Some(index);
                }
            }
        }
        recompute_thread_config(thread);
    }

    let _ = compacted;
    enqueue_replay_from_index(app, command_index + 1, replay_end, runtime, tools, app_tx).await
}

async fn delete_history_entry(
    app: &mut AppState,
    target_id: u64,
    runtime: &runtime::Runtime,
    tools: &[ToolSpec],
    app_tx: mpsc::UnboundedSender<AppEvent>,
) -> std_io::Result<()> {
    let replay_start_index = {
        let thread = active_thread_mut(app);
        let Some(index) = find_history_index(thread, target_id) else {
            return Ok(());
        };

        if matches!(
            thread.history.get(index + 1).map(|item| &item.entry),
            Some(HistoryEntry::LoadingModel { .. })
        ) {
            thread.history.remove(index + 1);
        }

        let replay_start_index = match thread.history[index].entry {
            HistoryEntry::Assistant { .. } => Some(index),
            HistoryEntry::Command { .. } | HistoryEntry::Break => Some(index),
            HistoryEntry::LoadingModel { .. } | HistoryEntry::SystemNotice(_) => None,
        };

        thread.history.remove(index);
        recompute_thread_config(thread);
        replay_start_index
    };

    if let Some(start_index) = replay_start_index {
        enqueue_replay_from_index(app, start_index, None, runtime, tools, app_tx).await?;
    }
    Ok(())
}

async fn submit_chat_turn(
    app: &mut AppState,
    input: String,
    runtime: &runtime::Runtime,
    tools: &[ToolSpec],
    app_tx: mpsc::UnboundedSender<AppEvent>,
) -> std_io::Result<()> {
    let compose_target = active_thread(app).compose_target;
    match compose_target {
        ComposeTarget::EditEntry(entry_id) => {
            let thread = active_thread_mut(app);
            if let Some(index) = find_history_index(thread, entry_id) {
                if let HistoryEntry::Assistant {
                    prompt,
                    blocks,
                    callouts,
                    status,
                    sequence_number,
                    ..
                } = &mut thread.history[index].entry
                {
                    *prompt = input;
                    blocks.clear();
                    callouts.clear();
                    *status = Some(AssistantStatus::Queued {
                        started_at: Instant::now(),
                    });
                    *sequence_number = None;
                }
                thread.selected_chat_entry = Some(index);
                thread.queue.clear();
            }
            if let Some(index) = active_thread(app)
                .history
                .iter()
                .position(|item| item.id == entry_id)
            {
                enqueue_replay_from_index(app, index, None, runtime, tools, app_tx).await?;
            }
        }
        ComposeTarget::AppendBottom => {
            let assistant_id = next_history_id(app);
            let model_id = active_thread(app).current_config.model_id.clone();
            {
                let thread = active_thread_mut(app);
                insert_pending_assistant_turn(thread, assistant_id, model_id, input);
                thread.selected_chat_entry = None;
            }
            queue_generation_for_turn(app, assistant_id, runtime, tools, app_tx).await?;
        }
    }

    active_thread_mut(app).compose_target = ComposeTarget::AppendBottom;
    app.mode = Mode::Insert;
    Ok(())
}

async fn queue_generation_for_turn(
    app: &mut AppState,
    assistant_id: u64,
    runtime: &runtime::Runtime,
    tools: &[ToolSpec],
    app_tx: mpsc::UnboundedSender<AppEvent>,
) -> std_io::Result<()> {
    let (assistant_index, model_id, needs_load) = {
        let thread = active_thread(app);
        let Some(index) = find_history_index(thread, assistant_id) else {
            return Ok(());
        };
        let model_id = match &thread.history[index].entry {
            HistoryEntry::Assistant { model_id, .. } => model_id.clone(),
            _ => return Ok(()),
        };
        let needs_load = !thread.loaded_models.contains(&model_id);
        (index, model_id, needs_load)
    };

    let loading_id = next_history_id(app);
    {
        let thread = active_thread_mut(app);
        if needs_load
            && !matches!(
                thread.history.get(assistant_index.saturating_sub(1)).map(|item| &item.entry),
                Some(HistoryEntry::LoadingModel { model_id: prev_model, .. }) if prev_model == &model_id
            )
        {
            maybe_insert_loading_entry(thread, assistant_index, loading_id);
        }
        if needs_load {
            set_loading_status(
                thread,
                &model_id,
                ModelLoadStatus::Loading {
                    started_at: Instant::now(),
                },
            );
        }
        thread.queue.push_back(ThreadJob {
            assistant_entry_id: assistant_id,
        });
    }
    maybe_start_next_job(app, runtime, tools, app_tx).await
}

async fn enqueue_replay_from_index(
    app: &mut AppState,
    start_index: usize,
    replay_end: Option<usize>,
    runtime: &runtime::Runtime,
    tools: &[ToolSpec],
    app_tx: mpsc::UnboundedSender<AppEvent>,
) -> std_io::Result<()> {
    let effective_end = {
        let thread = active_thread(app);
        replay_end.unwrap_or(thread.history.len())
    };
    let assistant_jobs = {
        let thread = active_thread(app);
        (start_index..effective_end)
            .filter_map(|index| {
                thread
                    .history
                    .get(index)
                    .and_then(|item| match &item.entry {
                        HistoryEntry::Assistant { model_id, .. } => {
                            Some((index, item.id, model_id.clone()))
                        }
                        _ => None,
                    })
            })
            .collect::<Vec<_>>()
    };

    let mut loading_insertions = Vec::new();
    {
        let thread = active_thread_mut(app);
        thread.queue.clear();
        for (index, assistant_id, model_id) in &assistant_jobs {
            if !thread.loaded_models.contains(model_id)
                && !matches!(
                    thread.history.get(index.saturating_sub(1)).map(|item| &item.entry),
                    Some(HistoryEntry::LoadingModel { model_id: prev_model, .. }) if prev_model == model_id
                )
            {
                loading_insertions.push((*index, model_id.clone()));
            }
            if let Some(item) = thread.history.get_mut(*index) {
                if let HistoryEntry::Assistant {
                    blocks,
                    callouts,
                    status,
                    sequence_number,
                    ..
                } = &mut item.entry
                {
                    blocks.clear();
                    callouts.clear();
                    *status = Some(AssistantStatus::Queued {
                        started_at: Instant::now(),
                    });
                    *sequence_number = None;
                }
            }
            thread.queue.push_back(ThreadJob {
                assistant_entry_id: *assistant_id,
            });
        }
    }

    let mut loading_insertions = loading_insertions
        .into_iter()
        .map(|(index, model_id)| (index, model_id, next_history_id(app)))
        .collect::<Vec<_>>();
    loading_insertions.sort_by(|left, right| right.0.cmp(&left.0));
    {
        let thread = active_thread_mut(app);
        for (index, model_id, loading_id) in loading_insertions {
            maybe_insert_loading_entry(thread, index, loading_id);
            set_loading_status(
                thread,
                &model_id,
                ModelLoadStatus::Loading {
                    started_at: Instant::now(),
                },
            );
        }
    }

    maybe_start_next_job(app, runtime, tools, app_tx).await
}

async fn maybe_start_next_job(
    app: &mut AppState,
    runtime: &runtime::Runtime,
    tools: &[ToolSpec],
    app_tx: mpsc::UnboundedSender<AppEvent>,
) -> std_io::Result<()> {
    let active_tab_id = active_thread(app).tab_id;
    let Some(thread_index) = thread_index_by_tab_id(app, active_tab_id) else {
        return Ok(());
    };
    start_next_job_for_thread(app, thread_index, runtime, tools, app_tx).await
}

async fn start_next_job_for_thread(
    app: &mut AppState,
    thread_index: usize,
    runtime: &runtime::Runtime,
    tools: &[ToolSpec],
    app_tx: mpsc::UnboundedSender<AppEvent>,
) -> std_io::Result<()> {
    let Some(job) = app.threads[thread_index].queue.pop_front() else {
        return Ok(());
    };
    if app.threads[thread_index].running_job {
        app.threads[thread_index].queue.push_front(job);
        return Ok(());
    }

    let tab_id = app.threads[thread_index].tab_id;
    let (request, config, tool_format) =
        build_request_for_turn(&app.threads[thread_index], job.assistant_entry_id, tools)?;
    ensure_thread_connection(&mut app.threads[thread_index], runtime, &config.model_id).await?;
    let connection = app.threads[thread_index]
        .connections
        .get(&config.model_id)
        .expect("connection should exist")
        .clone();
    if !app.threads[thread_index]
        .loaded_models
        .contains(&config.model_id)
    {
        set_loading_status(
            &mut app.threads[thread_index],
            &config.model_id,
            ModelLoadStatus::Loading {
                started_at: Instant::now(),
            },
        );
    }

    let tools = tools.to_vec();
    let (status_tx, mut status_rx) = mpsc::unbounded_channel();
    let mut stream = match connection
        .stream_with_status(request, Some(status_tx))
        .await
    {
        Ok(stream) => stream,
        Err(err) => {
            let _ = app_tx.send(AppEvent::Job(JobEvent::Error {
                tab_id,
                assistant_entry_id: job.assistant_entry_id,
                error: format!("runtime error: {err}"),
            }));
            app.threads[thread_index].running_job = false;
            app.threads[thread_index].running_assistant_id = None;
            app.threads[thread_index].queue.push_front(job);
            return Ok(());
        }
    };
    app.threads[thread_index].running_job = true;
    app.threads[thread_index].running_assistant_id = Some(job.assistant_entry_id);
    let sequence_number = stream.sequence_number();
    if let Some(index) = find_history_index(&app.threads[thread_index], job.assistant_entry_id)
        && let HistoryEntry::Assistant {
            sequence_number: entry_sequence,
            ..
        } = &mut app.threads[thread_index].history[index].entry
    {
        *entry_sequence = Some(sequence_number);
    }

    tokio::spawn(async move {
        let mut markdown = MarkdownAccumulator::new(tool_format);
        loop {
            tokio::select! {
                status = status_rx.recv() => {
                    if let Some(status) = status {
                        let _ = app_tx.send(AppEvent::Job(JobEvent::Status {
                            tab_id,
                            assistant_entry_id: job.assistant_entry_id,
                            status,
                        }));
                    }
                }
                event = stream.next() => {
                    match event {
                        Some(RuntimeResponseEvent::Chunk { content, .. }) => {
                            let parsed = markdown.append(&content);
                            if parsed.tool_call_detected {
                                tracing::info!(
                                    tab_id,
                                    assistant_entry_id = job.assistant_entry_id,
                                    "tool called"
                                );
                            }
                            let _ = app_tx.send(AppEvent::Job(JobEvent::Chunk {
                                tab_id,
                                assistant_entry_id: job.assistant_entry_id,
                                markdown: parsed,
                            }));
                        }
                        Some(RuntimeResponseEvent::Complete { .. }) => {
                            let parsed = markdown.finalize();
                            let callouts = if config.tools_enabled {
                                parsed_tool_callouts(
                                    tool_format.parse(&blocks_to_raw_text(&parsed.blocks)),
                                    &tools,
                                )
                            } else {
                                Vec::new()
                            };
                            let _ = app_tx.send(AppEvent::Job(JobEvent::Complete {
                                tab_id,
                                assistant_entry_id: job.assistant_entry_id,
                                markdown: parsed,
                                callouts,
                            }));
                            break;
                        }
                        Some(RuntimeResponseEvent::Interrupted { .. }) => {
                            let parsed = markdown.finalize();
                            let _ = app_tx.send(AppEvent::Job(JobEvent::Interrupted {
                                tab_id,
                                assistant_entry_id: job.assistant_entry_id,
                                markdown: parsed,
                            }));
                            break;
                        }
                        Some(RuntimeResponseEvent::Error { error, .. }) => {
                            let _ = app_tx.send(AppEvent::Job(JobEvent::Error {
                                tab_id,
                                assistant_entry_id: job.assistant_entry_id,
                                error: format!("runtime error: {error}"),
                            }));
                            break;
                        }
                        None => {
                            let _ = app_tx.send(AppEvent::Job(JobEvent::Error {
                                tab_id,
                                assistant_entry_id: job.assistant_entry_id,
                                error: String::from("runtime stream closed"),
                            }));
                            break;
                        }
                    }
                }
            }
        }
    });

    Ok(())
}

async fn handle_job_event(
    event: JobEvent,
    app: &mut AppState,
    runtime: &runtime::Runtime,
    tools: &[ToolSpec],
    app_tx: mpsc::UnboundedSender<AppEvent>,
) {
    let Some(thread_index) = thread_index_by_tab_id(app, event.tab_id()) else {
        return;
    };
    match event {
        JobEvent::Status {
            assistant_entry_id,
            status,
            ..
        } => {
            let thread = &mut app.threads[thread_index];
            mark_thread_changed(thread, thread_index != app.active_thread_idx);
            let model_id = match thread
                .history
                .iter()
                .find(|item| item.id == assistant_entry_id)
                .and_then(|item| match &item.entry {
                    HistoryEntry::Assistant { model_id, .. } => Some(model_id.clone()),
                    _ => None,
                }) {
                Some(model_id) => model_id,
                None => return,
            };
            match status {
                RuntimeStatus::Queued { .. } => {
                    set_assistant_status(
                        thread,
                        assistant_entry_id,
                        AssistantStatus::Queued {
                            started_at: Instant::now(),
                        },
                    );
                }
                RuntimeStatus::Loading { .. } => {
                    set_loading_status(
                        thread,
                        &model_id,
                        ModelLoadStatus::Loading {
                            started_at: Instant::now(),
                        },
                    );
                    set_assistant_status(
                        thread,
                        assistant_entry_id,
                        AssistantStatus::Loading {
                            started_at: Instant::now(),
                        },
                    );
                }
                RuntimeStatus::Generating { .. } => {
                    thread.loaded_models.insert(model_id.clone());
                    set_loading_status(thread, &model_id, ModelLoadStatus::Loaded);
                    set_assistant_status(
                        thread,
                        assistant_entry_id,
                        AssistantStatus::Generating {
                            started_at: Instant::now(),
                        },
                    );
                }
            }
        }
        JobEvent::Chunk {
            assistant_entry_id,
            markdown,
            ..
        } => {
            mark_thread_changed(
                &mut app.threads[thread_index],
                thread_index != app.active_thread_idx,
            );
            replace_assistant_blocks(
                &mut app.threads[thread_index],
                assistant_entry_id,
                markdown.blocks,
            );
            set_assistant_tool_callout(
                &mut app.threads[thread_index],
                assistant_entry_id,
                markdown.tool_call_detected,
            );
        }
        JobEvent::Complete {
            assistant_entry_id,
            markdown,
            callouts,
            ..
        } => {
            mark_thread_changed(
                &mut app.threads[thread_index],
                thread_index != app.active_thread_idx,
            );
            finalize_assistant(
                &mut app.threads[thread_index],
                assistant_entry_id,
                markdown.blocks,
                merge_tool_call_callout(markdown.tool_call_detected, callouts),
            );
            app.threads[thread_index].running_job = false;
            app.threads[thread_index].running_assistant_id = None;
            let _ = start_next_job_for_thread(app, thread_index, runtime, tools, app_tx).await;
        }
        JobEvent::Interrupted {
            assistant_entry_id,
            markdown,
            ..
        } => {
            mark_thread_changed(
                &mut app.threads[thread_index],
                thread_index != app.active_thread_idx,
            );
            finalize_interrupted_assistant(
                &mut app.threads[thread_index],
                assistant_entry_id,
                markdown.blocks,
            );
            set_assistant_tool_callout(
                &mut app.threads[thread_index],
                assistant_entry_id,
                markdown.tool_call_detected,
            );
            app.threads[thread_index].running_job = false;
            app.threads[thread_index].running_assistant_id = None;
            let _ = start_next_job_for_thread(app, thread_index, runtime, tools, app_tx).await;
        }
        JobEvent::Error {
            assistant_entry_id,
            error,
            ..
        } => {
            finalize_assistant(
                &mut app.threads[thread_index],
                assistant_entry_id,
                Vec::new(),
                Vec::new(),
            );
            let notice_id = next_history_id(app);
            push_history_entry(
                &mut app.threads[thread_index],
                notice_id,
                HistoryEntry::SystemNotice(error),
                HistoryMeta::None,
            );
            app.threads[thread_index].running_job = false;
            app.threads[thread_index].running_assistant_id = None;
            let _ = start_next_job_for_thread(app, thread_index, runtime, tools, app_tx).await;
        }
    }
}

fn redraw(ui: &mut TerminalUi, app: &AppState) -> std_io::Result<()> {
    let thread = active_thread(app);
    ui.draw(RenderState {
        entries: &history_entries(thread),
        selected_chat_entry: thread.selected_chat_entry,
        mode: app.mode,
        draft: &thread.draft,
        prompt_inline: matches!(app.mode, Mode::Insert)
            && matches!(thread.compose_target, ComposeTarget::EditEntry(_)),
        scroll_anchor: thread.scroll_anchor,
        command_palette: matches!(app.mode, Mode::Command).then(|| app.command_palette.view()),
        tabs: &tab_render_info(&app.threads),
        active_tab: app.active_thread_idx,
        delete_confirmation: matches!(app.mode, Mode::ConfirmDelete)
            .then_some("Confirmation: delete message [y/n]"),
    })
}

fn build_request_for_turn(
    thread: &ThreadState,
    assistant_entry_id: u64,
    tools: &[ToolSpec],
) -> std_io::Result<(RequestBuilder, ThreadConfig, agent::ToolFormat)> {
    let Some(target_index) = find_history_index(thread, assistant_entry_id) else {
        return Err(std_io::Error::other("target assistant vanished"));
    };
    let effective_config = effective_config_until(thread, target_index);

    let config = effective_config.clone();
    let registry = runtime::RuntimeRegistry::defaults();
    let model_config = registry
        .get(&config.model_id)
        .ok_or_else(|| std_io::Error::other("unknown runtime config"))?;
    let tool_format = model_config.model.tool_format();
    let system_prompt = tool_format.system_prompt(
        SYSTEM_PROMPT,
        if config.tools_enabled { tools } else { &[] },
    );
    let mut messages =
        TextMessages::new().add_message(TextMessageRole::System, system_prompt.clone());

    let mut start_index = 0usize;
    for (index, item) in thread.history.iter().enumerate().take(target_index + 1) {
        if matches!(item.entry, HistoryEntry::Break) {
            start_index = index + 1;
        }
    }

    for item in thread
        .history
        .iter()
        .skip(start_index)
        .take(target_index + 1 - start_index)
    {
        match &item.entry {
            HistoryEntry::Assistant {
                prompt,
                blocks,
                status,
                ..
            } => {
                messages = messages.add_message(TextMessageRole::User, prompt);
                if status.is_none() {
                    messages = messages
                        .add_message(TextMessageRole::Assistant, blocks_to_raw_text(blocks));
                }
            }
            HistoryEntry::LoadingModel { .. } => {}
            HistoryEntry::Command { .. } | HistoryEntry::Break | HistoryEntry::SystemNotice(_) => {}
        }
    }

    tracing::info!(
        max_generation_tokens = MAX_GENERATION_TOKENS,
        message_count = messages.messages_ref().len(),
        model = %config.model_id,
        tools_enabled = config.tools_enabled,
        "building chat generation request"
    );

    Ok((
        RequestBuilder::from(messages).set_sampler_max_len(MAX_GENERATION_TOKENS),
        config,
        tool_format,
    ))
}

fn effective_config_until(thread: &ThreadState, target_index: usize) -> ThreadConfig {
    let mut config = thread.base_config.clone();
    for item in thread.history.iter().take(target_index + 1) {
        if let HistoryMeta::Command(kind) = &item.meta {
            match kind {
                CommandKind::Model(model_id) => config.model_id = model_id.clone(),
                CommandKind::Tools(enabled) => config.tools_enabled = *enabled,
                CommandKind::ModelList => {}
            }
        }
    }
    config
}

fn recompute_thread_config(thread: &mut ThreadState) {
    let mut config = thread.base_config.clone();
    for item in &thread.history {
        if let HistoryMeta::Command(kind) = &item.meta {
            match kind {
                CommandKind::Model(model_id) => config.model_id = model_id.clone(),
                CommandKind::Tools(enabled) => config.tools_enabled = *enabled,
                CommandKind::ModelList => {}
            }
        }
    }
    thread.current_config = config;
}

async fn ensure_thread_connection(
    thread: &mut ThreadState,
    runtime: &runtime::Runtime,
    model_id: &str,
) -> std_io::Result<()> {
    if thread.connections.contains_key(model_id) {
        return Ok(());
    }
    let connection = runtime
        .open_connection(model_id)
        .await
        .map_err(|err| std_io::Error::other(err.to_string()))?;
    thread.connections.insert(model_id.to_string(), connection);
    Ok(())
}

fn parsed_tool_callouts(tool_calls: Vec<agent::ToolCall>, tools: &[ToolSpec]) -> Vec<String> {
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
        return Vec::new();
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
    callouts
}

fn render_tab_target(target: &TabTarget) -> String {
    match target {
        TabTarget::Id(id) => id.to_string(),
        TabTarget::Name(name) => name.clone(),
    }
}

fn resolve_tab_target(threads: &[ThreadState], target: &TabTarget) -> Option<usize> {
    match target {
        TabTarget::Id(id) => threads.iter().position(|thread| thread.tab_id == *id),
        TabTarget::Name(name) => tab_labels(threads).iter().position(|label| label == name),
    }
}

fn tab_render_info(threads: &[ThreadState]) -> Vec<TabRenderInfo> {
    threads
        .iter()
        .zip(tab_labels(threads))
        .map(|(thread, label)| TabRenderInfo {
            label: format!("{}. {}", thread.tab_id, label),
            has_unseen: thread.has_unseen_changes,
        })
        .collect()
}

fn tab_labels(threads: &[ThreadState]) -> Vec<String> {
    let mut counts = HashMap::<String, usize>::new();
    for thread in threads
        .iter()
        .filter(|thread| thread.name_override.is_none())
    {
        *counts
            .entry(thread.current_config.model_id.clone())
            .or_default() += 1;
    }
    let mut seen = HashMap::<String, usize>::new();
    threads
        .iter()
        .map(|thread| {
            if let Some(name) = &thread.name_override {
                return name.clone();
            }
            let model_id = thread.current_config.model_id.clone();
            if counts.get(&model_id).copied().unwrap_or(0) <= 1 {
                model_id
            } else {
                let index = seen.entry(model_id.clone()).or_default();
                *index += 1;
                format!("{model_id}-{}", *index)
            }
        })
        .collect()
}

fn active_thread(app: &AppState) -> &ThreadState {
    &app.threads[app.active_thread_idx]
}

fn active_thread_mut(app: &mut AppState) -> &mut ThreadState {
    &mut app.threads[app.active_thread_idx]
}

fn history_entries(thread: &ThreadState) -> Vec<HistoryEntry> {
    thread
        .history
        .iter()
        .map(|item| item.entry.clone())
        .collect()
}

fn next_history_id(app: &mut AppState) -> u64 {
    let id = app.next_history_id;
    app.next_history_id += 1;
    id
}

fn push_history_entry(thread: &mut ThreadState, id: u64, entry: HistoryEntry, meta: HistoryMeta) {
    thread.history.push(HistoryItem { id, entry, meta });
}

fn replace_history_entry(
    thread: &mut ThreadState,
    entry_id: u64,
    entry: HistoryEntry,
    meta: HistoryMeta,
) {
    if let Some(index) = find_history_index(thread, entry_id) {
        thread.history[index].entry = entry;
        thread.history[index].meta = meta;
    }
}

fn find_history_index(thread: &ThreadState, entry_id: u64) -> Option<usize> {
    thread.history.iter().position(|item| item.id == entry_id)
}

fn insert_pending_assistant_turn(
    thread: &mut ThreadState,
    assistant_id: u64,
    model_id: String,
    prompt: String,
) {
    insert_assistant_turn_at(
        thread,
        thread.history.len(),
        assistant_id,
        model_id,
        prompt,
        Some(AssistantStatus::Queued {
            started_at: Instant::now(),
        }),
    );
}

fn insert_assistant_turn_at(
    thread: &mut ThreadState,
    index: usize,
    assistant_id: u64,
    model_id: String,
    prompt: String,
    status: Option<AssistantStatus>,
) {
    thread.history.insert(
        index,
        HistoryItem {
            id: assistant_id,
            entry: HistoryEntry::Assistant {
                model_id,
                prompt,
                blocks: Vec::new(),
                callouts: Vec::new(),
                status,
                sequence_number: None,
            },
            meta: HistoryMeta::None,
        },
    );
}

fn maybe_insert_loading_entry(thread: &mut ThreadState, assistant_index: usize, loading_id: u64) {
    let Some(HistoryEntry::Assistant { model_id, .. }) =
        thread.history.get(assistant_index).map(|item| &item.entry)
    else {
        return;
    };

    if thread.loaded_models.contains(model_id) {
        return;
    }

    if assistant_index > 0
        && matches!(
            thread.history.get(assistant_index - 1).map(|item| &item.entry),
            Some(HistoryEntry::LoadingModel { model_id: prev_model, .. }) if prev_model == model_id
        )
    {
        return;
    }

    thread.history.insert(
        assistant_index,
        HistoryItem {
            id: loading_id,
            entry: HistoryEntry::LoadingModel {
                model_id: model_id.clone(),
                status: Some(ModelLoadStatus::Loading {
                    started_at: Instant::now(),
                }),
            },
            meta: HistoryMeta::None,
        },
    );
}

fn set_loading_status(thread: &mut ThreadState, model_id: &str, status: ModelLoadStatus) {
    if let Some(item) = thread.history.iter_mut().rev().find(|item| {
        matches!(
            item.entry,
            HistoryEntry::LoadingModel {
                model_id: ref entry_model_id,
                ..
            } if entry_model_id == model_id
        )
    }) && let HistoryEntry::LoadingModel {
        status: entry_status,
        ..
    } = &mut item.entry
    {
        *entry_status = Some(status);
    }
}

fn set_assistant_status(thread: &mut ThreadState, assistant_id: u64, status: AssistantStatus) {
    if let Some(index) = find_history_index(thread, assistant_id) {
        if let HistoryEntry::Assistant {
            status: entry_status,
            ..
        } = &mut thread.history[index].entry
        {
            *entry_status = Some(status);
        }
    }
}

fn replace_assistant_blocks(
    thread: &mut ThreadState,
    assistant_id: u64,
    blocks: Vec<MarkdownBlock>,
) {
    if let Some(index) = find_history_index(thread, assistant_id) {
        if let HistoryEntry::Assistant {
            blocks: assistant_blocks,
            status,
            sequence_number,
            ..
        } = &mut thread.history[index].entry
        {
            *status = None;
            *sequence_number = None;
            *assistant_blocks = blocks;
        }
    }
}

fn finalize_assistant(
    thread: &mut ThreadState,
    assistant_id: u64,
    blocks: Vec<MarkdownBlock>,
    callouts: Vec<String>,
) {
    if let Some(index) = find_history_index(thread, assistant_id) {
        if let HistoryEntry::Assistant {
            blocks: entry_blocks,
            callouts: entry_callouts,
            status,
            sequence_number,
            ..
        } = &mut thread.history[index].entry
        {
            *entry_blocks = blocks;
            *status = None;
            *sequence_number = None;
            entry_callouts.extend(callouts);
        }
    }
}

fn finalize_interrupted_assistant(
    thread: &mut ThreadState,
    assistant_id: u64,
    blocks: Vec<MarkdownBlock>,
) {
    if blocks.is_empty() {
        if let Some(index) = find_history_index(thread, assistant_id) {
            thread.history.remove(index);
        }
        return;
    }

    finalize_assistant(thread, assistant_id, blocks, Vec::new());
}

fn set_assistant_tool_callout(thread: &mut ThreadState, assistant_id: u64, detected: bool) {
    if let Some(index) = find_history_index(thread, assistant_id)
        && let HistoryEntry::Assistant { callouts, .. } = &mut thread.history[index].entry
    {
        callouts.retain(|line| line != "tool called");
        if detected {
            callouts.insert(0, String::from("tool called"));
        }
    }
}

fn merge_tool_call_callout(detected: bool, mut callouts: Vec<String>) -> Vec<String> {
    if detected {
        callouts.insert(0, String::from("tool called"));
    }
    callouts
}

fn normalize_selection(thread: &mut ThreadState) {
    sync_loaded_model_entries(thread);
    let positions = chat_entry_positions(&history_entries(thread));
    match (thread.selected_chat_entry, positions.len()) {
        (_, 0) => thread.selected_chat_entry = None,
        (Some(selected), len) if selected >= len => thread.selected_chat_entry = Some(len - 1),
        (None, len) if len > 0 && !matches!(thread.compose_target, ComposeTarget::AppendBottom) => {
            thread.selected_chat_entry = Some(len - 1)
        }
        _ => {}
    }
}

fn sync_loaded_model_entries(thread: &mut ThreadState) {
    let loaded_models = thread.loaded_models.clone();
    for item in &mut thread.history {
        if let HistoryEntry::LoadingModel { model_id, status } = &mut item.entry
            && loaded_models.contains(model_id)
        {
            *status = Some(ModelLoadStatus::Loaded);
        }
    }
}

fn move_selection(thread: &ThreadState, delta: isize) -> Option<usize> {
    let chat_positions = chat_entry_positions(&history_entries(thread));
    if chat_positions.is_empty() {
        return None;
    }
    let current = thread
        .selected_chat_entry
        .unwrap_or(chat_positions.len() - 1) as isize;
    let next = (current + delta).clamp(0, chat_positions.len().saturating_sub(1) as isize);
    Some(next as usize)
}

fn exit_insert_mode(thread: &mut ThreadState) {
    thread.draft.clear();
    thread.compose_target = ComposeTarget::AppendBottom;
    thread.selected_chat_entry = move_selection(thread, 0);
}

fn open_footer_compose(app: &mut AppState) {
    let thread = active_thread_mut(app);
    thread.selected_chat_entry = None;
    thread.compose_target = ComposeTarget::AppendBottom;
    thread.draft.clear();
    app.mode = Mode::Insert;
}

fn cycle_active_tab(app: &mut AppState) {
    if app.threads.is_empty() {
        return;
    }
    cycle_active_tab_relative(app, 1);
}

fn cycle_active_tab_relative(app: &mut AppState, delta: isize) {
    if app.threads.is_empty() {
        return;
    }

    let len = app.threads.len() as isize;
    let current = app.active_thread_idx as isize;
    let next = (current + delta).rem_euclid(len) as usize;
    activate_thread_index(app, next);
}

fn activate_thread_index(app: &mut AppState, index: usize) {
    app.active_thread_idx = index.min(app.threads.len().saturating_sub(1));
    if let Some(thread) = app.threads.get_mut(app.active_thread_idx) {
        thread.has_unseen_changes = false;
    }
}

fn mark_thread_changed(thread: &mut ThreadState, unseen: bool) {
    if unseen {
        thread.has_unseen_changes = true;
    }
}

fn selected_interrupt_target(app: &AppState) -> Option<(usize, u64, Option<u64>)> {
    let thread = active_thread(app);
    if let Some(selected) = thread.selected_chat_entry
        && let Some(history_index) = chat_entry_positions(&history_entries(thread)).get(selected)
        && let Some(HistoryItem {
            id,
            entry:
                HistoryEntry::Assistant {
                    status,
                    sequence_number,
                    ..
                },
            ..
        }) = thread.history.get(*history_index)
        && status.is_some()
    {
        return Some((app.active_thread_idx, *id, *sequence_number));
    }

    thread.running_assistant_id.map(|assistant_id| {
        let sequence_number = thread
            .history
            .iter()
            .find(|item| item.id == assistant_id)
            .and_then(|item| match &item.entry {
                HistoryEntry::Assistant {
                    sequence_number, ..
                } => *sequence_number,
                _ => None,
            });
        (app.active_thread_idx, assistant_id, sequence_number)
    })
}

fn cancel_queued_assistant(app: &mut AppState, thread_index: usize, assistant_id: u64) {
    let thread = &mut app.threads[thread_index];
    thread
        .queue
        .retain(|job| job.assistant_entry_id != assistant_id);
    if let Some(index) = find_history_index(thread, assistant_id) {
        thread.history.remove(index);
    }
}

fn close_command_mode(app: &mut AppState) {
    app.command_palette.close();
    app.command_edit_target = None;
    app.inline_command_placeholder = None;
    open_footer_compose(app);
}

fn open_inline_compose_relative(app: &mut AppState, position: InlineComposePosition) {
    let Some(selected_index) = active_thread(app).selected_chat_entry else {
        open_footer_compose(app);
        return;
    };

    let history_len = active_thread(app).history.len();
    let insert_index = match position {
        InlineComposePosition::Before => selected_index,
        InlineComposePosition::After => selected_index + 1,
    };

    if matches!(position, InlineComposePosition::After) && insert_index >= history_len {
        open_footer_compose(app);
        return;
    }

    let assistant_id = next_history_id(app);
    let model_id = active_thread(app).current_config.model_id.clone();
    let thread = active_thread_mut(app);
    insert_assistant_turn_at(
        thread,
        insert_index,
        assistant_id,
        model_id,
        String::new(),
        None,
    );
    thread.selected_chat_entry = Some(insert_index);
    thread.compose_target = ComposeTarget::EditEntry(assistant_id);
    thread.draft.clear();
    app.mode = Mode::Insert;
}

fn open_inline_command_relative(app: &mut AppState, position: InlineCommandPosition) {
    let Some(selected_index) = active_thread(app).selected_chat_entry else {
        app.command_edit_target = None;
        app.mode = Mode::Command;
        app.command_palette.open();
        return;
    };

    let history_len = active_thread(app).history.len();
    let insert_index = match position {
        InlineCommandPosition::Before => selected_index,
        InlineCommandPosition::After => selected_index + 1,
    };

    if matches!(position, InlineCommandPosition::After) && insert_index > history_len {
        app.mode = Mode::Command;
        app.command_palette.open();
        return;
    }

    let command_id = next_history_id(app);
    let thread = active_thread_mut(app);
    thread.history.insert(
        insert_index,
        HistoryItem {
            id: command_id,
            entry: HistoryEntry::Command {
                raw: String::new(),
                result: String::new(),
            },
            meta: HistoryMeta::None,
        },
    );
    thread.selected_chat_entry = Some(insert_index);
    app.command_edit_target = Some(command_id);
    app.inline_command_placeholder = Some((command_id, selected_index));
    app.mode = Mode::Command;
    app.command_palette.open();
}

fn cancel_inline_command_insert(app: &mut AppState) {
    let Some((placeholder_id, anchor_index)) = app.inline_command_placeholder.take() else {
        return;
    };

    let thread = active_thread_mut(app);
    if let Some(index) = find_history_index(thread, placeholder_id) {
        thread.history.remove(index);
        recompute_thread_config(thread);
        let len = chat_entry_positions(&history_entries(thread)).len();
        thread.selected_chat_entry = if len == 0 {
            None
        } else {
            Some(anchor_index.min(len - 1))
        };
    }
}

fn push_command_error(command_palette: &mut CommandPaletteState, err: CommandParseError) {
    match err {
        CommandParseError::Empty => {}
        CommandParseError::Incomplete(message) | CommandParseError::Invalid(message) => {
            command_palette.set_error(message);
        }
    }
}

fn push_notice_current(app: &mut AppState, message: String) {
    let id = next_history_id(app);
    push_history_entry(
        active_thread_mut(app),
        id,
        HistoryEntry::SystemNotice(message),
        HistoryMeta::None,
    );
}

fn selection_edit_action(entry: &HistoryEntry) -> Option<SelectionEditAction> {
    match entry {
        HistoryEntry::Assistant { prompt, .. } => Some(SelectionEditAction::Prompt(prompt.clone())),
        HistoryEntry::Command { raw, .. } => Some(SelectionEditAction::Command(raw.clone())),
        HistoryEntry::Break => Some(SelectionEditAction::Command(String::from("break"))),
        HistoryEntry::LoadingModel { .. } => None,
        HistoryEntry::SystemNotice(_) => None,
    }
}

fn handle_mouse_event(kind: MouseEventKind, app: &mut AppState) {
    if matches!(app.mode, Mode::Command | Mode::ConfirmDelete) {
        return;
    }
    let thread = active_thread_mut(app);
    let delta = match kind {
        MouseEventKind::ScrollUp => Some(-1),
        MouseEventKind::ScrollDown => Some(1),
        _ => None,
    };
    let Some(delta) = delta else {
        return;
    };

    thread.scroll_anchor = if delta > 0 {
        ScrollAnchor::Bottom
    } else {
        ScrollAnchor::Top
    };
    thread.selected_chat_entry = move_selection(thread, delta);
    if matches!(app.mode, Mode::Insert) && delta < 0 {
        app.mode = Mode::Normal;
    }
}

fn yank_entry(entries: &[HistoryEntry], index: usize) -> std_io::Result<()> {
    let mut clipboard = Clipboard::new().map_err(|err| std_io::Error::other(err.to_string()))?;
    clipboard
        .set_text(clipboard_text(entries, index))
        .map_err(|err| std_io::Error::other(err.to_string()))
}

fn thread_index_by_tab_id(app: &AppState, tab_id: usize) -> Option<usize> {
    app.threads
        .iter()
        .position(|thread| thread.tab_id == tab_id)
}

async fn close_thread_connections(thread: &mut ThreadState) {
    for (_, connection) in thread.connections.drain() {
        let _ = connection.end_connection().await;
    }
}

async fn shutdown_connections(app: &mut AppState) {
    for thread in &mut app.threads {
        close_thread_connections(thread).await;
    }
}

fn start_event_reader(tx: mpsc::UnboundedSender<AppEvent>) -> thread::JoinHandle<()> {
    thread::spawn(move || {
        loop {
            match event::read() {
                Ok(event) => {
                    if tx.send(AppEvent::Input(event)).is_err() {
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

fn start_tick_loop(tx: mpsc::UnboundedSender<AppEvent>) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(Duration::from_millis(100));
        loop {
            interval.tick().await;
            if tx.send(AppEvent::Tick).is_err() {
                break;
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
            _ => return Err(format!("unrecognized argument: {arg}")),
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
            "list_files",
            "Lists files in the target directory.",
            vec![ToolParameter::optional(
                "path",
                "The project-relative directory to list files or directories for.",
                ToolParameterKind::String,
            )],
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
    ]
}

impl JobEvent {
    fn tab_id(&self) -> usize {
        match self {
            Self::Status { tab_id, .. }
            | Self::Chunk { tab_id, .. }
            | Self::Complete { tab_id, .. }
            | Self::Interrupted { tab_id, .. }
            | Self::Error { tab_id, .. } => *tab_id,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mdstream::{Block, BlockId, BlockKind, BlockStatus};

    fn text_blocks(text: &str) -> Vec<Block> {
        if text.is_empty() {
            Vec::new()
        } else {
            vec![Block {
                id: BlockId(1),
                status: BlockStatus::Committed,
                kind: BlockKind::Paragraph,
                raw: text.to_string(),
                display: None,
            }]
        }
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
    fn tab_labels_number_duplicate_auto_named_models() {
        let thread = |tab_id, model: &str| ThreadState {
            tab_id,
            base_config: ThreadConfig {
                model_id: model.to_string(),
                tools_enabled: true,
            },
            current_config: ThreadConfig {
                model_id: model.to_string(),
                tools_enabled: true,
            },
            name_override: None,
            history: Vec::new(),
            selected_chat_entry: None,
            compose_target: ComposeTarget::AppendBottom,
            draft: String::new(),
            scroll_anchor: ScrollAnchor::Bottom,
            queue: VecDeque::new(),
            running_job: false,
            running_assistant_id: None,
            loaded_models: HashSet::new(),
            connections: HashMap::new(),
            has_unseen_changes: false,
        };

        assert_eq!(
            tab_labels(&[thread(1, "smollm2"), thread(2, "smollm2")]),
            vec![String::from("smollm2-1"), String::from("smollm2-2")]
        );
    }

    #[test]
    fn effective_config_respects_latest_commands_before_target() {
        let thread = ThreadState {
            tab_id: 1,
            base_config: ThreadConfig {
                model_id: String::from("smollm2"),
                tools_enabled: true,
            },
            current_config: ThreadConfig {
                model_id: String::from("smollm2"),
                tools_enabled: true,
            },
            name_override: None,
            history: vec![
                HistoryItem {
                    id: 1,
                    entry: HistoryEntry::Command {
                        raw: String::from("tools disable"),
                        result: String::new(),
                    },
                    meta: HistoryMeta::Command(CommandKind::Tools(false)),
                },
                HistoryItem {
                    id: 2,
                    entry: HistoryEntry::Command {
                        raw: String::from("model qwen3.5"),
                        result: String::new(),
                    },
                    meta: HistoryMeta::Command(CommandKind::Model(String::from("qwen3.5"))),
                },
                HistoryItem {
                    id: 3,
                    entry: HistoryEntry::Assistant {
                        model_id: String::from("qwen3.5"),
                        prompt: String::from("hello"),
                        blocks: text_blocks("reply"),
                        callouts: vec![],
                        status: None,
                        sequence_number: None,
                    },
                    meta: HistoryMeta::None,
                },
            ],
            selected_chat_entry: None,
            compose_target: ComposeTarget::AppendBottom,
            draft: String::new(),
            scroll_anchor: ScrollAnchor::Bottom,
            queue: VecDeque::new(),
            running_job: false,
            running_assistant_id: None,
            loaded_models: HashSet::new(),
            connections: HashMap::new(),
            has_unseen_changes: false,
        };

        let config = effective_config_until(&thread, 2);
        assert_eq!(config.model_id, "qwen3.5");
        assert!(!config.tools_enabled);
    }

    #[test]
    fn selection_edit_action_allows_prompt_edits_for_assistant_messages() {
        assert!(matches!(
            selection_edit_action(&HistoryEntry::Assistant {
                model_id: String::from("m"),
                prompt: String::from("question"),
                blocks: text_blocks("answer"),
                callouts: vec![],
                status: None,
                sequence_number: None,
            }),
            Some(SelectionEditAction::Prompt(value)) if value == "question"
        ));
        assert!(matches!(
            selection_edit_action(&HistoryEntry::Command {
                raw: String::from("tools disable"),
                result: String::new(),
            }),
            Some(SelectionEditAction::Command(value)) if value == "tools disable"
        ));
        assert!(matches!(
            selection_edit_action(&HistoryEntry::LoadingModel {
                model_id: String::from("m"),
                status: None,
            }),
            None
        ));
        assert!(selection_edit_action(&HistoryEntry::SystemNotice(String::from("note"))).is_none());
    }

    #[test]
    fn open_inline_compose_inserts_after_selected_message() {
        let thread = ThreadState {
            tab_id: 1,
            base_config: ThreadConfig {
                model_id: String::from("m"),
                tools_enabled: true,
            },
            current_config: ThreadConfig {
                model_id: String::from("m"),
                tools_enabled: true,
            },
            name_override: None,
            history: vec![
                HistoryItem {
                    id: 1,
                    entry: HistoryEntry::Assistant {
                        model_id: String::from("m"),
                        prompt: String::from("first"),
                        blocks: text_blocks("one"),
                        callouts: vec![],
                        status: None,
                        sequence_number: None,
                    },
                    meta: HistoryMeta::None,
                },
                HistoryItem {
                    id: 2,
                    entry: HistoryEntry::Assistant {
                        model_id: String::from("m"),
                        prompt: String::from("second"),
                        blocks: text_blocks("two"),
                        callouts: vec![],
                        status: None,
                        sequence_number: None,
                    },
                    meta: HistoryMeta::None,
                },
            ],
            selected_chat_entry: Some(0),
            compose_target: ComposeTarget::AppendBottom,
            draft: String::new(),
            scroll_anchor: ScrollAnchor::Bottom,
            queue: VecDeque::new(),
            running_job: false,
            running_assistant_id: None,
            loaded_models: HashSet::new(),
            connections: HashMap::new(),
            has_unseen_changes: false,
        };
        let mut app = AppState {
            threads: vec![thread],
            active_thread_idx: 0,
            mode: Mode::Normal,
            command_palette: CommandPaletteState::new(vec![]),
            next_history_id: 3,
            next_tab_id: 2,
            command_edit_target: None,
            inline_command_placeholder: None,
            delete_target_id: None,
        };

        open_inline_compose_relative(&mut app, InlineComposePosition::After);

        let thread = &app.threads[0];
        assert_eq!(thread.history.len(), 3);
        assert!(matches!(
            thread.history[1].entry,
            HistoryEntry::Assistant { ref prompt, status: None, .. } if prompt.is_empty()
        ));
        assert_eq!(thread.selected_chat_entry, Some(1));
        assert_eq!(thread.compose_target, ComposeTarget::EditEntry(3));
        assert!(matches!(app.mode, Mode::Insert));
    }

    #[test]
    fn open_inline_compose_after_last_message_falls_back_to_footer() {
        let thread = ThreadState {
            tab_id: 1,
            base_config: ThreadConfig {
                model_id: String::from("m"),
                tools_enabled: true,
            },
            current_config: ThreadConfig {
                model_id: String::from("m"),
                tools_enabled: true,
            },
            name_override: None,
            history: vec![HistoryItem {
                id: 1,
                entry: HistoryEntry::Assistant {
                    model_id: String::from("m"),
                    prompt: String::from("last"),
                    blocks: text_blocks("done"),
                    callouts: vec![],
                    status: None,
                    sequence_number: None,
                },
                meta: HistoryMeta::None,
            }],
            selected_chat_entry: Some(0),
            compose_target: ComposeTarget::AppendBottom,
            draft: String::new(),
            scroll_anchor: ScrollAnchor::Bottom,
            queue: VecDeque::new(),
            running_job: false,
            running_assistant_id: None,
            loaded_models: HashSet::new(),
            connections: HashMap::new(),
            has_unseen_changes: false,
        };
        let mut app = AppState {
            threads: vec![thread],
            active_thread_idx: 0,
            mode: Mode::Normal,
            command_palette: CommandPaletteState::new(vec![]),
            next_history_id: 2,
            next_tab_id: 2,
            command_edit_target: None,
            inline_command_placeholder: None,
            delete_target_id: None,
        };

        open_inline_compose_relative(&mut app, InlineComposePosition::After);

        let thread = &app.threads[0];
        assert_eq!(thread.history.len(), 1);
        assert!(matches!(thread.compose_target, ComposeTarget::AppendBottom));
        assert!(thread.selected_chat_entry.is_none());
        assert!(matches!(app.mode, Mode::Insert));
    }

    #[test]
    fn open_inline_command_inserts_before_or_after_selected_message() {
        let base_thread = |selected_chat_entry| ThreadState {
            tab_id: 1,
            base_config: ThreadConfig {
                model_id: String::from("m"),
                tools_enabled: true,
            },
            current_config: ThreadConfig {
                model_id: String::from("m"),
                tools_enabled: true,
            },
            name_override: None,
            history: vec![
                HistoryItem {
                    id: 1,
                    entry: HistoryEntry::Assistant {
                        model_id: String::from("m"),
                        prompt: String::from("first"),
                        blocks: text_blocks("one"),
                        callouts: vec![],
                        status: None,
                        sequence_number: None,
                    },
                    meta: HistoryMeta::None,
                },
                HistoryItem {
                    id: 2,
                    entry: HistoryEntry::Assistant {
                        model_id: String::from("m"),
                        prompt: String::from("second"),
                        blocks: text_blocks("two"),
                        callouts: vec![],
                        status: None,
                        sequence_number: None,
                    },
                    meta: HistoryMeta::None,
                },
            ],
            selected_chat_entry,
            compose_target: ComposeTarget::AppendBottom,
            draft: String::new(),
            scroll_anchor: ScrollAnchor::Bottom,
            queue: VecDeque::new(),
            running_job: false,
            running_assistant_id: None,
            loaded_models: HashSet::new(),
            connections: HashMap::new(),
            has_unseen_changes: false,
        };

        let mut app = AppState {
            threads: vec![base_thread(Some(0))],
            active_thread_idx: 0,
            mode: Mode::Normal,
            command_palette: CommandPaletteState::new(vec![]),
            next_history_id: 3,
            next_tab_id: 2,
            command_edit_target: None,
            inline_command_placeholder: None,
            delete_target_id: None,
        };

        open_inline_command_relative(&mut app, InlineCommandPosition::After);
        let thread = &app.threads[0];
        assert_eq!(thread.history.len(), 3);
        assert!(matches!(
            thread.history[1].entry,
            HistoryEntry::Command { ref raw, ref result } if raw.is_empty() && result.is_empty()
        ));
        assert_eq!(thread.selected_chat_entry, Some(1));
        assert_eq!(app.command_edit_target, Some(3));
        assert!(matches!(app.mode, Mode::Command));

        let mut app = AppState {
            threads: vec![base_thread(Some(1))],
            active_thread_idx: 0,
            mode: Mode::Normal,
            command_palette: CommandPaletteState::new(vec![]),
            next_history_id: 3,
            next_tab_id: 2,
            command_edit_target: None,
            inline_command_placeholder: None,
            delete_target_id: None,
        };

        open_inline_command_relative(&mut app, InlineCommandPosition::Before);
        let thread = &app.threads[0];
        assert_eq!(thread.history.len(), 3);
        assert!(matches!(
            thread.history[1].entry,
            HistoryEntry::Command { ref raw, ref result } if raw.is_empty() && result.is_empty()
        ));
        assert_eq!(thread.selected_chat_entry, Some(1));
        assert_eq!(app.command_edit_target, Some(3));
        assert!(matches!(app.mode, Mode::Command));
    }

    #[test]
    fn canceling_inline_command_removes_placeholder_entry() {
        let thread = ThreadState {
            tab_id: 1,
            base_config: ThreadConfig {
                model_id: String::from("m"),
                tools_enabled: true,
            },
            current_config: ThreadConfig {
                model_id: String::from("m"),
                tools_enabled: true,
            },
            name_override: None,
            history: vec![
                HistoryItem {
                    id: 1,
                    entry: HistoryEntry::Assistant {
                        model_id: String::from("m"),
                        prompt: String::from("first"),
                        blocks: text_blocks("one"),
                        callouts: vec![],
                        status: None,
                        sequence_number: None,
                    },
                    meta: HistoryMeta::None,
                },
                HistoryItem {
                    id: 2,
                    entry: HistoryEntry::Assistant {
                        model_id: String::from("m"),
                        prompt: String::from("second"),
                        blocks: text_blocks("two"),
                        callouts: vec![],
                        status: None,
                        sequence_number: None,
                    },
                    meta: HistoryMeta::None,
                },
            ],
            selected_chat_entry: Some(0),
            compose_target: ComposeTarget::AppendBottom,
            draft: String::new(),
            scroll_anchor: ScrollAnchor::Bottom,
            queue: VecDeque::new(),
            running_job: false,
            running_assistant_id: None,
            loaded_models: HashSet::new(),
            connections: HashMap::new(),
            has_unseen_changes: false,
        };
        let mut app = AppState {
            threads: vec![thread],
            active_thread_idx: 0,
            mode: Mode::Normal,
            command_palette: CommandPaletteState::new(vec![]),
            next_history_id: 3,
            next_tab_id: 2,
            command_edit_target: None,
            inline_command_placeholder: None,
            delete_target_id: None,
        };

        open_inline_command_relative(&mut app, InlineCommandPosition::After);
        cancel_inline_command_insert(&mut app);

        let thread = &app.threads[0];
        assert_eq!(thread.history.len(), 2);
        assert_eq!(thread.selected_chat_entry, Some(0));
        assert!(app.inline_command_placeholder.is_none());
    }

    #[test]
    fn normalize_selection_completes_loaded_model_spinners() {
        let mut thread = ThreadState {
            tab_id: 1,
            base_config: ThreadConfig {
                model_id: String::from("m"),
                tools_enabled: true,
            },
            current_config: ThreadConfig {
                model_id: String::from("m"),
                tools_enabled: true,
            },
            name_override: None,
            history: vec![HistoryItem {
                id: 1,
                entry: HistoryEntry::LoadingModel {
                    model_id: String::from("m"),
                    status: Some(ModelLoadStatus::Loading {
                        started_at: Instant::now(),
                    }),
                },
                meta: HistoryMeta::None,
            }],
            selected_chat_entry: None,
            compose_target: ComposeTarget::AppendBottom,
            draft: String::new(),
            scroll_anchor: ScrollAnchor::Bottom,
            queue: VecDeque::new(),
            running_job: false,
            running_assistant_id: None,
            loaded_models: HashSet::from([String::from("m")]),
            connections: HashMap::new(),
            has_unseen_changes: false,
        };

        normalize_selection(&mut thread);

        assert!(matches!(
            thread.history[0].entry,
            HistoryEntry::LoadingModel {
                status: Some(ModelLoadStatus::Loaded),
                ..
            }
        ));
    }
}
