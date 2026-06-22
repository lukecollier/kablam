use std::collections::HashMap;
use std::io;
use std::sync::OnceLock;
use std::time::Instant;

use crossterm::event::{DisableMouseCapture, EnableMouseCapture};
use crossterm::execute;
use crossterm::{cursor::MoveTo, terminal::ClearType};
use mdstream::{Block as MarkdownBlock, BlockKind};
use ratatui::buffer::Buffer;
use ratatui::layout::{Margin, Rect};
use ratatui::style::{Color, Modifier, Style, Stylize};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Clear, Paragraph, Tabs, Widget, Wrap};
use syntect::easy::HighlightLines;
use syntect::highlighting::{Style as SyntectStyle, Theme, ThemeSet};
use syntect::parsing::SyntaxSet;
use syntect::util::LinesWithEndings;

use crate::command_palette::CommandPaletteView;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HistoryEntry {
    Assistant {
        model_id: String,
        prompt: String,
        blocks: Vec<MarkdownBlock>,
        callouts: Vec<String>,
        status: Option<AssistantStatus>,
        sequence_number: Option<u64>,
    },
    LoadingModel {
        model_id: String,
        status: Option<ModelLoadStatus>,
    },
    Command {
        raw: String,
        result: String,
    },
    Break,
    SystemNotice(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModelLoadStatus {
    Loading { started_at: Instant },
    Loaded,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AssistantStatus {
    Queued { started_at: Instant },
    Loading { started_at: Instant },
    Generating { started_at: Instant },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    Insert,
    Normal,
    Command,
    ConfirmDelete,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScrollAnchor {
    Top,
    Bottom,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TabRenderInfo {
    pub label: String,
    pub has_unseen: bool,
}

pub struct RenderState<'a> {
    pub entries: &'a [HistoryEntry],
    pub selected_chat_entry: Option<usize>,
    pub mode: Mode,
    pub draft: &'a str,
    pub prompt_inline: bool,
    pub scroll_anchor: ScrollAnchor,
    pub command_palette: Option<CommandPaletteView<'a>>,
    pub tabs: &'a [TabRenderInfo],
    pub active_tab: usize,
    pub delete_confirmation: Option<&'a str>,
}

pub struct TerminalUi {
    terminal: ratatui::DefaultTerminal,
    mouse_capture: Option<MouseCaptureGuard>,
    history_scroll: HistoryScrollState,
}

impl TerminalUi {
    pub fn new() -> io::Result<Self> {
        let terminal = ratatui::init();
        let mut stdout = io::stdout();
        execute!(stdout, EnableMouseCapture)?;
        execute!(
            stdout,
            crossterm::terminal::Clear(ClearType::All),
            MoveTo(0, 0)
        )?;

        Ok(Self {
            terminal,
            mouse_capture: Some(MouseCaptureGuard),
            history_scroll: HistoryScrollState::default(),
        })
    }

    pub fn draw(&mut self, state: RenderState<'_>) -> io::Result<()> {
        self.terminal
            .draw(|frame| {
                let area = frame.area();
                frame.render_widget(Clear, area);
                render_screen(frame, area, &mut self.history_scroll, state);
            })
            .map(|_| ())
            .map_err(|err| io::Error::other(err.to_string()))
    }
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
struct HistoryScrollState {
    offset: usize,
}

impl Drop for TerminalUi {
    fn drop(&mut self) {
        let _ = self.mouse_capture.take();
        ratatui::restore();
    }
}

struct MouseCaptureGuard;

impl Drop for MouseCaptureGuard {
    fn drop(&mut self) {
        let mut stdout = io::stdout();
        let _ = execute!(stdout, DisableMouseCapture);
    }
}

pub fn chat_entry_positions(entries: &[HistoryEntry]) -> Vec<usize> {
    entries.iter().enumerate().map(|(index, _)| index).collect()
}

pub fn clipboard_text(entries: &[HistoryEntry], index: usize) -> String {
    match entries.get(index) {
        Some(HistoryEntry::Assistant {
            model_id,
            prompt,
            blocks,
            ..
        }) => format!(
            "assistant {model_id}:\n{prompt}\n\n{}",
            assistant_blocks_text(blocks)
        ),
        Some(HistoryEntry::LoadingModel { model_id, status }) => match status {
            Some(ModelLoadStatus::Loading { .. }) => format!("loading model:\n{model_id}"),
            Some(ModelLoadStatus::Loaded) | None => format!("model loaded:\n{model_id}"),
        },
        Some(HistoryEntry::Command { raw, result }) => format!("command:\n{raw}\n\n{result}"),
        Some(HistoryEntry::Break) => String::from("break"),
        Some(HistoryEntry::SystemNotice(content)) => format!("system:\n{content}"),
        None => String::new(),
    }
}

pub fn selected_transcript_index(
    entries: &[HistoryEntry],
    selected_chat_entry: Option<usize>,
) -> Option<usize> {
    let positions = chat_entry_positions(entries);
    selected_chat_entry.and_then(|selected| positions.get(selected).copied())
}

fn build_model_color_map(entries: &[HistoryEntry]) -> HashMap<String, Color> {
    const MODEL_COLORS: [Color; 12] = [
        Color::Blue,
        Color::Red,
        Color::Green,
        Color::Yellow,
        Color::Magenta,
        Color::Cyan,
        Color::LightBlue,
        Color::LightRed,
        Color::LightGreen,
        Color::LightYellow,
        Color::LightMagenta,
        Color::LightCyan,
    ];

    let mut colors = HashMap::new();
    let mut next_color = 0usize;

    for entry in entries {
        match entry {
            HistoryEntry::Assistant { model_id, .. }
            | HistoryEntry::LoadingModel { model_id, .. } => {
                if !colors.contains_key(model_id) {
                    colors.insert(
                        model_id.clone(),
                        MODEL_COLORS[next_color % MODEL_COLORS.len()],
                    );
                    next_color += 1;
                }
            }
            _ => {}
        }
    }

    colors
}

fn render_screen<T: RenderTarget>(
    target: &mut T,
    area: Rect,
    history_scroll: &mut HistoryScrollState,
    state: RenderState<'_>,
) {
    if area.width == 0 || area.height == 0 {
        return;
    }

    let tabs_height = 3u16.min(area.height);
    let after_tabs = Rect::new(
        area.x,
        area.y + tabs_height,
        area.width,
        area.height.saturating_sub(tabs_height),
    );
    render_tabs(
        target,
        Rect::new(area.x, area.y, area.width, tabs_height),
        state.tabs,
        state.active_tab,
    );

    let prompt_height = match state.mode {
        Mode::Insert if !state.prompt_inline => prompt_box_height(after_tabs.height),
        _ => 0,
    };
    let footer_height = prompt_height + 1;
    let transcript_height = after_tabs.height.saturating_sub(footer_height);

    if transcript_height > 0 {
        let model_colors = build_model_color_map(state.entries);
        let transcript_area = Rect::new(
            after_tabs.x,
            after_tabs.y,
            after_tabs.width,
            transcript_height,
        );
        target.render_block(
            Block::default().borders(Borders::ALL).title("history"),
            transcript_area,
        );
        let transcript_inner = transcript_area.inner(Margin {
            vertical: 1,
            horizontal: 1,
        });
        render_history_viewport(
            target,
            transcript_inner,
            history_scroll,
            state.entries,
            state.selected_chat_entry,
            state.mode,
            state.draft,
            state.prompt_inline,
            state.scroll_anchor,
            Some(&model_colors),
        );
    }

    let footer_area = Rect::new(
        after_tabs.x,
        after_tabs.y + transcript_height,
        after_tabs.width,
        after_tabs.height.saturating_sub(transcript_height),
    );
    render_footer(
        target,
        footer_area,
        state.mode,
        state.draft,
        state.prompt_inline,
    );

    if let Some(command_palette) = state.command_palette {
        render_command_palette_overlay(target, after_tabs, command_palette);
    }
    if let Some(text) = state.delete_confirmation {
        render_floating_message_overlay(target, after_tabs, "confirm", text, false);
    }
}

fn render_history_viewport(
    target: &mut impl RenderTarget,
    area: Rect,
    history_scroll: &mut HistoryScrollState,
    entries: &[HistoryEntry],
    selected_chat_entry: Option<usize>,
    mode: Mode,
    draft: &str,
    prompt_inline: bool,
    scroll_anchor: ScrollAnchor,
    model_colors: Option<&HashMap<String, Color>>,
) {
    if area.width == 0 || area.height == 0 {
        return;
    }

    let chat_positions = chat_entry_positions(entries);
    let content_width = transcript_content_area(area).width;
    let mut items = Vec::new();
    let mut anchor_index = None;

    for (index, entry) in entries.iter().enumerate() {
        let chat_position = chat_positions
            .iter()
            .position(|candidate| *candidate == index);
        let selected = chat_position == selected_chat_entry;
        let paragraph_index = items.len();

        if selected && prompt_inline && matches!(entry, HistoryEntry::Assistant { .. }) {
            let (prompt_paragraph, height, cursor) =
                entry.editing_paragraph(draft, content_width, model_colors);
            items.push(RenderItem::Prompt {
                paragraph: prompt_paragraph,
                height,
                cursor,
            });
            anchor_index = Some(paragraph_index);
        } else {
            let paragraph = entry.paragraph(selected, mode, model_colors);
            let height = paragraph.line_count(content_width).max(1) as u16;
            items.push(RenderItem::Paragraph { paragraph, height });
            if selected {
                anchor_index = Some(paragraph_index);
            }
        }

        if index + 1 < entries.len() {
            items.push(RenderItem::Divider { height: 1 });
        }
    }

    render_scrollbox(
        target,
        area,
        history_scroll,
        items,
        anchor_index,
        scroll_anchor,
    );
}

fn transcript_content_area(area: Rect) -> Rect {
    let padding = 2;
    let x = area.x.saturating_add(padding);
    let width = area.width.saturating_sub(padding * 2);
    Rect::new(x, area.y, width, area.height)
}

fn render_tabs(
    target: &mut impl RenderTarget,
    area: Rect,
    tabs: &[TabRenderInfo],
    active_tab: usize,
) {
    if area.width == 0 || area.height == 0 {
        return;
    }

    let titles = tabs
        .iter()
        .map(|tab| {
            let label = if tab.has_unseen {
                format!("{} *", tab.label)
            } else {
                tab.label.clone()
            };
            Line::from(label)
        })
        .collect::<Vec<_>>();
    let tabs = Tabs::new(titles)
        .block(Block::default().borders(Borders::ALL).title("threads"))
        .highlight_style(
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        )
        .select(active_tab.min(tabs.len().saturating_sub(1)));
    target.render_tabs(tabs, area);
}

fn render_scrollbox(
    target: &mut impl RenderTarget,
    area: Rect,
    history_scroll: &mut HistoryScrollState,
    items: Vec<RenderItem>,
    anchor_index: Option<usize>,
    scroll_anchor: ScrollAnchor,
) {
    let total_height = items
        .iter()
        .map(RenderItem::height)
        .map(usize::from)
        .sum::<usize>();
    let viewport_height = area.height as usize;
    if viewport_height == 0 || total_height == 0 {
        history_scroll.offset = 0;
        return;
    }

    let max_offset = total_height.saturating_sub(viewport_height);
    history_scroll.offset = history_scroll.offset.min(max_offset);
    let item_ranges = item_ranges(&items);
    if let Some(anchor_index) = anchor_index {
        let (anchor_start, anchor_end) = item_ranges[anchor_index];
        if anchor_start < history_scroll.offset
            || anchor_end > history_scroll.offset + viewport_height
        {
            history_scroll.offset = match scroll_anchor {
                ScrollAnchor::Top => anchor_start,
                ScrollAnchor::Bottom => anchor_end.saturating_sub(viewport_height),
            }
            .min(max_offset);
        }
    } else {
        history_scroll.offset = max_offset;
    }

    let viewport_start = history_scroll.offset;
    let viewport_end = viewport_start + viewport_height;
    let visible_content_height = total_height
        .saturating_sub(viewport_start)
        .min(viewport_height);
    let render_origin_y = area.y + viewport_height.saturating_sub(visible_content_height) as u16;

    for (index, item) in items.into_iter().enumerate() {
        let (item_start, item_end) = item_ranges[index];
        if item_end <= viewport_start {
            continue;
        }
        if item_start >= viewport_end {
            break;
        }

        let clipped_top = viewport_start.saturating_sub(item_start);
        let visible_start = item_start.max(viewport_start);
        let visible_end = item_end.min(viewport_end);
        let visible_height = (visible_end - visible_start) as u16;
        let render_y = render_origin_y + (visible_start - viewport_start) as u16;

        render_clipped_item(
            target,
            area,
            item,
            render_y,
            visible_height,
            clipped_top as u16,
        );
    }
}

fn render_clipped_item(
    target: &mut impl RenderTarget,
    area: Rect,
    item: RenderItem,
    render_y: u16,
    visible_height: u16,
    clipped_top: u16,
) {
    match item {
        RenderItem::Paragraph { paragraph, .. } => {
            target.render_paragraph(
                paragraph.scroll((clipped_top, 0)),
                Rect::new(area.x, render_y, area.width, visible_height),
            );
        }
        RenderItem::Prompt {
            paragraph,
            height,
            cursor,
        } => {
            let render_area = Rect::new(area.x, render_y, area.width, visible_height);
            target.render_paragraph(paragraph.scroll((clipped_top, 0)), render_area);
            set_prompt_cursor_if_visible(target, render_area, height, clipped_top, cursor);
        }
        RenderItem::Divider { .. } => {
            if clipped_top == 0 && visible_height > 0 {
                target.render_divider(Rect::new(area.x, render_y, area.width, 1));
            }
        }
    }
}

fn set_prompt_cursor_if_visible(
    target: &mut impl RenderTarget,
    render_area: Rect,
    _full_height: u16,
    clipped_top: u16,
    cursor: PromptCursor,
) {
    if render_area.width == 0 {
        return;
    }
    if cursor.row >= clipped_top && cursor.row < clipped_top + render_area.height {
        target.set_cursor(Some((
            render_area.x + cursor.column.min(render_area.width.saturating_sub(1)),
            render_area.y + cursor.row - clipped_top,
        )));
    }
}

fn render_footer(
    target: &mut impl RenderTarget,
    area: Rect,
    mode: Mode,
    draft: &str,
    prompt_inline: bool,
) {
    if area.width == 0 || area.height == 0 {
        return;
    }

    let prompt_height = match mode {
        Mode::Insert if !prompt_inline && area.height > 1 => {
            prompt_box_height(area.height.saturating_sub(1))
        }
        Mode::Insert if !prompt_inline => area.height,
        _ => 0,
    };

    if matches!(mode, Mode::Insert) && !prompt_inline && prompt_height > 0 {
        let prompt_area = Rect::new(area.x, area.y, area.width, prompt_height);
        let visible_draft = prompt_visible_text(draft, prompt_area.width, prompt_height >= 3);
        let prompt = prompt_paragraph(&visible_draft, mode, prompt_height);
        target.render_paragraph(prompt, prompt_area);
        if prompt_height >= 3 {
            target.set_cursor(Some(prompt_cursor(
                prompt_area,
                &visible_draft,
                prompt_height,
            )));
        }
    }

    if area.height > prompt_height {
        let mode_y = area.y + prompt_height;
        let mode_area = Rect::new(area.x, mode_y, area.width, 1);
        let mode_label = match mode {
            Mode::Insert => "-- INSERT --",
            Mode::Normal => "-- NORMAL --",
            Mode::Command => "-- COMMAND --",
            Mode::ConfirmDelete => "-- CONFIRM --",
        };
        let style = match mode {
            Mode::Insert => Style::default()
                .fg(Color::Green)
                .add_modifier(Modifier::BOLD),
            Mode::Normal => Style::default()
                .fg(Color::Yellow)
                .add_modifier(Modifier::BOLD),
            Mode::Command => Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
            Mode::ConfirmDelete => Style::default().fg(Color::Red).add_modifier(Modifier::BOLD),
        };
        target.render_paragraph(
            Paragraph::new(Line::from(vec![Span::styled(mode_label, style)])),
            mode_area,
        );
    }
}

fn render_command_palette_overlay(
    target: &mut impl RenderTarget,
    area: Rect,
    palette: CommandPaletteView<'_>,
) {
    if area.width == 0 || area.height == 0 {
        return;
    }

    let Some(layout) = command_palette_layout(
        area,
        palette.draft,
        palette.preview_text,
        palette.suggestions,
        palette.error_text,
    ) else {
        return;
    };

    if let Some(error_text) = palette.error_text {
        render_overlay_at(
            target,
            "error",
            error_text,
            true,
            overlay_above(layout.box_area, area, error_text),
        );
    }

    let border_style = if palette.has_error {
        Style::default().fg(Color::Red)
    } else {
        Style::default()
    };
    target.render_block(
        Block::default()
            .borders(Borders::ALL)
            .title("cmd")
            .bg(Color::Black)
            .border_style(border_style),
        layout.box_area,
    );
    target.render_paragraph(
        command_prompt_paragraph(palette.draft, palette.preview_text),
        layout.box_area.inner(Margin {
            vertical: 1,
            horizontal: 1,
        }),
    );
    target.set_cursor(Some(command_palette_cursor(layout.box_area, palette.draft)));

    if layout.list_area.height > 0 {
        target.render_paragraph(
            command_suggestions_paragraph(palette.suggestions, palette.highlighted)
                .block(Block::default().bg(Color::Black)),
            layout.list_area,
        );
    }
}

fn render_floating_message_overlay(
    target: &mut impl RenderTarget,
    area: Rect,
    title: &str,
    text: &str,
    error: bool,
) {
    render_overlay_at(target, title, text, error, overlay_centered(area, text));
}

fn render_overlay_at(
    target: &mut impl RenderTarget,
    title: &str,
    text: &str,
    error: bool,
    box_area: Rect,
) {
    if box_area.width == 0 || box_area.height == 0 {
        return;
    }
    target.render_block(
        Block::default()
            .borders(Borders::ALL)
            .title(title.to_string())
            .border_style(if error {
                Style::default().fg(Color::Red)
            } else {
                Style::default()
            }),
        box_area,
    );
    target.render_paragraph(
        Paragraph::new(Line::from(text.to_string())),
        box_area.inner(Margin {
            vertical: 1,
            horizontal: 1,
        }),
    );
}

fn overlay_centered(area: Rect, text: &str) -> Rect {
    let width = (text.chars().count() as u16 + 4)
        .max(24)
        .min(area.width.saturating_sub(2).max(1));
    let height = 3;
    let x = area.x + area.width.saturating_sub(width) / 2;
    let y = area.y + area.height.saturating_sub(6);
    Rect::new(x, y, width, height)
}

fn overlay_above(anchor: Rect, bounds: Rect, text: &str) -> Rect {
    let width = (text.chars().count() as u16 + 4)
        .max(24)
        .min(bounds.width.saturating_sub(2).max(1));
    let height = 3;
    let x = anchor.x + anchor.width.saturating_sub(width) / 2;
    let y = anchor.y.saturating_sub(height).max(bounds.y);
    Rect::new(x, y, width, height)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CommandPaletteLayout {
    pub box_area: Rect,
    pub list_area: Rect,
}

fn command_palette_layout(
    area: Rect,
    draft: &str,
    preview_text: Option<&str>,
    suggestions: &[crate::command_palette::CommandSuggestion],
    error_text: Option<&str>,
) -> Option<CommandPaletteLayout> {
    let box_height = 3;
    if area.width < 6 || area.height < box_height {
        return None;
    }
    let suggestion_width = suggestions
        .iter()
        .map(|suggestion| suggestion.display.chars().count() as u16)
        .max()
        .unwrap_or(0);
    let preview_width = preview_text
        .map(|text| text.chars().count() as u16)
        .unwrap_or(0);
    let draft_width = draft.chars().count() as u16;
    let error_width = error_text
        .map(|text| text.chars().count() as u16)
        .unwrap_or(0);
    let content_width = suggestion_width
        .max(preview_width)
        .max(draft_width)
        .max(error_width);
    let width = (content_width + 4)
        .max(24)
        .min(area.width.saturating_sub(2).max(1));
    let list_height = suggestions.len().min(6) as u16;
    let total_height = box_height + list_height;
    let x = area.x + area.width.saturating_sub(width) / 2;
    let y = area.y + area.height.saturating_sub(total_height) / 2;
    Some(CommandPaletteLayout {
        box_area: Rect::new(x, y, width, box_height),
        list_area: Rect::new(x, y + box_height, width, list_height),
    })
}

fn command_prompt_paragraph(draft: &str, preview_text: Option<&str>) -> Paragraph<'static> {
    let line = if let Some(preview_text) = preview_text {
        if preview_text.starts_with(draft) && !draft.is_empty() {
            let suffix = &preview_text[draft.len()..];
            Line::from(vec![
                Span::raw(draft.to_string()),
                Span::styled(suffix.to_string(), Style::default().fg(Color::DarkGray)),
            ])
        } else {
            Line::from(vec![Span::styled(
                preview_text.to_string(),
                Style::default().fg(Color::DarkGray),
            )])
        }
    } else {
        Line::from(draft.to_string())
    };
    Paragraph::new(line)
}

fn command_suggestions_paragraph(
    suggestions: &[crate::command_palette::CommandSuggestion],
    highlighted: Option<usize>,
) -> Paragraph<'static> {
    let mut lines = Vec::with_capacity(suggestions.len());
    for (index, suggestion) in suggestions.iter().enumerate() {
        let selected = highlighted == Some(index);
        let style = if selected {
            Style::default()
                .fg(Color::Yellow)
                .bg(Color::Gray)
                .add_modifier(Modifier::BOLD | Modifier::REVERSED)
        } else {
            Style::default().fg(Color::White)
        };
        lines.push(Line::from(vec![Span::styled(
            suggestion.display.clone(),
            style,
        )]));
    }
    Paragraph::new(lines)
}

fn command_palette_cursor(area: Rect, draft: &str) -> (u16, u16) {
    let inner_x = area.x.saturating_add(1);
    let inner_y = area.y.saturating_add(1);
    let x = inner_x + draft.chars().count() as u16;
    (x.min(area.x + area.width.saturating_sub(2)), inner_y)
}

impl HistoryEntry {
    fn paragraph(
        &self,
        selected: bool,
        _mode: Mode,
        model_colors: Option<&HashMap<String, Color>>,
    ) -> Paragraph<'static> {
        let base_style = if selected {
            Style::default().reversed().add_modifier(Modifier::BOLD)
        } else {
            Style::default().fg(Color::Gray)
        };
        let label_style = if selected {
            Style::default()
                .reversed()
                .fg(Color::Gray)
                .add_modifier(Modifier::DIM)
        } else {
            Style::default()
                .fg(Color::DarkGray)
                .add_modifier(Modifier::DIM)
        };

        match self {
            Self::Assistant {
                model_id,
                prompt,
                blocks,
                callouts,
                status,
                ..
            } => {
                let empty_model_colors = HashMap::new();
                let model_colors = model_colors.unwrap_or(&empty_model_colors);
                let model_label_style = model_colors.get(model_id).copied().map_or(
                    Style::default()
                        .fg(Color::DarkGray)
                        .add_modifier(Modifier::DIM),
                    |color| Style::default().fg(color).add_modifier(Modifier::DIM),
                );
                let mut lines = Vec::new();
                lines.push(Line::from(vec![Span::styled(
                    prompt.to_string(),
                    base_style,
                )]));
                lines.push(Line::from(Vec::<Span<'static>>::new()));
                lines.push(Line::from(vec![Span::styled(
                    model_id.to_string(),
                    if selected {
                        model_label_style.reversed()
                    } else {
                        model_label_style
                    },
                )]));

                match status {
                    Some(AssistantStatus::Queued { started_at }) => {
                        lines.push(Line::from(Vec::<Span<'static>>::new()));
                        lines.push(Line::from(vec![Span::styled(
                            format!(
                                "{} queued... {}s",
                                spinner_frame(started_at.elapsed()),
                                started_at.elapsed().as_secs()
                            ),
                            base_style,
                        )]));
                    }
                    Some(AssistantStatus::Loading { started_at }) => {
                        lines.push(Line::from(Vec::<Span<'static>>::new()));
                        lines.push(Line::from(vec![Span::styled(
                            format!(
                                "{} loading... {}s",
                                spinner_frame(started_at.elapsed()),
                                started_at.elapsed().as_secs()
                            ),
                            base_style,
                        )]));
                    }
                    Some(AssistantStatus::Generating { started_at }) => {
                        lines.push(Line::from(Vec::<Span<'static>>::new()));
                        lines.push(Line::from(vec![Span::styled(
                            format!(
                                "{} generating... {}s",
                                spinner_frame(started_at.elapsed()),
                                started_at.elapsed().as_secs()
                            ),
                            base_style,
                        )]));
                    }
                    None => {
                        if !blocks.is_empty() {
                            lines.push(Line::from(Vec::<Span<'static>>::new()));
                            lines.extend(render_markdown_blocks(blocks, base_style));
                        }
                    }
                }

                if !callouts.is_empty() {
                    lines.push(Line::from(Vec::<Span<'static>>::new()));
                    lines.extend(
                        callouts
                            .iter()
                            .cloned()
                            .map(|line| Line::from(vec![Span::styled(line, base_style)])),
                    );
                }

                Paragraph::new(lines).wrap(Wrap { trim: false })
            }
            Self::LoadingModel { model_id, status } => {
                let rendered_content = match status {
                    Some(ModelLoadStatus::Loading { started_at }) => format!(
                        "{} loading model... {}s",
                        spinner_frame(started_at.elapsed()),
                        started_at.elapsed().as_secs()
                    ),
                    Some(ModelLoadStatus::Loaded) | None => {
                        format!("model loaded: {model_id}")
                    }
                };
                labeled_paragraph(
                    "loading",
                    &rendered_content,
                    base_style.fg(Color::DarkGray),
                    label_style,
                )
            }
            Self::Command { raw, result } => labeled_paragraph(
                "cmd",
                &format!(":{raw}\n{result}"),
                base_style.fg(Color::LightBlue),
                if selected {
                    Style::default()
                        .reversed()
                        .fg(Color::LightBlue)
                        .add_modifier(Modifier::DIM)
                } else {
                    Style::default()
                        .fg(Color::LightBlue)
                        .add_modifier(Modifier::DIM)
                },
            ),
            Self::Break => labeled_paragraph(
                "break",
                "history disconnected here",
                base_style.fg(Color::LightRed),
                label_style.fg(Color::LightRed),
            ),
            Self::SystemNotice(content) => labeled_paragraph(
                "system",
                content,
                base_style.fg(Color::DarkGray),
                label_style,
            ),
        }
    }

    fn editing_paragraph(
        &self,
        draft: &str,
        content_width: u16,
        _model_colors: Option<&HashMap<String, Color>>,
    ) -> (Paragraph<'static>, u16, PromptCursor) {
        let body_style = Style::default().reversed().add_modifier(Modifier::BOLD);
        let gutter_style = Style::default()
            .fg(Color::Yellow)
            .add_modifier(Modifier::BOLD);
        let paragraph = Paragraph::new(Line::from(vec![
            Span::styled("> ", gutter_style),
            Span::styled(draft.to_string(), body_style),
        ]))
        .wrap(Wrap { trim: false });
        let height = paragraph.line_count(content_width).max(1) as u16;
        let mut cursor = wrapped_cursor_position(draft, content_width.saturating_sub(2), 0);
        if cursor.row == 0 {
            cursor.column = cursor.column.saturating_add(2);
        }
        (paragraph, height, cursor)
    }
}

fn assistant_blocks_text(blocks: &[MarkdownBlock]) -> String {
    blocks.iter().map(|block| block.raw.as_str()).collect()
}

fn render_markdown_blocks(blocks: &[MarkdownBlock], base_style: Style) -> Vec<Line<'static>> {
    let mut lines = Vec::new();

    for (index, block) in blocks.iter().enumerate() {
        if index > 0 {
            lines.push(Line::from(Vec::<Span<'static>>::new()));
        }

        if block.kind == BlockKind::CodeFence {
            lines.extend(render_code_block(block));
            continue;
        }

        let text = block.display_or_raw().trim_end_matches('\n');
        if text.is_empty() {
            continue;
        }

        lines.extend(
            text.split('\n')
                .map(|line| Line::from(vec![Span::styled(line.to_string(), base_style)])),
        );
    }

    lines
}

fn render_code_block(block: &MarkdownBlock) -> Vec<Line<'static>> {
    let body = fenced_code_body(block.display_or_raw());
    if body.is_empty() {
        return Vec::new();
    }
    let assets = syntax_assets();
    let mut lines = assets.highlight_codeblock(block);
    expand_tabs_lines(&mut lines, 2);
    lines
}

fn fenced_code_body(text: &str) -> &str {
    let Some(first_newline) = text.find('\n') else {
        return "";
    };
    let body = &text[first_newline + 1..];
    match body.rfind("\n```") {
        Some(index) => &body[..index],
        None => body,
    }
}

fn syntect_style_to_ratatui(style: SyntectStyle) -> Style {
    Style::default().fg(Color::Rgb(
        style.foreground.r,
        style.foreground.g,
        style.foreground.b,
    ))
}

fn expand_tabs_lines(lines: &mut [Line<'_>], tab_width: usize) {
    for line in lines {
        let mut col = 0;

        for span in &mut line.spans {
            span.content = expand_tabs(&span.content, tab_width, &mut col).into();
        }
    }
}

fn expand_tabs(s: &str, tab_width: usize, col: &mut usize) -> String {
    let mut out = String::new();

    for ch in s.chars() {
        match ch {
            '\t' => {
                let spaces = tab_width - (*col % tab_width);
                out.push_str(&" ".repeat(spaces));
                *col += spaces;
            }
            '\n' => {
                out.push('\n');
                *col = 0;
            }
            _ => {
                out.push(ch);
                *col += 1;
            }
        }
    }

    out
}

struct SyntaxAssets {
    syntaxes: SyntaxSet,
    theme: Theme,
}

impl SyntaxAssets {
    fn highlight_codeblock(&self, raw_code_block: &MarkdownBlock) -> Vec<Line<'static>> {
        let assets = self;
        let lang = raw_code_block.code_fence_language().unwrap_or("txt");
        let Some(ref syntax) = assets
            .syntaxes
            .find_syntax_by_extension(lang)
            .or(assets.syntaxes.find_syntax_by_name(lang))
        else {
            return raw_code_block
                .display_or_raw()
                .lines()
                .map(|line| Line::from(vec![Span::raw(line.to_string())]))
                .collect();
        };

        let code = fenced_code_body(raw_code_block.display_or_raw());
        let mut highlighter = HighlightLines::new(&syntax, &assets.theme);
        let mut lines = Vec::new();
        for line in LinesWithEndings::from(code) {
            let trimmed = line.strip_suffix('\n').unwrap_or(line);
            let ranges = match highlighter.highlight_line(line, &assets.syntaxes) {
                Ok(ranges) => ranges,
                Err(_) => {
                    lines.push(Line::from(vec![Span::raw(trimmed.to_string())]));
                    continue;
                }
            };

            let spans = ranges
                .into_iter()
                .filter(|(_, segment)| !segment.is_empty())
                .map(|(style, segment)| {
                    Span::styled(
                        segment.strip_suffix('\n').unwrap_or(segment).to_string(),
                        syntect_style_to_ratatui(style),
                    )
                })
                .collect::<Vec<_>>();
            lines.push(Line::from(spans));
        }

        if code.ends_with('\n') {
            lines.push(Line::from(Vec::<Span<'static>>::new()));
        }

        lines
    }
}

fn syntax_assets() -> &'static SyntaxAssets {
    static ASSETS: OnceLock<SyntaxAssets> = OnceLock::new();
    ASSETS.get_or_init(|| {
        let syntaxes = SyntaxSet::load_defaults_newlines();
        let theme = ThemeSet::load_defaults()
            .themes
            .remove("base16-ocean.dark")
            .or_else(|| ThemeSet::load_defaults().themes.into_values().next())
            .expect("syntect default theme set should include at least one theme");

        SyntaxAssets { syntaxes, theme }
    })
}

fn labeled_paragraph(
    label: &str,
    content: &str,
    body_style: Style,
    label_style: Style,
) -> Paragraph<'static> {
    let mut lines = Vec::with_capacity(2);
    lines.push(Line::from(vec![Span::styled(
        label.to_string(),
        label_style,
    )]));
    if !content.is_empty() {
        lines.extend(
            content
                .split('\n')
                .map(|line| Line::from(vec![Span::styled(line.to_string(), body_style)])),
        );
    }
    Paragraph::new(lines).wrap(Wrap { trim: false })
}

fn wrapped_cursor_position(text: &str, width: u16, row_offset: u16) -> PromptCursor {
    let width = width.max(1) as usize;
    let mut row = row_offset as usize;
    let lines: Vec<&str> = text.split('\n').collect();
    for (index, line) in lines.iter().enumerate() {
        let line_len = line.chars().count();
        if index + 1 == lines.len() {
            row += line_len / width;
            let column = if line_len == 0 {
                0
            } else {
                (line_len % width) as u16
            };
            return PromptCursor {
                row: row as u16,
                column,
            };
        }
        row += line_len.max(1).div_ceil(width);
    }
    PromptCursor {
        row: row as u16,
        column: 0,
    }
}

enum RenderItem {
    Paragraph {
        paragraph: Paragraph<'static>,
        height: u16,
    },
    Prompt {
        paragraph: Paragraph<'static>,
        height: u16,
        cursor: PromptCursor,
    },
    Divider {
        height: u16,
    },
}

impl RenderItem {
    fn height(&self) -> u16 {
        match self {
            Self::Paragraph { height, .. }
            | Self::Prompt { height, .. }
            | Self::Divider { height } => *height,
        }
    }
}

fn item_ranges(items: &[RenderItem]) -> Vec<(usize, usize)> {
    let mut ranges = Vec::with_capacity(items.len());
    let mut cursor = 0usize;
    for item in items {
        let start = cursor;
        cursor += item.height() as usize;
        ranges.push((start, cursor));
    }
    ranges
}

fn prompt_box_height(area_height: u16) -> u16 {
    if area_height >= 3 { 3 } else { 1 }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PromptCursor {
    row: u16,
    column: u16,
}

fn prompt_paragraph(draft: &str, mode: Mode, area_height: u16) -> Paragraph<'static> {
    let line = Line::from(vec![
        Span::styled(
            "> ",
            Style::default()
                .fg(Color::Yellow)
                .add_modifier(Modifier::BOLD),
        ),
        Span::raw(draft.to_string()),
    ]);
    match prompt_box_height(area_height) {
        3 => Paragraph::new(line)
            .block(Block::default().borders(Borders::ALL).title(match mode {
                Mode::Insert => "prompt",
                _ => "hidden",
            }))
            .wrap(Wrap { trim: false }),
        _ => Paragraph::new(line),
    }
}

fn prompt_visible_text(draft: &str, width: u16, bordered: bool) -> String {
    if width == 0 {
        return String::new();
    }
    let inner_width = if bordered {
        width.saturating_sub(3) as usize
    } else {
        width.saturating_sub(1) as usize
    };
    let chars: Vec<char> = draft.chars().collect();
    let start = chars.len().saturating_sub(inner_width);
    chars[start..].iter().collect()
}

fn prompt_cursor(area: Rect, draft: &str, height: u16) -> (u16, u16) {
    if height >= 3 {
        let inner_x = area.x + 1;
        let inner_y = area.y + 1;
        let max_x = area.x + area.width.saturating_sub(2);
        let x = inner_x + 1 + draft.chars().count() as u16;
        (x.min(max_x), inner_y)
    } else {
        (area.x + 1 + draft.chars().count() as u16, area.y)
    }
}

pub struct DividerWidget;

impl Widget for DividerWidget {
    fn render(self, area: Rect, buf: &mut Buffer) {
        if area.width == 0 || area.height == 0 {
            return;
        }
        let divider = "─".repeat(area.width as usize);
        buf.set_string(area.x, area.y, divider, Style::default().dim());
    }
}

fn spinner_frame(elapsed: std::time::Duration) -> &'static str {
    const FRAMES: [&str; 4] = ["|", "/", "-", "\\"];
    let tick = (elapsed.as_millis() / 100) as usize;
    FRAMES[tick % FRAMES.len()]
}

trait RenderTarget {
    fn render_paragraph(&mut self, paragraph: Paragraph<'static>, area: Rect);
    fn render_divider(&mut self, area: Rect);
    fn render_block(&mut self, block: Block<'static>, area: Rect);
    fn render_tabs(&mut self, tabs: Tabs<'static>, area: Rect);
    fn set_cursor(&mut self, _position: Option<(u16, u16)>) {}
}

impl RenderTarget for ratatui::Frame<'_> {
    fn render_paragraph(&mut self, paragraph: Paragraph<'static>, area: Rect) {
        self.render_widget(paragraph, area);
    }

    fn render_divider(&mut self, area: Rect) {
        self.render_widget(DividerWidget, area);
    }

    fn render_block(&mut self, block: Block<'static>, area: Rect) {
        self.render_widget(block, area);
    }

    fn render_tabs(&mut self, tabs: Tabs<'static>, area: Rect) {
        self.render_widget(tabs, area);
    }

    fn set_cursor(&mut self, position: Option<(u16, u16)>) {
        if let Some(position) = position {
            self.set_cursor_position(position);
        }
    }
}

struct BufferTarget<'a> {
    buf: &'a mut Buffer,
}

impl<'a> RenderTarget for BufferTarget<'a> {
    fn render_paragraph(&mut self, paragraph: Paragraph<'static>, area: Rect) {
        paragraph.render(area, self.buf);
    }

    fn render_divider(&mut self, area: Rect) {
        DividerWidget.render(area, self.buf);
    }

    fn render_block(&mut self, block: Block<'static>, area: Rect) {
        block.render(area, self.buf);
    }

    fn render_tabs(&mut self, tabs: Tabs<'static>, area: Rect) {
        tabs.render(area, self.buf);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::buffer::Buffer;

    #[test]
    fn can_expand_tabs() {
        let mut lines = vec![Line::from("\t\t\t")];
        expand_tabs_lines(&mut lines, 2);

        assert_eq!(lines, vec![Line::from(" ".repeat(6))]);

        let mut lines = vec![Line::from("hello\tworld\t!")];
        expand_tabs_lines(&mut lines, 1);

        assert_eq!(lines, vec![Line::from("hello world !")]);
    }

    #[test]
    fn command_palette_overlay_renders_error_box_above_command_box() {
        let area = Rect::new(0, 0, 60, 18);
        let suggestions = vec![crate::command_palette::CommandSuggestion {
            display: String::from("model ls"),
            dispatch: crate::command_palette::CommandDispatch::ListModels,
        }];
        let view = CommandPaletteView {
            draft: "model bad",
            preview_text: None,
            suggestions: &suggestions,
            highlighted: None,
            error_text: Some("unknown model"),
            has_error: true,
        };

        let mut buf = Buffer::empty(area);
        {
            let mut target = BufferTarget { buf: &mut buf };
            render_command_palette_overlay(&mut target, area, view);
        }

        assert!(buffer_contains(&buf, area, "error"));
        assert!(buffer_contains(&buf, area, "unknown model"));
        assert!(buffer_contains(&buf, area, "cmd"));
        assert!(
            buffer_line_index(&buf, area, "unknown model") < buffer_line_index(&buf, area, "cmd")
        );
    }

    #[test]
    fn tabs_render_in_header() {
        let area = Rect::new(0, 0, 60, 4);
        let mut buf = Buffer::empty(area);
        {
            let mut target = BufferTarget { buf: &mut buf };
            render_tabs(
                &mut target,
                area,
                &[
                    TabRenderInfo {
                        label: String::from("1. smollm2"),
                        has_unseen: false,
                    },
                    TabRenderInfo {
                        label: String::from("2. qwen"),
                        has_unseen: false,
                    },
                ],
                1,
            );
        }

        assert!(buffer_contains(&buf, area, "1. smollm2"));
        assert!(buffer_contains(&buf, area, "2. qwen"));
    }

    #[test]
    fn break_entry_is_rendered_distinctly() {
        let area = Rect::new(0, 0, 40, 6);
        let mut buf = Buffer::empty(area);
        {
            let mut target = BufferTarget { buf: &mut buf };
            render_history_viewport(
                &mut target,
                area,
                &mut HistoryScrollState::default(),
                &[HistoryEntry::Break],
                Some(0),
                Mode::Normal,
                "",
                false,
                ScrollAnchor::Bottom,
                None,
            );
        }

        assert!(buffer_contains(&buf, area, "history disconnected here"));
    }

    fn buffer_contains(buf: &Buffer, area: Rect, needle: &str) -> bool {
        let haystack = (0..area.height)
            .map(|row| buffer_line(buf, row, area.width))
            .collect::<Vec<_>>()
            .join("\n");
        haystack.contains(needle)
    }

    fn buffer_line_index(buf: &Buffer, area: Rect, needle: &str) -> usize {
        (0..area.height)
            .position(|row| buffer_line(buf, row, area.width).contains(needle))
            .unwrap_or(usize::MAX)
    }

    fn buffer_line(buf: &Buffer, row: u16, width: u16) -> String {
        (0..width)
            .map(|column| buf[(column, row)].symbol())
            .collect::<String>()
    }
}
