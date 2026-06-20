use std::collections::HashMap;
use std::io;
use std::time::Instant;

use crossterm::event::{DisableMouseCapture, EnableMouseCapture};
use crossterm::execute;
use crossterm::{cursor::MoveTo, terminal::ClearType};
use ratatui::buffer::Buffer;
use ratatui::layout::{Margin, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Clear, Paragraph, Widget, Wrap};

use crate::command_palette::CommandPaletteView;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TranscriptEntry {
    User(String),
    Assistant {
        model_id: String,
        content: String,
        callouts: Vec<String>,
        status: Option<AssistantStatus>,
    },
    System {
        content: String,
        status: Option<SystemStatus>,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AssistantStatus {
    Loading {
        started_at: Instant,
    },
    #[allow(dead_code)]
    Loaded,
    Generating {
        started_at: Instant,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SystemStatus {
    Loading {
        model_id: String,
        started_at: Instant,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    Insert,
    Normal,
    Command,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScrollAnchor {
    Top,
    Bottom,
}

pub struct RenderState<'a> {
    pub entries: &'a [TranscriptEntry],
    pub selected_chat_entry: Option<usize>,
    pub mode: Mode,
    pub draft: &'a str,
    pub prompt_inline: bool,
    pub scroll_anchor: ScrollAnchor,
    pub command_palette: Option<CommandPaletteView<'a>>,
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
                render_screen(
                    frame,
                    area,
                    &mut self.history_scroll,
                    state.entries,
                    state.selected_chat_entry,
                    state.mode,
                    state.draft,
                    state.prompt_inline,
                    state.scroll_anchor,
                    state.command_palette,
                );
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

pub fn chat_entry_positions(entries: &[TranscriptEntry]) -> Vec<usize> {
    entries.iter().enumerate().map(|(index, _)| index).collect()
}

pub fn clipboard_text(entry: &TranscriptEntry) -> String {
    match entry {
        TranscriptEntry::User(content) => format!("user:\n{content}"),
        TranscriptEntry::Assistant {
            model_id, content, ..
        } => format!("assistant {model_id}:\n{content}"),
        TranscriptEntry::System { content, .. } => format!("system:\n{content}"),
    }
}

pub fn selected_transcript_index(
    entries: &[TranscriptEntry],
    selected_chat_entry: Option<usize>,
) -> Option<usize> {
    let positions = chat_entry_positions(entries);
    selected_chat_entry.and_then(|selected| positions.get(selected).copied())
}

fn build_model_color_map(entries: &[TranscriptEntry]) -> HashMap<String, Color> {
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
        if let TranscriptEntry::Assistant { model_id, .. } = entry {
            if !colors.contains_key(model_id) {
                let color = MODEL_COLORS[next_color % MODEL_COLORS.len()];
                colors.insert(model_id.clone(), color);
                next_color += 1;
            }
        }
    }

    colors
}

fn render_screen<T: RenderTarget>(
    target: &mut T,
    area: Rect,
    history_scroll: &mut HistoryScrollState,
    entries: &[TranscriptEntry],
    selected_chat_entry: Option<usize>,
    mode: Mode,
    draft: &str,
    prompt_inline: bool,
    scroll_anchor: ScrollAnchor,
    command_palette: Option<CommandPaletteView<'_>>,
) {
    if area.width == 0 || area.height == 0 {
        return;
    }

    let prompt_height = match mode {
        Mode::Insert if !prompt_inline => prompt_box_height(area.height),
        _ => 0,
    };
    let footer_height = prompt_height + 1;
    let transcript_height = area.height.saturating_sub(footer_height);

    if transcript_height > 0 {
        let model_colors = build_model_color_map(entries);
        let transcript_area = Rect::new(area.x, area.y, area.width, transcript_height);
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
            entries,
            selected_chat_entry,
            mode,
            draft,
            prompt_inline,
            scroll_anchor,
            Some(&model_colors),
        );
    }

    let footer_area = Rect::new(
        area.x,
        area.y + transcript_height,
        area.width,
        area.height.saturating_sub(transcript_height),
    );
    render_footer(target, footer_area, mode, draft, prompt_inline);

    if let Some(command_palette) = command_palette {
        render_command_palette_overlay(target, area, command_palette);
    }
}

fn render_history_viewport(
    target: &mut impl RenderTarget,
    area: Rect,
    history_scroll: &mut HistoryScrollState,
    entries: &[TranscriptEntry],
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
    let content_area = transcript_content_area(area);
    let content_width = content_area.width;
    let mut items = Vec::new();
    let mut anchor_index = None;

    for (index, entry) in entries.iter().enumerate() {
        let chat_position = chat_positions
            .iter()
            .position(|candidate| *candidate == index);
        let selected = chat_position == selected_chat_entry;
        let paragraph_index = items.len();

        if selected && prompt_inline {
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
        content_area,
        history_scroll,
        items,
        anchor_index,
        prompt_inline,
        scroll_anchor,
    );
}

fn render_scrollbox(
    target: &mut impl RenderTarget,
    area: Rect,
    content_area: Rect,
    history_scroll: &mut HistoryScrollState,
    items: Vec<RenderItem>,
    anchor_index: Option<usize>,
    _prompt_inline: bool,
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
        if anchor_start < history_scroll.offset {
            history_scroll.offset = match scroll_anchor {
                ScrollAnchor::Top => anchor_start,
                ScrollAnchor::Bottom => anchor_end.saturating_sub(viewport_height),
            }
            .min(max_offset);
        } else if anchor_end > history_scroll.offset + viewport_height {
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
            content_area,
            item,
            render_y,
            visible_height,
            clipped_top as u16,
        );
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

fn render_clipped_item(
    target: &mut impl RenderTarget,
    area: Rect,
    content_area: Rect,
    item: RenderItem,
    render_y: u16,
    visible_height: u16,
    clipped_top: u16,
) {
    match item {
        RenderItem::Paragraph { paragraph, height } => {
            let render_area =
                Rect::new(content_area.x, render_y, content_area.width, visible_height);
            target.render_paragraph(paragraph.scroll((clipped_top, 0)), render_area);
            let _ = height;
        }
        RenderItem::Prompt {
            paragraph,
            height,
            cursor,
        } => {
            let render_area =
                Rect::new(content_area.x, render_y, content_area.width, visible_height);
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
        } else if prompt_area.width > 0 {
            let cursor_x = prompt_area.x
                + visible_draft
                    .chars()
                    .count()
                    .min(prompt_area.width.saturating_sub(1) as usize) as u16;
            target.set_cursor(Some((cursor_x, prompt_area.y)));
        }
    }

    if area.height > prompt_height {
        let mode_y = area.y + prompt_height;
        let mode_area = Rect::new(area.x, mode_y, area.width, 1);
        target.render_paragraph(
            Paragraph::new(Line::from(vec![Span::styled(
                match mode {
                    Mode::Insert => "-- INSERT --",
                    Mode::Normal => "-- NORMAL --",
                    Mode::Command => "-- COMMAND --",
                }
                .to_string(),
                match mode {
                    Mode::Insert => Style::default()
                        .fg(Color::Green)
                        .add_modifier(Modifier::BOLD),
                    Mode::Normal => Style::default()
                        .fg(Color::Yellow)
                        .add_modifier(Modifier::BOLD),
                    Mode::Command => Style::default()
                        .fg(Color::Cyan)
                        .add_modifier(Modifier::BOLD),
                },
            )])),
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

    let layout = command_palette_layout(
        area,
        palette.draft,
        palette.preview_text,
        palette.suggestions,
    );
    let Some(layout) = layout else {
        return;
    };

    target.render_block(
        Block::default().borders(Borders::ALL).title("cmd"),
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

    if layout.list_area.height == 0 {
        return;
    }

    target.render_paragraph(
        command_suggestions_paragraph(palette.suggestions, palette.highlighted),
        layout.list_area,
    );
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
    let content_width = suggestion_width.max(preview_width).max(draft_width);
    let width = (content_width + 4)
        .max(24)
        .min(area.width.saturating_sub(2).max(1));
    let list_height = suggestions.len().min(6) as u16;
    let total_height = box_height + list_height;
    let x = area.x + area.width.saturating_sub(width) / 2;
    let y = area.y + area.height.saturating_sub(total_height) / 2;

    let box_area = Rect::new(x, y, width, box_height);
    let list_area = Rect::new(x, y + box_height, width, list_height);
    Some(CommandPaletteLayout {
        box_area,
        list_area,
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
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD | Modifier::REVERSED)
        } else {
            Style::default().fg(Color::Gray)
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

fn transcript_content_area(area: Rect) -> Rect {
    let padding = 2;
    let x = area.x.saturating_add(padding);
    let width = area.width.saturating_sub(padding * 2);
    Rect::new(x, area.y, width, area.height)
}

impl TranscriptEntry {
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
            Self::User(content) => Paragraph::new(content.clone())
                .style(base_style)
                .wrap(Wrap { trim: false }),
            Self::Assistant {
                model_id,
                content,
                callouts,
                status,
            } => {
                let rendered_content = match status {
                    Some(AssistantStatus::Loading { started_at }) => {
                        format!(
                            "{} loading {}... {}s",
                            spinner_frame(started_at.elapsed()),
                            model_id,
                            started_at.elapsed().as_secs()
                        )
                    }
                    Some(AssistantStatus::Loaded) => format!("{model_id} model loaded"),
                    Some(AssistantStatus::Generating { started_at }) => format!(
                        "{} generating message... {}s",
                        spinner_frame(started_at.elapsed()),
                        started_at.elapsed().as_secs()
                    ),
                    None => content.clone(),
                };

                let rendered_content = if callouts.is_empty() {
                    rendered_content
                } else {
                    format!("{rendered_content}\n\n{}", callouts.join("\n"))
                };

                let empty_model_colors = HashMap::new();
                let model_colors = model_colors.unwrap_or(&empty_model_colors);
                let model_label_style = model_colors.get(model_id).copied().map_or(
                    Style::default()
                        .fg(Color::DarkGray)
                        .add_modifier(Modifier::DIM),
                    |color| Style::default().fg(color).add_modifier(Modifier::DIM),
                );

                let model_label_style = if selected {
                    model_label_style.reversed()
                } else {
                    model_label_style
                };

                labeled_paragraph(model_id, &rendered_content, base_style, model_label_style)
            }
            Self::System { content, status } => {
                let rendered_content = match status {
                    Some(SystemStatus::Loading {
                        model_id,
                        started_at,
                    }) => format!(
                        "{} loading {}... {}s",
                        spinner_frame(started_at.elapsed()),
                        model_id,
                        started_at.elapsed().as_secs()
                    ),
                    None => content.clone(),
                };

                labeled_paragraph(
                    "system",
                    &rendered_content,
                    base_style.fg(Color::DarkGray),
                    label_style,
                )
            }
        }
    }

    fn editing_paragraph(
        &self,
        draft: &str,
        content_width: u16,
        model_colors: Option<&HashMap<String, Color>>,
    ) -> (Paragraph<'static>, u16, PromptCursor) {
        let body_style = Style::default().reversed().add_modifier(Modifier::BOLD);
        let label_style = Style::default()
            .fg(Color::Gray)
            .reversed()
            .add_modifier(Modifier::DIM);

        match self {
            Self::User(_) => {
                let paragraph = Paragraph::new(draft.to_string())
                    .style(body_style)
                    .wrap(Wrap { trim: false });
                let height = paragraph.line_count(content_width).max(1) as u16;
                let cursor = wrapped_cursor_position(draft, content_width, 0);
                (paragraph, height, cursor)
            }
            Self::Assistant { model_id, .. } => {
                let empty_model_colors = HashMap::new();
                let model_colors = model_colors.unwrap_or(&empty_model_colors);
                let model_label_style = model_colors.get(model_id).copied().map_or(
                    Style::default()
                        .fg(Color::DarkGray)
                        .reversed()
                        .add_modifier(Modifier::DIM),
                    |color| {
                        Style::default()
                            .fg(color)
                            .reversed()
                            .add_modifier(Modifier::DIM)
                    },
                );

                let paragraph =
                    labeled_editing_paragraph(model_id, draft, body_style, model_label_style);
                let height = paragraph.line_count(content_width).max(1) as u16;
                let cursor = wrapped_cursor_position(draft, content_width, 1);
                (paragraph, height, cursor)
            }
            Self::System { .. } => {
                let paragraph = labeled_editing_paragraph("system", draft, body_style, label_style);
                let height = paragraph.line_count(content_width).max(1) as u16;
                let cursor = wrapped_cursor_position(draft, content_width, 1);
                (paragraph, height, cursor)
            }
        }
    }
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

fn labeled_editing_paragraph(
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
    if content.is_empty() {
        lines.push(Line::from(vec![Span::styled(String::new(), body_style)]));
    } else {
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

pub struct DividerWidget;

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

fn prompt_box_height(area_height: u16) -> u16 {
    if area_height >= 3 { 3 } else { 1 }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PromptCursor {
    row: u16,
    column: u16,
}

fn prompt_paragraph(draft: &str, mode: Mode, area_height: u16) -> Paragraph<'static> {
    match prompt_box_height(area_height) {
        3 => Paragraph::new(draft.to_string())
            .block(Block::default().borders(Borders::ALL).title(match mode {
                Mode::Insert => "prompt",
                Mode::Normal => "hidden",
                Mode::Command => "hidden",
            }))
            .wrap(Wrap { trim: false }),
        _ => Paragraph::new(draft.to_string()),
    }
}

fn prompt_visible_text(draft: &str, width: u16, bordered: bool) -> String {
    if width == 0 {
        return String::new();
    }

    let inner_width = if bordered {
        width.saturating_sub(2) as usize
    } else {
        width as usize
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
        let x = inner_x + draft.chars().count() as u16;
        (x.min(max_x), inner_y)
    } else {
        (area.x + draft.chars().count() as u16, area.y)
    }
}

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
}

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::buffer::Buffer;
    use ratatui::layout::Rect;
    use ratatui::style::Modifier;

    #[test]
    fn divider_fills_the_render_area_width() {
        for width in [1, 4, 9] {
            let area = Rect::new(0, 0, width, 1);
            let mut buf = Buffer::empty(area);
            let mut target = BufferTarget { buf: &mut buf };

            target.render_divider(area);

            let rendered = buffer_line(&buf, 0, width);
            assert_eq!(rendered, "─".repeat(width as usize));
        }
    }

    #[test]
    fn selected_chat_is_highlighted() {
        let area = Rect::new(0, 0, 40, 6);
        let mut buf = Buffer::empty(area);
        let entries = vec![
            TranscriptEntry::User("hello".to_string()),
            TranscriptEntry::Assistant {
                model_id: "qwen3.5".to_string(),
                content: "world".to_string(),
                callouts: vec![],
                status: None,
            },
        ];

        {
            let mut target = BufferTarget { buf: &mut buf };
            render_history_viewport(
                &mut target,
                area,
                &mut HistoryScrollState::default(),
                &entries,
                Some(1),
                Mode::Normal,
                "",
                false,
                ScrollAnchor::Bottom,
                None,
            );
        }

        assert!(buffer_has_bold_cell(&buf, area));
    }

    #[test]
    fn transcript_messages_have_two_column_horizontal_padding() {
        let area = Rect::new(0, 0, 20, 4);
        let mut buf = Buffer::empty(area);
        let entries = vec![TranscriptEntry::Assistant {
            model_id: "qwen3.5".to_string(),
            content: "hello".to_string(),
            callouts: vec![],
            status: None,
        }];

        {
            let mut target = BufferTarget { buf: &mut buf };
            render_history_viewport(
                &mut target,
                area,
                &mut HistoryScrollState::default(),
                &entries,
                Some(0),
                Mode::Normal,
                "",
                false,
                ScrollAnchor::Bottom,
                None,
            );
        }

        assert!(buffer_contains(&buf, area, "  qwen3.5"));
    }

    #[test]
    fn insert_mode_keeps_history_intact_above_the_footer_prompt() {
        let area = Rect::new(0, 0, 40, 10);
        let mut buf = Buffer::empty(area);
        let entries = vec![
            TranscriptEntry::User("hello".to_string()),
            TranscriptEntry::Assistant {
                model_id: "qwen3.5".to_string(),
                content: "world".to_string(),
                callouts: vec![],
                status: None,
            },
            TranscriptEntry::User("tail".to_string()),
        ];

        {
            let mut target = BufferTarget { buf: &mut buf };
            render_history_viewport(
                &mut target,
                area,
                &mut HistoryScrollState::default(),
                &entries,
                Some(1),
                Mode::Insert,
                "",
                false,
                ScrollAnchor::Bottom,
                None,
            );
        }

        assert!(buffer_contains(&buf, area, "hello"));
        assert!(buffer_contains(&buf, area, "world"));
        assert!(buffer_contains(&buf, area, "tail"));
        assert_eq!(buffer_occurrences(&buf, area, "draft"), 0);
    }

    #[test]
    fn insert_mode_inline_prompt_replaces_selected_message_in_place() {
        let area = Rect::new(0, 0, 40, 10);
        let mut buf = Buffer::empty(area);
        let entries = vec![
            TranscriptEntry::User("hello".to_string()),
            TranscriptEntry::Assistant {
                model_id: "qwen3.5".to_string(),
                content: "world".to_string(),
                callouts: vec![],
                status: None,
            },
            TranscriptEntry::User("tail".to_string()),
        ];

        {
            let mut target = BufferTarget { buf: &mut buf };
            render_history_viewport(
                &mut target,
                area,
                &mut HistoryScrollState::default(),
                &entries,
                Some(1),
                Mode::Insert,
                "draft",
                true,
                ScrollAnchor::Bottom,
                None,
            );
        }

        assert!(buffer_contains(&buf, area, "draft"));
        assert!(!buffer_contains(&buf, area, "world"));
        assert!(buffer_contains(&buf, area, "tail"));
        assert!(buffer_contains(&buf, area, "qwen3.5"));
    }

    #[test]
    fn inline_edit_preserves_existing_scroll_offset_when_selection_is_visible() {
        let area = Rect::new(0, 0, 40, 3);
        let entries = (0..8)
            .map(|index| TranscriptEntry::User(format!("message {index}")))
            .collect::<Vec<_>>();
        let mut state = HistoryScrollState { offset: 4 };
        let mut buf = Buffer::empty(area);

        {
            let mut target = BufferTarget { buf: &mut buf };
            render_history_viewport(
                &mut target,
                area,
                &mut state,
                &entries,
                Some(3),
                Mode::Insert,
                "message 3",
                true,
                ScrollAnchor::Bottom,
                None,
            );
        }

        assert_eq!(state.offset, 4);
        assert!(buffer_contains(&buf, area, "message 3"));
    }

    #[test]
    fn insert_mode_prompt_is_rendered_in_footer() {
        let area = Rect::new(0, 0, 40, 4);
        let mut buf = Buffer::empty(area);
        let mut target = BufferTarget { buf: &mut buf };

        render_footer(&mut target, area, Mode::Insert, "draft", false);

        assert!(buffer_contains(&buf, area, "draft"));
        assert!(buffer_contains(&buf, area, "INSERT"));
    }

    #[test]
    fn command_palette_layout_centers_the_box_and_places_the_list_below_it() {
        let area = Rect::new(0, 0, 80, 24);
        let suggestions = vec![
            crate::command_palette::CommandSuggestion {
                display: String::from("model smollm2"),
                dispatch: crate::command_palette::CommandDispatch::SwitchModel(String::from(
                    "smollm2",
                )),
            },
            crate::command_palette::CommandSuggestion {
                display: String::from("q"),
                dispatch: crate::command_palette::CommandDispatch::Quit,
            },
        ];

        let layout = command_palette_layout(area, "model ", Some("model smollm2"), &suggestions)
            .expect("layout should exist");

        assert_eq!(
            layout.box_area.x,
            area.x + area.width.saturating_sub(layout.box_area.width) / 2
        );
        assert_eq!(layout.list_area.x, layout.box_area.x);
        assert_eq!(
            layout.list_area.y,
            layout.box_area.y + layout.box_area.height
        );
        assert_eq!(layout.list_area.width, layout.box_area.width);
        assert_eq!(layout.list_area.height, suggestions.len() as u16);
    }

    #[test]
    fn command_palette_layout_hides_the_list_when_empty() {
        let area = Rect::new(0, 0, 40, 12);
        let layout = command_palette_layout(area, "", None, &[]).expect("layout should exist");

        assert_eq!(layout.list_area.height, 0);
    }

    #[test]
    fn command_palette_overlay_renders_suggestions_below_the_box() {
        let area = Rect::new(0, 0, 60, 12);
        let suggestions = vec![
            crate::command_palette::CommandSuggestion {
                display: String::from("model smollm2"),
                dispatch: crate::command_palette::CommandDispatch::SwitchModel(String::from(
                    "smollm2",
                )),
            },
            crate::command_palette::CommandSuggestion {
                display: String::from("q"),
                dispatch: crate::command_palette::CommandDispatch::Quit,
            },
        ];
        let view = CommandPaletteView {
            draft: "model ",
            preview_text: Some("model smollm2"),
            suggestions: &suggestions,
            highlighted: Some(0),
        };

        let mut buf = Buffer::empty(area);
        {
            let mut target = BufferTarget { buf: &mut buf };
            render_command_palette_overlay(&mut target, area, view);
        }

        assert!(buffer_contains(&buf, area, "cmd"));
        assert!(buffer_contains(&buf, area, "model smollm2"));
        assert!(
            buffer_line_index(&buf, area, "model smollm2") > buffer_line_index(&buf, area, "cmd")
        );
    }

    #[test]
    fn assistant_callouts_render_inside_the_same_box() {
        let area = Rect::new(0, 0, 40, 10);
        let mut buf = Buffer::empty(area);
        let entries = vec![TranscriptEntry::Assistant {
            model_id: "qwen3.5".to_string(),
            content: "answer".to_string(),
            callouts: vec![
                String::from("parsed tool calls:"),
                String::from("1. search_docs"),
                String::from("{\n  \"query\": \"docs\"\n}"),
            ],
            status: None,
        }];

        {
            let mut target = BufferTarget { buf: &mut buf };
            render_history_viewport(
                &mut target,
                area,
                &mut HistoryScrollState::default(),
                &entries,
                Some(0),
                Mode::Normal,
                "",
                false,
                ScrollAnchor::Bottom,
                None,
            );
        }

        assert!(buffer_contains(&buf, area, "answer"));
        assert!(buffer_contains(&buf, area, "parsed tool calls:"));
        assert_eq!(buffer_occurrences(&buf, area, "parsed tool calls:"), 1);
    }

    #[test]
    fn history_viewport_preserves_offset_until_selection_leaves_view() {
        let area = Rect::new(0, 0, 40, 3);
        let entries = (0..8)
            .map(|index| TranscriptEntry::User(format!("message {index}")))
            .collect::<Vec<_>>();

        let mut state = HistoryScrollState { offset: 4 };
        let mut buf = Buffer::empty(area);
        {
            let mut target = BufferTarget { buf: &mut buf };
            render_history_viewport(
                &mut target,
                area,
                &mut state,
                &entries,
                Some(2),
                Mode::Normal,
                "",
                false,
                ScrollAnchor::Top,
                None,
            );
        }
        assert_eq!(state.offset, 4);

        let mut buf = Buffer::empty(area);
        {
            let mut target = BufferTarget { buf: &mut buf };
            render_history_viewport(
                &mut target,
                area,
                &mut state,
                &entries,
                Some(5),
                Mode::Normal,
                "",
                false,
                ScrollAnchor::Bottom,
                None,
            );
        }

        assert_eq!(state.offset, 8);
        assert!(buffer_contains(&buf, area, "message 5"));
    }

    #[test]
    fn history_viewport_bottom_aligns_short_transcripts() {
        let area = Rect::new(0, 0, 40, 5);
        let mut buf = Buffer::empty(area);
        let entries = vec![TranscriptEntry::User("latest".to_string())];

        {
            let mut target = BufferTarget { buf: &mut buf };
            render_history_viewport(
                &mut target,
                area,
                &mut HistoryScrollState::default(),
                &entries,
                None,
                Mode::Insert,
                "",
                false,
                ScrollAnchor::Bottom,
                None,
            );
        }

        assert!(!buffer_line(&buf, 0, area.width).contains("latest"));
        assert!(buffer_line(&buf, area.height - 1, area.width).contains("latest"));
    }

    #[test]
    fn status_bar_shows_mode_indicator() {
        let area = Rect::new(0, 0, 24, 1);
        let mut buf = Buffer::empty(area);
        let mut target = BufferTarget { buf: &mut buf };

        render_footer(&mut target, area, Mode::Normal, "draft", false);

        assert!(buffer_contains(&buf, area, "NORMAL"));
        assert!(!buffer_contains(&buf, area, "draft"));
    }

    #[test]
    fn status_bar_shows_command_mode_indicator() {
        let area = Rect::new(0, 0, 24, 1);
        let mut buf = Buffer::empty(area);
        let mut target = BufferTarget { buf: &mut buf };

        render_footer(&mut target, area, Mode::Command, "", false);

        assert!(buffer_contains(&buf, area, "COMMAND"));
    }

    #[test]
    fn assistant_models_receive_distinct_colors_in_first_seen_order() {
        let entries = vec![
            TranscriptEntry::Assistant {
                model_id: "alpha".to_string(),
                content: "a".to_string(),
                callouts: vec![],
                status: None,
            },
            TranscriptEntry::Assistant {
                model_id: "beta".to_string(),
                content: "b".to_string(),
                callouts: vec![],
                status: None,
            },
            TranscriptEntry::Assistant {
                model_id: "gamma".to_string(),
                content: "c".to_string(),
                callouts: vec![],
                status: None,
            },
        ];

        let colors = build_model_color_map(&entries);

        assert_eq!(colors.get("alpha"), Some(&Color::Blue));
        assert_eq!(colors.get("beta"), Some(&Color::Red));
        assert_eq!(colors.get("gamma"), Some(&Color::Green));
    }

    #[test]
    fn clipboard_text_copies_selected_entry() {
        let entry = TranscriptEntry::Assistant {
            model_id: "qwen3.5".to_string(),
            content: "hello".to_string(),
            callouts: vec![],
            status: None,
        };

        assert_eq!(clipboard_text(&entry), "assistant qwen3.5:\nhello");
    }

    #[test]
    fn system_messages_are_selectable_and_copyable() {
        let entries = vec![
            TranscriptEntry::User("user".to_string()),
            TranscriptEntry::System {
                content: "system".to_string(),
                status: None,
            },
            TranscriptEntry::Assistant {
                model_id: "qwen3.5".to_string(),
                content: "assistant".to_string(),
                callouts: vec![],
                status: None,
            },
        ];

        assert_eq!(chat_entry_positions(&entries), vec![0, 1, 2]);
        assert_eq!(selected_transcript_index(&entries, Some(1)), Some(1));
        assert_eq!(clipboard_text(&entries[1]), "system:\nsystem");
    }

    fn buffer_line(buf: &Buffer, y: u16, width: u16) -> String {
        (0..width)
            .map(|x| buf[(x, y)].symbol().to_string())
            .collect::<Vec<_>>()
            .join("")
            .trim_end()
            .to_string()
    }

    fn buffer_contains(buf: &Buffer, area: Rect, needle: &str) -> bool {
        let mut text = String::new();
        for y in 0..area.height {
            text.push_str(&buffer_line(buf, y, area.width));
            text.push('\n');
        }
        text.contains(needle)
    }

    fn buffer_line_index(buf: &Buffer, area: Rect, needle: &str) -> u16 {
        for y in 0..area.height {
            if buffer_line(buf, y, area.width).contains(needle) {
                return y;
            }
        }
        panic!("buffer did not contain {needle:?}");
    }

    fn buffer_has_bold_cell(buf: &Buffer, area: Rect) -> bool {
        for y in 0..area.height {
            for x in 0..area.width {
                if buf[(x, y)].style().add_modifier.contains(Modifier::BOLD) {
                    return true;
                }
            }
        }
        false
    }

    fn buffer_occurrences(buf: &Buffer, area: Rect, needle: &str) -> usize {
        let mut text = String::new();
        for y in 0..area.height {
            text.push_str(&buffer_line(buf, y, area.width));
            text.push('\n');
        }
        text.match_indices(needle).count()
    }
}
