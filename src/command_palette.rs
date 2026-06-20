use fst::automaton::Levenshtein;
use fst::{IntoStreamer, Set, Streamer};

const AUTOCOMPLETE_STATE_LIMIT: usize = 10_000;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TabTarget {
    Id(usize),
    Name(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CommandDispatch {
    Quit,
    OpenModelPrefix,
    OpenToolsPrefix,
    OpenTabPrefix,
    OpenCommandAfter,
    OpenCommandBefore,
    ListModels,
    SwitchModel(String),
    SetToolsEnabled(bool),
    Break,
    ActivateTab(TabTarget),
    NewTab(String),
    RenameTab { target: TabTarget, new_name: String },
    KillTab(TabTarget),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CommandParseError {
    Empty,
    Invalid(String),
    Incomplete(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CommandSuggestion {
    pub display: String,
    pub dispatch: CommandDispatch,
}

#[derive(Debug, Clone, Copy)]
pub struct CommandPaletteView<'a> {
    pub draft: &'a str,
    pub preview_text: Option<&'a str>,
    pub suggestions: &'a [CommandSuggestion],
    pub highlighted: Option<usize>,
    pub error_text: Option<&'a str>,
    pub has_error: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CommandCommit {
    Execute(CommandDispatch),
    StayOpen,
}

#[derive(Debug, Clone)]
pub struct CommandPaletteState {
    draft: String,
    highlighted: Option<usize>,
    suggestions: Vec<CommandSuggestion>,
    model_ids: Vec<String>,
    tab_labels: Vec<String>,
    error_text: Option<String>,
}

impl CommandPaletteState {
    pub fn new(model_ids: Vec<String>) -> Self {
        let mut state = Self {
            draft: String::new(),
            highlighted: None,
            suggestions: Vec::new(),
            model_ids,
            tab_labels: Vec::new(),
            error_text: None,
        };
        state.refresh_suggestions();
        state
    }

    pub fn open(&mut self) {
        self.set_draft(String::new());
        self.clear_error();
    }

    pub fn open_with_draft(&mut self, draft: impl Into<String>) {
        self.set_draft(draft.into());
        self.clear_error();
    }

    pub fn close(&mut self) {
        self.draft.clear();
        self.highlighted = None;
        self.suggestions.clear();
        self.error_text = None;
    }

    pub fn set_model_ids<I, S>(&mut self, model_ids: I)
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        let next = model_ids.into_iter().map(Into::into).collect::<Vec<_>>();
        if self.model_ids != next {
            self.model_ids = next;
            self.refresh_suggestions();
        }
    }

    pub fn set_tab_labels<I, S>(&mut self, tab_labels: I)
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        let next = tab_labels.into_iter().map(Into::into).collect::<Vec<_>>();
        if self.tab_labels != next {
            self.tab_labels = next;
            self.refresh_suggestions();
        }
    }

    pub fn draft(&self) -> &str {
        &self.draft
    }

    pub fn suggestions(&self) -> &[CommandSuggestion] {
        &self.suggestions
    }

    pub fn highlighted(&self) -> Option<usize> {
        self.highlighted
    }

    pub fn preview_text(&self) -> Option<&str> {
        self.highlighted
            .and_then(|index| self.suggestions.get(index))
            .map(|suggestion| suggestion.display.as_str())
    }

    pub fn error_text(&self) -> Option<&str> {
        self.error_text.as_deref()
    }

    pub fn has_error(&self) -> bool {
        self.error_text.is_some()
    }

    pub fn clear_error(&mut self) {
        self.error_text = None;
    }

    pub fn set_error(&mut self, error: impl Into<String>) {
        self.error_text = Some(error.into());
    }

    pub fn view(&self) -> CommandPaletteView<'_> {
        CommandPaletteView {
            draft: self.draft(),
            preview_text: self.preview_text(),
            suggestions: self.suggestions(),
            highlighted: self.highlighted(),
            error_text: self.error_text(),
            has_error: self.has_error(),
        }
    }

    pub fn input_char(&mut self, ch: char) {
        self.draft.push(ch);
        self.highlighted = None;
        self.clear_error();
        self.refresh_suggestions();
    }

    pub fn backspace(&mut self) {
        self.draft.pop();
        self.highlighted = None;
        self.clear_error();
        self.refresh_suggestions();
    }

    pub fn cycle_selection(&mut self) -> bool {
        if self.suggestions.is_empty() {
            return false;
        }

        self.highlighted = match self.highlighted {
            None => Some(0),
            Some(index) if index + 1 < self.suggestions.len() => Some(index + 1),
            Some(_) => None,
        };

        self.highlighted.is_none()
    }

    pub fn commit(&mut self) -> Result<CommandCommit, CommandParseError> {
        self.clear_error();

        if let Some(index) = self.highlighted {
            if let Some(suggestion) = self.suggestions.get(index).cloned() {
                self.draft = suggestion.display.clone();
                self.highlighted = None;
                self.refresh_suggestions();

                return match suggestion.dispatch {
                    CommandDispatch::OpenModelPrefix
                    | CommandDispatch::OpenToolsPrefix
                    | CommandDispatch::OpenTabPrefix => Ok(CommandCommit::StayOpen),
                    dispatch => Ok(CommandCommit::Execute(dispatch)),
                };
            }
        }

        let parsed = parse_command(self.draft())?;
        Ok(CommandCommit::Execute(parsed))
    }

    fn set_draft(&mut self, draft: String) {
        self.draft = draft;
        self.highlighted = None;
        self.refresh_suggestions();
    }

    fn refresh_suggestions(&mut self) {
        self.suggestions =
            autocomplete_suggestions(self.draft(), &self.model_ids, &self.tab_labels);
        if let Some(index) = self.highlighted {
            if index >= self.suggestions.len() {
                self.highlighted = None;
            }
        }
    }
}

pub fn parse_command(input: &str) -> Result<CommandDispatch, CommandParseError> {
    let input = input.trim();
    if input.is_empty() {
        return Err(CommandParseError::Empty);
    }

    if input == "q" {
        return Ok(CommandDispatch::Quit);
    }
    if input == "break" {
        return Ok(CommandDispatch::Break);
    }
    if input == "o" {
        return Ok(CommandDispatch::OpenCommandAfter);
    }
    if input == "O" {
        return Ok(CommandDispatch::OpenCommandBefore);
    }
    if input == "model" {
        return Err(CommandParseError::Incomplete(String::from(
            "usage: model <model-id>|ls",
        )));
    }
    if input == "tools" {
        return Err(CommandParseError::Incomplete(String::from(
            "usage: tools enable|disable",
        )));
    }
    if input == "tab" {
        return Err(CommandParseError::Incomplete(String::from(
            "usage: tab <id>|new <model-id>|rename <tab> <name>|kill <tab>",
        )));
    }

    if let Some(model_id) = input.strip_prefix("model ") {
        let model_id = model_id.trim();
        if model_id.is_empty() {
            return Err(CommandParseError::Incomplete(String::from(
                "usage: model <model-id>|ls",
            )));
        }
        if model_id == "ls" {
            return Ok(CommandDispatch::ListModels);
        }
        return Ok(CommandDispatch::SwitchModel(model_id.to_string()));
    }

    if let Some(value) = input.strip_prefix("tools ") {
        let value = value.trim().to_ascii_lowercase();
        return match value.as_str() {
            "enable" => Ok(CommandDispatch::SetToolsEnabled(true)),
            "disable" => Ok(CommandDispatch::SetToolsEnabled(false)),
            "" => Err(CommandParseError::Incomplete(String::from(
                "usage: tools enable|disable",
            ))),
            _ => Err(CommandParseError::Invalid(String::from(
                "usage: tools enable|disable",
            ))),
        };
    }

    if let Some(value) = input.strip_prefix("tab ") {
        return parse_tab_command(value);
    }

    Err(CommandParseError::Invalid(format!(
        "unknown command: {input}"
    )))
}

fn parse_tab_command(input: &str) -> Result<CommandDispatch, CommandParseError> {
    let trimmed = input.trim();
    if trimmed.is_empty() {
        return Err(CommandParseError::Incomplete(String::from(
            "usage: tab <id>|new <model-id>|rename <tab> <name>|kill <tab>",
        )));
    }

    if let Some(rest) = trimmed.strip_prefix("new ") {
        let model_id = rest.trim();
        if model_id.is_empty() {
            return Err(CommandParseError::Incomplete(String::from(
                "usage: tab new <model-id>",
            )));
        }
        return Ok(CommandDispatch::NewTab(model_id.to_string()));
    }

    if let Some(rest) = trimmed.strip_prefix("rename ") {
        let (target, new_name) = split_once_whitespace(rest).ok_or_else(|| {
            CommandParseError::Incomplete(String::from("usage: tab rename <tab> <new-name>"))
        })?;
        if new_name.trim().is_empty() {
            return Err(CommandParseError::Incomplete(String::from(
                "usage: tab rename <tab> <new-name>",
            )));
        }
        return Ok(CommandDispatch::RenameTab {
            target: parse_tab_target(target),
            new_name: new_name.trim().to_string(),
        });
    }

    if let Some(rest) = trimmed.strip_prefix("kill ") {
        let target = rest.trim();
        if target.is_empty() {
            return Err(CommandParseError::Incomplete(String::from(
                "usage: tab kill <tab>",
            )));
        }
        return Ok(CommandDispatch::KillTab(parse_tab_target(target)));
    }

    Ok(CommandDispatch::ActivateTab(parse_tab_target(trimmed)))
}

fn split_once_whitespace(input: &str) -> Option<(&str, &str)> {
    let trimmed = input.trim();
    let split_at = trimmed.find(char::is_whitespace)?;
    let left = &trimmed[..split_at];
    let right = trimmed[split_at..].trim_start();
    Some((left, right))
}

fn parse_tab_target(input: &str) -> TabTarget {
    match input.trim().parse::<usize>() {
        Ok(id) if id > 0 => TabTarget::Id(id),
        _ => TabTarget::Name(input.trim().to_string()),
    }
}

fn autocomplete_suggestions(
    draft: &str,
    model_ids: &[String],
    tab_labels: &[String],
) -> Vec<CommandSuggestion> {
    let trimmed = draft.trim_start();
    if trimmed.starts_with("model ") {
        let query = trimmed.trim_start_matches("model ").trim_start();
        return model_suggestions(query, model_ids);
    }
    if trimmed.starts_with("tools ") {
        let query = trimmed.trim_start_matches("tools ").trim_start();
        return tool_state_suggestions(query);
    }
    if trimmed.starts_with("tab ") {
        let query = trimmed.trim_start_matches("tab ").trim_start();
        return tab_suggestions(query, model_ids, tab_labels);
    }
    if trimmed == "model" {
        return vec![CommandSuggestion {
            display: String::from("model "),
            dispatch: CommandDispatch::OpenModelPrefix,
        }];
    }
    if trimmed == "tools" {
        return vec![CommandSuggestion {
            display: String::from("tools "),
            dispatch: CommandDispatch::OpenToolsPrefix,
        }];
    }
    if trimmed == "tab" {
        return vec![CommandSuggestion {
            display: String::from("tab "),
            dispatch: CommandDispatch::OpenTabPrefix,
        }];
    }
    if trimmed == "o" {
        return vec![CommandSuggestion {
            display: String::from("o"),
            dispatch: CommandDispatch::OpenCommandAfter,
        }];
    }
    if trimmed == "O" {
        return vec![CommandSuggestion {
            display: String::from("O"),
            dispatch: CommandDispatch::OpenCommandBefore,
        }];
    }

    root_suggestions(trimmed)
}

fn root_suggestions(query: &str) -> Vec<CommandSuggestion> {
    let mut suggestions = vec![
        CommandSuggestion {
            display: String::from("break"),
            dispatch: CommandDispatch::Break,
        },
        CommandSuggestion {
            display: String::from("O"),
            dispatch: CommandDispatch::OpenCommandBefore,
        },
        CommandSuggestion {
            display: String::from("o"),
            dispatch: CommandDispatch::OpenCommandAfter,
        },
        CommandSuggestion {
            display: String::from("model "),
            dispatch: CommandDispatch::OpenModelPrefix,
        },
        CommandSuggestion {
            display: String::from("q"),
            dispatch: CommandDispatch::Quit,
        },
        CommandSuggestion {
            display: String::from("tab "),
            dispatch: CommandDispatch::OpenTabPrefix,
        },
        CommandSuggestion {
            display: String::from("tools "),
            dispatch: CommandDispatch::OpenToolsPrefix,
        },
    ];

    if query.is_empty() {
        suggestions.sort_by(|left, right| left.display.cmp(&right.display));
        return suggestions;
    }

    fuzzy_rank(query, &suggestions)
        .into_iter()
        .map(|suggestion| suggestion.0)
        .collect()
}

fn model_suggestions(query: &str, model_ids: &[String]) -> Vec<CommandSuggestion> {
    let mut suggestions = model_ids
        .iter()
        .cloned()
        .map(|model_id| CommandSuggestion {
            display: format!("model {model_id}"),
            dispatch: CommandDispatch::SwitchModel(model_id),
        })
        .collect::<Vec<_>>();
    suggestions.push(CommandSuggestion {
        display: String::from("model ls"),
        dispatch: CommandDispatch::ListModels,
    });

    if query.is_empty() {
        suggestions.sort_by(|left, right| left.display.cmp(&right.display));
        return suggestions;
    }

    fuzzy_rank(query, &suggestions)
        .into_iter()
        .map(|suggestion| suggestion.0)
        .collect()
}

fn tool_state_suggestions(query: &str) -> Vec<CommandSuggestion> {
    let mut suggestions = vec![
        CommandSuggestion {
            display: String::from("tools disable"),
            dispatch: CommandDispatch::SetToolsEnabled(false),
        },
        CommandSuggestion {
            display: String::from("tools enable"),
            dispatch: CommandDispatch::SetToolsEnabled(true),
        },
    ];

    if query.is_empty() {
        suggestions.sort_by(|left, right| left.display.cmp(&right.display));
        return suggestions;
    }

    fuzzy_rank(query, &suggestions)
        .into_iter()
        .map(|suggestion| suggestion.0)
        .collect()
}

fn tab_suggestions(
    query: &str,
    model_ids: &[String],
    tab_labels: &[String],
) -> Vec<CommandSuggestion> {
    let mut suggestions = Vec::new();
    suggestions.push(CommandSuggestion {
        display: String::from("tab 1"),
        dispatch: CommandDispatch::ActivateTab(TabTarget::Id(1)),
    });
    suggestions.push(CommandSuggestion {
        display: String::from("tab kill 1"),
        dispatch: CommandDispatch::KillTab(TabTarget::Id(1)),
    });
    suggestions.push(CommandSuggestion {
        display: String::from("tab rename 1 renamed"),
        dispatch: CommandDispatch::RenameTab {
            target: TabTarget::Id(1),
            new_name: String::from("renamed"),
        },
    });

    for label in tab_labels {
        suggestions.push(CommandSuggestion {
            display: format!("tab {label}"),
            dispatch: CommandDispatch::ActivateTab(TabTarget::Name(label.clone())),
        });
    }

    for model_id in model_ids {
        suggestions.push(CommandSuggestion {
            display: format!("tab new {model_id}"),
            dispatch: CommandDispatch::NewTab(model_id.clone()),
        });
    }

    if query.is_empty() {
        suggestions.sort_by(|left, right| left.display.cmp(&right.display));
        suggestions.dedup_by(|left, right| left.display == right.display);
        return suggestions;
    }

    let mut ranked = fuzzy_rank(query, &suggestions)
        .into_iter()
        .map(|suggestion| suggestion.0)
        .collect::<Vec<_>>();
    ranked.dedup_by(|left, right| left.display == right.display);
    ranked
}

fn fuzzy_rank(query: &str, suggestions: &[CommandSuggestion]) -> Vec<(CommandSuggestion, usize)> {
    let candidates = suggestions
        .iter()
        .map(|suggestion| suggestion.display.as_str())
        .collect::<Vec<_>>();
    let mut candidates = candidates;
    candidates.sort_unstable();
    let set = match Set::from_iter(candidates.iter().copied()) {
        Ok(set) => set,
        Err(_) => return fallback_rank(query, suggestions),
    };

    let limit = query.chars().count() as u32 + 6;
    let automaton = match Levenshtein::new_with_limit(query, limit, AUTOCOMPLETE_STATE_LIMIT) {
        Ok(automaton) => automaton,
        Err(_) => return fallback_rank(query, suggestions),
    };

    let mut stream = set.search(&automaton).into_stream();
    let mut matches = Vec::new();
    while let Some(key) = stream.next() {
        let Some(display) = std::str::from_utf8(key).ok() else {
            continue;
        };
        if let Some(suggestion) = suggestions
            .iter()
            .find(|suggestion| suggestion.display == display)
            .cloned()
        {
            matches.push((
                suggestion.clone(),
                edit_distance(query, &suggestion.display),
            ));
        }
    }

    matches.sort_by(|left, right| {
        let left_prefix = left.0.display.starts_with(query);
        let right_prefix = right.0.display.starts_with(query);
        left_prefix
            .cmp(&right_prefix)
            .reverse()
            .then_with(|| left.1.cmp(&right.1))
            .then_with(|| left.0.display.cmp(&right.0.display))
    });
    matches
}

fn fallback_rank(
    query: &str,
    suggestions: &[CommandSuggestion],
) -> Vec<(CommandSuggestion, usize)> {
    let mut ranked = suggestions
        .iter()
        .cloned()
        .map(|suggestion| {
            let distance = edit_distance(query, &suggestion.display);
            (suggestion, distance)
        })
        .collect::<Vec<_>>();
    ranked.sort_by(|left, right| {
        left.1
            .cmp(&right.1)
            .then_with(|| left.0.display.cmp(&right.0.display))
    });
    ranked
}

fn edit_distance(left: &str, right: &str) -> usize {
    let left = left.chars().collect::<Vec<_>>();
    let right = right.chars().collect::<Vec<_>>();
    let mut prev = (0..=right.len()).collect::<Vec<_>>();
    let mut curr = vec![0; right.len() + 1];

    for (i, left_ch) in left.iter().enumerate() {
        curr[0] = i + 1;
        for (j, right_ch) in right.iter().enumerate() {
            let substitution = usize::from(left_ch != right_ch);
            curr[j + 1] = (prev[j + 1] + 1)
                .min(curr[j] + 1)
                .min(prev[j] + substitution);
        }
        std::mem::swap(&mut prev, &mut curr);
    }

    prev[right.len()]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_command_accepts_extended_commands() {
        assert_eq!(parse_command("q"), Ok(CommandDispatch::Quit));
        assert_eq!(parse_command("break"), Ok(CommandDispatch::Break));
        assert_eq!(parse_command("o"), Ok(CommandDispatch::OpenCommandAfter));
        assert_eq!(parse_command("O"), Ok(CommandDispatch::OpenCommandBefore));
        assert_eq!(parse_command("model ls"), Ok(CommandDispatch::ListModels));
        assert_eq!(
            parse_command("model qwen3.5-q2k"),
            Ok(CommandDispatch::SwitchModel(String::from("qwen3.5-q2k")))
        );
        assert_eq!(
            parse_command("tools disable"),
            Ok(CommandDispatch::SetToolsEnabled(false))
        );
        assert_eq!(
            parse_command("tab new smollm2"),
            Ok(CommandDispatch::NewTab(String::from("smollm2")))
        );
        assert_eq!(
            parse_command("tab 2"),
            Ok(CommandDispatch::ActivateTab(TabTarget::Id(2)))
        );
    }

    #[test]
    fn parse_command_rejects_empty_and_incomplete_commands() {
        assert_eq!(parse_command(""), Err(CommandParseError::Empty));
        assert!(matches!(
            parse_command("model"),
            Err(CommandParseError::Incomplete(_))
        ));
        assert!(matches!(
            parse_command("tab"),
            Err(CommandParseError::Incomplete(_))
        ));
        assert!(matches!(
            parse_command("unknown"),
            Err(CommandParseError::Invalid(_))
        ));
        assert!(matches!(
            parse_command("tools"),
            Err(CommandParseError::Incomplete(_))
        ));
    }

    #[test]
    fn autocomplete_ranks_model_prefix_first() {
        let suggestions = root_suggestions("mo");
        assert_eq!(suggestions[0].display, "model ");
    }

    #[test]
    fn command_palette_cycles_and_returns_to_editor() {
        let mut state =
            CommandPaletteState::new(vec![String::from("smollm2"), String::from("qwen")]);
        state.open();

        assert!(!state.cycle_selection());
        assert_eq!(state.highlighted(), Some(0));
        while !state.cycle_selection() {}
        assert_eq!(state.highlighted(), None);
    }

    #[test]
    fn enter_commit_then_execute_flow_distinguishes_prefix_and_dispatch() {
        let mut state = CommandPaletteState::new(vec![String::from("smollm2")]);
        state.open();
        state.input_char('m');
        state.cycle_selection();

        assert_eq!(state.preview_text(), Some("model "));
        assert_eq!(state.commit(), Ok(CommandCommit::StayOpen));

        state.input_char('l');
        state.input_char('s');

        assert_eq!(
            state.commit(),
            Ok(CommandCommit::Execute(CommandDispatch::ListModels))
        );
    }
}
