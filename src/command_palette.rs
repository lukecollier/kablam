use fst::automaton::Levenshtein;
use fst::{IntoStreamer, Set, Streamer};

const AUTOCOMPLETE_STATE_LIMIT: usize = 10_000;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CommandDispatch {
    Quit,
    OpenModelPrefix,
    SwitchModel(String),
    OpenToolsPrefix,
    SetToolsEnabled(bool),
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
}

impl CommandPaletteState {
    pub fn new(model_ids: Vec<String>) -> Self {
        let mut state = Self {
            draft: String::new(),
            highlighted: None,
            suggestions: Vec::new(),
            model_ids,
        };
        state.refresh_suggestions();
        state
    }

    pub fn open(&mut self) {
        self.draft.clear();
        self.highlighted = None;
        self.refresh_suggestions();
    }

    pub fn close(&mut self) {
        self.draft.clear();
        self.highlighted = None;
        self.suggestions.clear();
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

    pub fn view(&self) -> CommandPaletteView<'_> {
        CommandPaletteView {
            draft: self.draft(),
            preview_text: self.preview_text(),
            suggestions: self.suggestions(),
            highlighted: self.highlighted(),
        }
    }

    pub fn input_char(&mut self, ch: char) {
        self.draft.push(ch);
        self.highlighted = None;
        self.refresh_suggestions();
    }

    pub fn backspace(&mut self) {
        self.draft.pop();
        self.highlighted = None;
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
        if let Some(index) = self.highlighted {
            if let Some(suggestion) = self.suggestions.get(index).cloned() {
                self.draft = suggestion.display.clone();
                self.highlighted = None;
                self.refresh_suggestions();

                return match suggestion.dispatch {
                    CommandDispatch::OpenModelPrefix => Ok(CommandCommit::StayOpen),
                    dispatch => Ok(CommandCommit::Execute(dispatch)),
                };
            }
        }

        let parsed = parse_command(self.draft())?;
        Ok(CommandCommit::Execute(parsed))
    }

    fn refresh_suggestions(&mut self) {
        self.suggestions = autocomplete_suggestions(self.draft(), &self.model_ids);
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

    if input == "model" {
        return Err(CommandParseError::Incomplete(String::from(
            "usage: model <model-id>",
        )));
    }

    if input == "tools" {
        return Err(CommandParseError::Incomplete(String::from(
            "usage: tools enable|disable",
        )));
    }

    if let Some(model_id) = input.strip_prefix("model ") {
        let model_id = model_id.trim();
        if model_id.is_empty() {
            return Err(CommandParseError::Incomplete(String::from(
                "usage: model <model-id>",
            )));
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

    Err(CommandParseError::Invalid(format!(
        "unknown command: {input}"
    )))
}

fn autocomplete_suggestions(draft: &str, model_ids: &[String]) -> Vec<CommandSuggestion> {
    let trimmed = draft.trim_start();
    if trimmed.starts_with("model ") {
        let query = trimmed.trim_start_matches("model ").trim_start();
        return model_id_suggestions(query, model_ids);
    }

    if trimmed.starts_with("tools ") {
        let query = trimmed.trim_start_matches("tools ").trim_start();
        return tool_state_suggestions(query);
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

    root_suggestions(trimmed, model_ids)
}

fn root_suggestions(query: &str, _model_ids: &[String]) -> Vec<CommandSuggestion> {
    let mut suggestions = vec![
        CommandSuggestion {
            display: String::from("q"),
            dispatch: CommandDispatch::Quit,
        },
        CommandSuggestion {
            display: String::from("model "),
            dispatch: CommandDispatch::OpenModelPrefix,
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
        .collect::<Vec<_>>()
}

fn model_id_suggestions(query: &str, model_ids: &[String]) -> Vec<CommandSuggestion> {
    let mut suggestions = model_ids
        .iter()
        .cloned()
        .map(|model_id| CommandSuggestion {
            display: format!("model {model_id}"),
            dispatch: CommandDispatch::SwitchModel(model_id),
        })
        .collect::<Vec<_>>();

    if query.is_empty() {
        suggestions.sort_by(|left, right| left.display.cmp(&right.display));
        return suggestions;
    }

    fuzzy_rank(query, &suggestions)
        .into_iter()
        .map(|suggestion| suggestion.0)
        .collect::<Vec<_>>()
}

fn tool_state_suggestions(query: &str) -> Vec<CommandSuggestion> {
    let mut suggestions = vec![
        CommandSuggestion {
            display: String::from("tools enable"),
            dispatch: CommandDispatch::SetToolsEnabled(true),
        },
        CommandSuggestion {
            display: String::from("tools disable"),
            dispatch: CommandDispatch::SetToolsEnabled(false),
        },
    ];

    if query.is_empty() {
        suggestions.sort_by(|left, right| left.display.cmp(&right.display));
        return suggestions;
    }

    fuzzy_rank(query, &suggestions)
        .into_iter()
        .map(|suggestion| suggestion.0)
        .collect::<Vec<_>>()
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
        Err(_) => {
            return suggestions
                .iter()
                .cloned()
                .map(|suggestion| {
                    let distance = edit_distance(query, &suggestion.display);
                    (suggestion, distance)
                })
                .collect::<Vec<_>>();
        }
    };

    let limit = query.chars().count() as u32 + 6;
    let automaton = match Levenshtein::new_with_limit(query, limit, AUTOCOMPLETE_STATE_LIMIT) {
        Ok(automaton) => automaton,
        Err(_) => {
            return suggestions
                .iter()
                .cloned()
                .map(|suggestion| {
                    let distance = edit_distance(query, &suggestion.display);
                    (suggestion, distance)
                })
                .collect::<Vec<_>>();
        }
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
            let distance = edit_distance(query, &suggestion.display);
            matches.push((suggestion, distance));
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
    fn parse_command_accepts_quit_and_model_switches() {
        assert_eq!(parse_command("q"), Ok(CommandDispatch::Quit));
        assert_eq!(
            parse_command("model qwen3.5-q2k"),
            Ok(CommandDispatch::SwitchModel(String::from("qwen3.5-q2k")))
        );
        assert_eq!(
            parse_command("tools disable"),
            Ok(CommandDispatch::SetToolsEnabled(false))
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
            parse_command("unknown"),
            Err(CommandParseError::Invalid(_))
        ));
        assert!(matches!(
            parse_command("tools"),
            Err(CommandParseError::Incomplete(_))
        ));
    }

    #[test]
    fn autocomplete_ranks_by_edit_distance() {
        let suggestions = root_suggestions("mo", &["smollm2".to_string(), "q".to_string()]);
        assert_eq!(suggestions[0].display, "model ");
        assert_eq!(suggestions[1].display, "q");
    }

    #[test]
    fn command_palette_cycles_and_returns_to_editor() {
        let mut state =
            CommandPaletteState::new(vec![String::from("smollm2"), String::from("qwen")]);
        state.open();

        assert!(!state.cycle_selection());
        assert_eq!(state.highlighted(), Some(0));
        assert!(!state.cycle_selection());
        assert_eq!(state.highlighted(), Some(1));
        assert!(!state.cycle_selection());
        assert_eq!(state.highlighted(), Some(2));
        assert!(state.cycle_selection());
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

        state.input_char(' ');
        state.input_char('s');
        state.input_char('m');
        state.input_char('o');

        assert!(matches!(
            state.commit(),
            Ok(CommandCommit::Execute(CommandDispatch::SwitchModel(_)))
        ));
    }
}
