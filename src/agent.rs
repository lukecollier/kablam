use serde_json::{Value, json};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolSpec {
    pub name: String,
    pub description: String,
    pub parameters: Vec<ToolParameter>,
}

impl ToolSpec {
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        parameters: Vec<ToolParameter>,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            parameters,
        }
    }

    fn schema(&self) -> Value {
        let mut properties = serde_json::Map::new();
        let mut required = Vec::new();

        for parameter in &self.parameters {
            properties.insert(
                parameter.name.clone(),
                json!({
                    "type": parameter.kind.as_json_type(),
                    "description": parameter.description,
                }),
            );

            if parameter.required {
                required.push(Value::String(parameter.name.clone()));
            }
        }

        json!({
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            }
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolParameter {
    pub name: String,
    pub description: String,
    pub kind: ToolParameterKind,
    pub required: bool,
}

impl ToolParameter {
    pub fn required(
        name: impl Into<String>,
        description: impl Into<String>,
        kind: ToolParameterKind,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            kind,
            required: true,
        }
    }

    pub fn optional(
        name: impl Into<String>,
        description: impl Into<String>,
        kind: ToolParameterKind,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            kind,
            required: false,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolParameterKind {
    String,
    Integer,
    Boolean,
}

impl ToolParameterKind {
    fn as_json_type(self) -> &'static str {
        match self {
            Self::String => "string",
            Self::Integer => "integer",
            Self::Boolean => "boolean",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolFormat {
    XmlJson,
    JsonEnvelope,
}

impl ToolFormat {
    pub fn system_prompt(self, base_prompt: &str, tools: &[ToolSpec]) -> String {
        if tools.is_empty() {
            return base_prompt.to_string();
        }

        let tools_json = render_tools_json(tools);
        match self {
            Self::XmlJson => format!(
                "{base_prompt}\n\n\
                Use tools only when the user explicitly asks for project-local search, file reading, or the fake weather tool, or when the request cannot be answered from the conversation alone.\n\
                Do not use tools for greetings, identity questions, casual conversation, or other general knowledge questions.\n\
                Only use one of the listed tool names. Never invent a new tool name.\n\
                Available tools:\n{tools_json}\n\n\
                For SmolLM3 tool calls, output only this exact XML tag with a JSON object inside:\n\
                <tool_call>{{\"name\":\"tool_name\",\"arguments\":{{}}}}</tool_call>\n\
                Do not add prose around a tool call."
            ),
            Self::JsonEnvelope => format!(
                "{base_prompt}\n\n\
                Use tools only when the user explicitly asks for project-local search, file reading, or the fake weather tool, or when the request cannot be answered from the conversation alone.\n\
                Do not use tools for greetings, identity questions, casual conversation, or other general knowledge questions.\n\
                Only use one of the listed tool names. Never invent a new tool name.\n\
                Available tools:\n{tools_json}\n\n\
                For Qwen tool calls, output only this JSON object and no prose:\n\
                {{\"tool_call\":{{\"name\":\"tool_name\",\"arguments\":{{}}}}}}"
            ),
        }
    }

    pub fn parse(self, content: &str) -> Vec<ToolCall> {
        match self {
            Self::XmlJson => parse_xml_json_tool_calls(content),
            Self::JsonEnvelope => parse_json_envelope_tool_call(content).into_iter().collect(),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ToolCall {
    pub name: String,
    pub arguments: Value,
}

fn render_tools_json(tools: &[ToolSpec]) -> String {
    let values = tools.iter().map(ToolSpec::schema).collect::<Vec<_>>();
    serde_json::to_string_pretty(&values).expect("tool schema should serialize")
}

fn parse_xml_json_tool_calls(content: &str) -> Vec<ToolCall> {
    const OPEN: &str = "<tool_call>";
    const CLOSE: &str = "</tool_call>";

    let mut calls = Vec::new();
    let mut rest = content;

    while let Some(start) = rest.find(OPEN) {
        rest = &rest[start + OPEN.len()..];
        let Some(end) = rest.find(CLOSE) else {
            break;
        };

        let candidate = rest[..end].trim();
        if let Some(call) = parse_tool_call_value(candidate) {
            calls.push(call);
        }
        rest = &rest[end + CLOSE.len()..];
    }

    calls
}

fn parse_json_envelope_tool_call(content: &str) -> Option<ToolCall> {
    parse_tool_call_value(content.trim()).or_else(|| {
        let start = content.find('{')?;
        let end = content.rfind('}')?;
        parse_tool_call_value(&content[start..=end])
    })
}

fn parse_tool_call_value(content: &str) -> Option<ToolCall> {
    let value = serde_json::from_str::<Value>(content).ok()?;

    if let Some(envelope) = value.get("tool_call") {
        return tool_call_from_value(envelope);
    }

    tool_call_from_value(&value)
}

fn tool_call_from_value(value: &Value) -> Option<ToolCall> {
    let name = value.get("name")?.as_str()?.to_string();
    let arguments = value.get("arguments").cloned().unwrap_or_else(|| json!({}));

    Some(ToolCall { name, arguments })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn smollm_xml_tool_calls_parse_json_payloads() {
        let calls = ToolFormat::XmlJson.parse(
            r#"<tool_call>{"name":"search_docs","arguments":{"query":"cache","limit":3}}</tool_call>"#,
        );

        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "search_docs");
        assert_eq!(calls[0].arguments["query"], "cache");
        assert_eq!(calls[0].arguments["limit"], 3);
    }

    #[test]
    fn qwen_json_envelope_tool_calls_parse() {
        let calls = ToolFormat::JsonEnvelope
            .parse(r#"{"tool_call":{"name":"search_docs","arguments":{"query":"qwen tools"}}}"#);

        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "search_docs");
        assert_eq!(calls[0].arguments["query"], "qwen tools");
    }
}
