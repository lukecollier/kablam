#![allow(dead_code)]

use std::collections::HashMap;
use std::fmt;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::thread;
use std::time::Instant;

use crate::agent::ToolFormat;
use crossbeam::channel;
use mistralrs::{
    GgufModelBuilder, IsqBits, Model as MistralRuntimeModel, ModelBuilder, RequestBuilder, Response,
};
use tokio::sync::{mpsc, oneshot};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Quantization {
    AutoIsq4,
    Unquantized,
}

impl Quantization {
    fn apply(self, builder: ModelBuilder) -> ModelBuilder {
        match self {
            Self::AutoIsq4 => builder.with_auto_isq(IsqBits::Four),
            Self::Unquantized => builder,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoadMode {
    AutoIsq,
    Gguf {
        quantized_model_id: &'static str,
        quantized_filenames: &'static [&'static str],
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModelSource {
    repo_id: &'static str,
    quantization: Quantization,
    load_mode: LoadMode,
}

impl ModelSource {
    pub const fn new(repo_id: &'static str, quantization: Quantization) -> Self {
        Self {
            repo_id,
            quantization,
            load_mode: LoadMode::AutoIsq,
        }
    }

    pub const fn gguf(
        repo_id: &'static str,
        quantized_model_id: &'static str,
        quantized_filenames: &'static [&'static str],
    ) -> Self {
        Self {
            repo_id,
            quantization: Quantization::Unquantized,
            load_mode: LoadMode::Gguf {
                quantized_model_id,
                quantized_filenames,
            },
        }
    }

    pub const fn repo_id(self) -> &'static str {
        self.repo_id
    }

    pub const fn quantization(self) -> Quantization {
        self.quantization
    }

    pub const fn load_mode(self) -> LoadMode {
        self.load_mode
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Model {
    SmolLM2360MInstruct(ModelSource),
    SmolLM2(ModelSource),
    SmolLM3(ModelSource),
    Qwen35(ModelSource),
}

const QWEN3_GGUF_FILENAMES: &[&str] = &["Qwen3-4B-Q2_K.gguf"];

impl Model {
    pub fn repo_id(&self) -> &'static str {
        match self {
            Self::SmolLM2360MInstruct(source)
            | Self::SmolLM2(source)
            | Self::SmolLM3(source)
            | Self::Qwen35(source) => source.repo_id(),
        }
    }

    pub fn quantization(&self) -> Quantization {
        match self {
            Self::SmolLM2360MInstruct(source)
            | Self::SmolLM2(source)
            | Self::SmolLM3(source)
            | Self::Qwen35(source) => source.quantization(),
        }
    }

    pub fn load_mode(&self) -> LoadMode {
        match self {
            Self::SmolLM2360MInstruct(source)
            | Self::SmolLM2(source)
            | Self::SmolLM3(source)
            | Self::Qwen35(source) => source.load_mode(),
        }
    }

    pub fn tool_format(&self) -> ToolFormat {
        match self {
            Self::SmolLM2360MInstruct(_) | Self::SmolLM2(_) | Self::SmolLM3(_) => {
                ToolFormat::XmlJson
            }
            Self::Qwen35(_) => ToolFormat::JsonEnvelope,
        }
    }
}

pub fn runtime_backend() -> &'static str {
    if cfg!(feature = "mistral-cuda") {
        "cuda"
    } else if cfg!(any(feature = "mistral-metal", target_os = "macos")) {
        "metal"
    } else {
        "cpu"
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuntimeConfig {
    pub id: String,
    pub model: Model,
}

impl RuntimeConfig {
    pub fn new(id: impl Into<String>, model: Model) -> Self {
        Self {
            id: id.into(),
            model,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct RuntimeRegistry {
    configs: Vec<RuntimeConfig>,
}

impl RuntimeRegistry {
    pub fn new(configs: Vec<RuntimeConfig>) -> Self {
        Self { configs }
    }

    pub fn defaults() -> Self {
        Self::new(vec![
            RuntimeConfig::new(
                "smollm2-360m",
                Model::SmolLM2360MInstruct(ModelSource::new(
                    "HuggingFaceTB/SmolLM2-360M-Instruct",
                    Quantization::Unquantized,
                )),
            ),
            RuntimeConfig::new(
                "smollm2-360m-isq4",
                Model::SmolLM2360MInstruct(ModelSource::new(
                    "HuggingFaceTB/SmolLM2-360M-Instruct",
                    Quantization::AutoIsq4,
                )),
            ),
            RuntimeConfig::new(
                "smollm2-360m-q4",
                Model::SmolLM2360MInstruct(ModelSource::new(
                    "HuggingFaceTB/SmolLM2-360M-Instruct",
                    Quantization::AutoIsq4,
                )),
            ),
            RuntimeConfig::new(
                "smollm2",
                Model::SmolLM2(ModelSource::new(
                    "HuggingFaceTB/SmolLM2-1.7B-Instruct",
                    Quantization::AutoIsq4,
                )),
            ),
            RuntimeConfig::new(
                "smollm2-q4",
                Model::SmolLM2(ModelSource::new(
                    "nakue/SmolLM2-1.7B-W4A16-instruct",
                    Quantization::Unquantized,
                )),
            ),
            RuntimeConfig::new(
                "smollm2-fp16",
                Model::SmolLM2(ModelSource::new(
                    "HuggingFaceTB/SmolLM2-1.7B-Instruct",
                    Quantization::Unquantized,
                )),
            ),
            RuntimeConfig::new(
                "smollm3",
                Model::SmolLM3(ModelSource::new(
                    "HuggingFaceTB/SmolLM3-3B",
                    Quantization::AutoIsq4,
                )),
            ),
            RuntimeConfig::new(
                "smollm3-q4",
                Model::SmolLM3(ModelSource::new(
                    "AINovice2005/quantized-SmolLM3-3B",
                    Quantization::Unquantized,
                )),
            ),
            RuntimeConfig::new(
                "smollm3-fp16",
                Model::SmolLM3(ModelSource::new(
                    "HuggingFaceTB/SmolLM3-3B",
                    Quantization::Unquantized,
                )),
            ),
            RuntimeConfig::new(
                "qwen3.5",
                Model::Qwen35(ModelSource::new("Qwen/Qwen3-4B", Quantization::AutoIsq4)),
            ),
            RuntimeConfig::new(
                "qwen3.5-q2k",
                Model::Qwen35(ModelSource::gguf(
                    "unsloth/Qwen3-4B-GGUF",
                    "unsloth/Qwen3-4B-GGUF",
                    QWEN3_GGUF_FILENAMES,
                )),
            ),
            RuntimeConfig::new(
                "qwen3.5-fp16",
                Model::Qwen35(ModelSource::new("Qwen/Qwen3-4B", Quantization::Unquantized)),
            ),
        ])
    }

    pub fn list(&self) -> &[RuntimeConfig] {
        &self.configs
    }

    pub fn get(&self, id: &str) -> Option<&RuntimeConfig> {
        self.configs.iter().find(|config| config.id == id)
    }
}

pub enum RuntimeRequest {
    Connect {
        connection_id: ConnectionId,
        config_id: String,
        reply: oneshot::Sender<Result<(), RuntimeError>>,
    },
    Disconnect {
        connection_id: ConnectionId,
        reply: Option<oneshot::Sender<Result<(), RuntimeError>>>,
    },
    Generate {
        config_id: String,
        request: RequestBuilder,
        status: Option<mpsc::UnboundedSender<RuntimeStatus>>,
        connection_id: Option<ConnectionId>,
        sequence_number: Option<u64>,
        response: Option<mpsc::UnboundedSender<RuntimeResponseEvent>>,
        reply: Option<oneshot::Sender<Result<String, RuntimeError>>>,
    },
    Shutdown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ConnectionId(u64);

impl fmt::Display for ConnectionId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

pub enum ConnectionRequest {
    Generate {
        request: RequestBuilder,
        status: Option<mpsc::UnboundedSender<RuntimeStatus>>,
        response: mpsc::UnboundedSender<RuntimeResponseEvent>,
        sequence_number: u64,
    },
    Close {
        reply: oneshot::Sender<Result<(), RuntimeError>>,
    },
}

#[derive(Debug, PartialEq, Eq)]
pub enum RuntimeResponseEvent {
    Chunk {
        connection_id: ConnectionId,
        sequence_number: u64,
        content: String,
    },
    Complete {
        connection_id: ConnectionId,
        sequence_number: u64,
    },
    Error {
        connection_id: ConnectionId,
        sequence_number: u64,
        error: RuntimeError,
    },
}

pub struct RuntimeResponseStream {
    connection_id: ConnectionId,
    sequence_number: u64,
    rx: mpsc::UnboundedReceiver<RuntimeResponseEvent>,
}

impl RuntimeResponseStream {
    fn new(
        connection_id: ConnectionId,
        sequence_number: u64,
        rx: mpsc::UnboundedReceiver<RuntimeResponseEvent>,
    ) -> Self {
        Self {
            connection_id,
            sequence_number,
            rx,
        }
    }

    pub fn connection_id(&self) -> ConnectionId {
        self.connection_id
    }

    pub fn sequence_number(&self) -> u64 {
        self.sequence_number
    }

    pub async fn next(&mut self) -> Option<RuntimeResponseEvent> {
        self.rx.recv().await
    }

    pub async fn collect_string(mut self) -> Result<String, RuntimeError> {
        let mut response = String::new();

        while let Some(event) = self.next().await {
            match event {
                RuntimeResponseEvent::Chunk { content, .. } => response.push_str(&content),
                RuntimeResponseEvent::Complete { .. } => return Ok(response),
                RuntimeResponseEvent::Error { error, .. } => return Err(error),
            }
        }

        Err(RuntimeError::ChannelClosed)
    }
}

#[derive(Clone)]
pub struct RuntimeConnection {
    id: ConnectionId,
    config_id: String,
    tx: mpsc::UnboundedSender<ConnectionRequest>,
    signal: mpsc::Sender<ForwarderSignal>,
    closed: Arc<AtomicBool>,
    next_sequence_number: Arc<AtomicU64>,
}

impl RuntimeConnection {
    pub fn id(&self) -> ConnectionId {
        self.id
    }

    pub fn config_id(&self) -> &str {
        &self.config_id
    }

    pub fn sender(&self) -> mpsc::UnboundedSender<ConnectionRequest> {
        self.tx.clone()
    }

    async fn signal_forwarder(&self) -> Result<(), RuntimeError> {
        self.signal
            .send(ForwarderSignal::Ready(self.id))
            .await
            .map_err(|_| RuntimeError::ChannelClosed)
    }

    pub async fn stream(
        &self,
        request: impl Into<RequestBuilder>,
    ) -> Result<RuntimeResponseStream, RuntimeError> {
        self.stream_with_status(request, None).await
    }

    pub async fn stream_with_status(
        &self,
        request: impl Into<RequestBuilder>,
        status: Option<mpsc::UnboundedSender<RuntimeStatus>>,
    ) -> Result<RuntimeResponseStream, RuntimeError> {
        if self.closed.load(Ordering::Acquire) {
            return Err(RuntimeError::ConnectionClosed);
        }

        let sequence_number = self.next_sequence_number.fetch_add(1, Ordering::Relaxed);
        let (response_tx, response_rx) = mpsc::unbounded_channel();
        self.tx
            .send(ConnectionRequest::Generate {
                request: request.into(),
                status,
                response: response_tx,
                sequence_number,
            })
            .map_err(|_| RuntimeError::ConnectionClosed)?;

        self.signal_forwarder().await?;

        Ok(RuntimeResponseStream::new(
            self.id,
            sequence_number,
            response_rx,
        ))
    }

    pub async fn send(&self, request: impl Into<RequestBuilder>) -> Result<String, RuntimeError> {
        self.send_with_status(request, None).await
    }

    pub async fn send_with_status(
        &self,
        request: impl Into<RequestBuilder>,
        status: Option<mpsc::UnboundedSender<RuntimeStatus>>,
    ) -> Result<String, RuntimeError> {
        self.stream_with_status(request, status)
            .await?
            .collect_string()
            .await
    }

    pub async fn generate(
        &self,
        request: impl Into<RequestBuilder>,
    ) -> Result<String, RuntimeError> {
        self.send(request).await
    }

    pub async fn generate_with_status(
        &self,
        request: impl Into<RequestBuilder>,
        status: Option<mpsc::UnboundedSender<RuntimeStatus>>,
    ) -> Result<String, RuntimeError> {
        self.send_with_status(request, status).await
    }

    pub async fn end_connection(&self) -> Result<(), RuntimeError> {
        if self.closed.swap(true, Ordering::AcqRel) {
            return Ok(());
        }

        tracing::info!(
            connection_id = ?self.id,
            config_id = %self.config_id,
            "closing runtime connection"
        );
        let (reply_tx, reply_rx) = oneshot::channel();
        self.tx
            .send(ConnectionRequest::Close { reply: reply_tx })
            .map_err(|_| RuntimeError::ChannelClosed)?;

        self.signal_forwarder().await?;

        reply_rx.await.map_err(|_| RuntimeError::ChannelClosed)?
    }
}

impl Drop for RuntimeConnection {
    fn drop(&mut self) {
        let _ = self.signal.try_send(ForwarderSignal::Ready(self.id));
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RuntimeStatus {
    Queued { config_id: String },
    Loading { config_id: String, repo_id: String },
    Generating { config_id: String },
}

impl RuntimeStatus {
    pub fn message(&self) -> String {
        match self {
            Self::Queued { config_id } => format!("queued for {config_id}"),
            Self::Loading { config_id, repo_id } => {
                format!("loading model {config_id} from {repo_id}")
            }
            Self::Generating { config_id } => format!("generating with {config_id}"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RuntimeError {
    UnknownConfig(String),
    ModelBuild(String),
    Inference(String),
    ChannelClosed,
    ConnectionClosed,
}

impl fmt::Display for RuntimeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownConfig(id) => write!(f, "unknown runtime config: {id}"),
            Self::ModelBuild(message) => write!(f, "failed to build model: {message}"),
            Self::Inference(message) => write!(f, "inference failed: {message}"),
            Self::ChannelClosed => f.write_str("runtime request channel closed"),
            Self::ConnectionClosed => f.write_str("runtime connection closed"),
        }
    }
}

impl std::error::Error for RuntimeError {}

pub struct RuntimeBuilder {
    registry: RuntimeRegistry,
    worker_count: Option<usize>,
}

impl RuntimeBuilder {
    pub fn new() -> Self {
        Self {
            registry: RuntimeRegistry::defaults(),
            worker_count: None,
        }
    }

    pub fn with_registry(mut self, registry: RuntimeRegistry) -> Self {
        self.registry = registry;
        self
    }

    pub fn with_config(mut self, config: RuntimeConfig) -> Self {
        self.registry.configs.push(config);
        self
    }

    pub fn with_worker_count(mut self, worker_count: usize) -> Self {
        self.worker_count = Some(worker_count.max(1));
        self
    }

    pub fn build(self) -> Runtime {
        Runtime::spawn(self.registry, self.worker_count)
    }
}

impl Default for RuntimeBuilder {
    fn default() -> Self {
        Self::new()
    }
}

pub struct Runtime {
    registry: Arc<RuntimeRegistry>,
    tx: channel::Sender<RuntimeRequest>,
    next_connection_id: AtomicU64,
    connections: Arc<Mutex<HashMap<ConnectionId, ConnectionEntry>>>,
    forwarder_tx: mpsc::Sender<ForwarderSignal>,
    router: Option<thread::JoinHandle<()>>,
    forwarder: Option<thread::JoinHandle<()>>,
    workers: Vec<thread::JoinHandle<()>>,
}

impl Runtime {
    fn spawn(registry: RuntimeRegistry, worker_count: Option<usize>) -> Self {
        let registry = Arc::new(registry);
        let (tx, rx) = channel::unbounded();
        let (forwarder_tx, forwarder_rx) = mpsc::channel(1024);
        let connections = Arc::new(Mutex::new(HashMap::new()));

        let worker_count = worker_count
            .unwrap_or_else(|| registry.list().len().max(1))
            .max(1);
        let worker_layout = build_worker_layout(Arc::clone(&registry), worker_count);
        let router_worker_senders = worker_layout
            .iter()
            .map(|worker| WorkerRoute {
                config_id: worker.config_id.clone(),
                tx: worker.tx.clone(),
            })
            .collect::<Vec<_>>();
        let router_registry = Arc::clone(&registry);
        let router = thread::spawn(move || router_loop(router_registry, rx, router_worker_senders));
        let forwarder_connections = Arc::clone(&connections);
        let forwarder_router_tx = tx.clone();
        let forwarder = thread::spawn(move || {
            connection_forwarder_loop(forwarder_connections, forwarder_rx, forwarder_router_tx)
        });

        Self {
            registry,
            tx,
            next_connection_id: AtomicU64::new(1),
            connections,
            forwarder_tx,
            router: Some(router),
            forwarder: Some(forwarder),
            workers: worker_layout
                .into_iter()
                .map(|worker| worker.join)
                .collect(),
        }
    }

    pub fn list_configs(&self) -> &[RuntimeConfig] {
        self.registry.list()
    }

    pub fn config(&self, id: &str) -> Option<&RuntimeConfig> {
        self.registry.get(id)
    }

    pub fn request_sender(&self) -> channel::Sender<RuntimeRequest> {
        self.tx.clone()
    }

    pub fn request_shutdown(&self) {
        let _ = self.tx.send(RuntimeRequest::Shutdown);
        let _ = self.forwarder_tx.try_send(ForwarderSignal::Shutdown);
        if let Ok(mut connections) = self.connections.lock() {
            connections.clear();
        }
    }

    pub async fn open_connection(
        &self,
        config_id: impl Into<String>,
    ) -> Result<RuntimeConnection, RuntimeError> {
        let config_id = config_id.into();
        if self.registry.get(&config_id).is_none() {
            return Err(RuntimeError::UnknownConfig(config_id));
        }

        let id = ConnectionId(self.next_connection_id.fetch_add(1, Ordering::Relaxed));
        let (connection_tx, connection_rx) = mpsc::unbounded_channel();
        {
            let mut connections = self
                .connections
                .lock()
                .expect("connection registry poisoned");
            connections.insert(
                id,
                ConnectionEntry {
                    config_id: config_id.clone(),
                    rx: connection_rx,
                },
            );
        }
        let (reply_tx, reply_rx) = oneshot::channel();
        if let Err(err) = self.tx.send(RuntimeRequest::Connect {
            connection_id: id,
            config_id: config_id.clone(),
            reply: reply_tx,
        }) {
            if let Ok(mut connections) = self.connections.lock() {
                connections.remove(&id);
            }
            if let RuntimeRequest::Connect { reply, .. } = err.into_inner() {
                let _ = reply.send(Err(RuntimeError::ChannelClosed));
            }
            return Err(RuntimeError::ChannelClosed);
        }

        if let Err(err) = reply_rx.await.map_err(|_| RuntimeError::ChannelClosed)? {
            if let Ok(mut connections) = self.connections.lock() {
                connections.remove(&id);
            }
            return Err(err);
        }

        Ok(RuntimeConnection {
            id,
            config_id,
            tx: connection_tx,
            signal: self.forwarder_tx.clone(),
            closed: Arc::new(AtomicBool::new(false)),
            next_sequence_number: Arc::new(AtomicU64::new(1)),
        })
    }

    pub async fn generate(
        &self,
        config_id: impl Into<String>,
        request: impl Into<RequestBuilder>,
    ) -> Result<String, RuntimeError> {
        self.generate_with_status(config_id, request, None).await
    }

    pub async fn generate_with_status(
        &self,
        config_id: impl Into<String>,
        request: impl Into<RequestBuilder>,
        status: Option<mpsc::UnboundedSender<RuntimeStatus>>,
    ) -> Result<String, RuntimeError> {
        let (reply_tx, reply_rx) = oneshot::channel();
        self.tx
            .send(RuntimeRequest::Generate {
                config_id: config_id.into(),
                request: request.into(),
                status,
                connection_id: None,
                sequence_number: None,
                response: None,
                reply: Some(reply_tx),
            })
            .map_err(|_| RuntimeError::ChannelClosed)?;

        reply_rx.await.map_err(|_| RuntimeError::ChannelClosed)?
    }
}

impl Drop for Runtime {
    fn drop(&mut self) {
        self.request_shutdown();
        if let Some(router) = self.router.take() {
            let _ = router.join();
        }
        if let Some(forwarder) = self.forwarder.take() {
            let _ = forwarder.join();
        }
        for worker in self.workers.drain(..) {
            let _ = worker.join();
        }
    }
}

#[derive(Clone)]
struct WorkerRoute {
    config_id: String,
    tx: channel::Sender<WorkerRequest>,
}

struct WorkerHandle {
    config_id: String,
    tx: channel::Sender<WorkerRequest>,
    join: thread::JoinHandle<()>,
}

struct ConnectionEntry {
    config_id: String,
    rx: mpsc::UnboundedReceiver<ConnectionRequest>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ForwarderSignal {
    Ready(ConnectionId),
    Shutdown,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum ConnectionLifecycleEvent {
    Load(String),
    Unload(String),
}

#[derive(Debug, Default)]
struct ConnectionTracker {
    connection_configs: HashMap<ConnectionId, String>,
    connection_counts: HashMap<String, usize>,
}

impl ConnectionTracker {
    fn connect(
        &mut self,
        connection_id: ConnectionId,
        config_id: String,
    ) -> Option<ConnectionLifecycleEvent> {
        if self.connection_configs.contains_key(&connection_id) {
            return None;
        }

        self.connection_configs
            .insert(connection_id, config_id.clone());
        let count = self.connection_counts.entry(config_id.clone()).or_default();
        *count += 1;

        (*count == 1).then_some(ConnectionLifecycleEvent::Load(config_id))
    }

    fn disconnect(&mut self, connection_id: ConnectionId) -> Option<ConnectionLifecycleEvent> {
        let config_id = self.connection_configs.remove(&connection_id)?;
        let count = self.connection_counts.get_mut(&config_id)?;

        *count = count.saturating_sub(1);
        if *count == 0 {
            self.connection_counts.remove(&config_id);
            return Some(ConnectionLifecycleEvent::Unload(config_id));
        }

        None
    }
}

enum WorkerRequest {
    Load {
        config: RuntimeConfig,
    },
    Unload {
        config_id: String,
    },
    Generate {
        config: RuntimeConfig,
        request: RequestBuilder,
        status: Option<mpsc::UnboundedSender<RuntimeStatus>>,
        connection_id: Option<ConnectionId>,
        sequence_number: Option<u64>,
        response: Option<mpsc::UnboundedSender<RuntimeResponseEvent>>,
        reply: Option<oneshot::Sender<Result<String, RuntimeError>>>,
    },
    Shutdown,
}

fn build_worker_layout(registry: Arc<RuntimeRegistry>, worker_count: usize) -> Vec<WorkerHandle> {
    let configs = registry.list();
    if configs.is_empty() {
        return Vec::new();
    }

    let worker_count = worker_count.min(configs.len()).max(1);
    let mut workers = Vec::with_capacity(worker_count);

    for config in configs.iter().take(worker_count) {
        let (tx, rx) = channel::unbounded();
        let config_id = config.id.clone();
        let worker_registry = Arc::clone(&registry);
        let join = thread::spawn(move || worker_loop(worker_registry, rx));

        workers.push(WorkerHandle {
            config_id,
            tx,
            join,
        });
    }

    workers
}

fn connection_forwarder_loop(
    connections: Arc<Mutex<HashMap<ConnectionId, ConnectionEntry>>>,
    signal_rx: mpsc::Receiver<ForwarderSignal>,
    router_tx: channel::Sender<RuntimeRequest>,
) {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("connection forwarder tokio runtime");

    runtime.block_on(async move {
        let mut signal_rx = signal_rx;
        while let Some(signal) = signal_rx.recv().await {
            let ForwarderSignal::Ready(connection_id) = signal else {
                break;
            };

            let mut pending = Vec::new();
            let mut disconnect = None;

            {
                let mut registry = connections.lock().expect("connection registry poisoned");

                let Some(entry) = registry.get_mut(&connection_id) else {
                    continue;
                };

                loop {
                    match entry.rx.try_recv() {
                        Ok(ConnectionRequest::Generate {
                            request,
                            status,
                            response,
                            sequence_number,
                        }) => {
                            pending.push((
                                entry.config_id.clone(),
                                connection_id,
                                request,
                                status,
                                response,
                                sequence_number,
                            ));
                        }
                        Ok(ConnectionRequest::Close { reply }) => {
                            disconnect = Some(Some(reply));
                            break;
                        }
                        Err(mpsc::error::TryRecvError::Empty) => break,
                        Err(mpsc::error::TryRecvError::Disconnected) => {
                            disconnect = Some(None);
                            break;
                        }
                    }
                }

                if disconnect.is_some() {
                    registry.remove(&connection_id);
                }
            }

            for (config_id, connection_id, request, status, response, sequence_number) in pending {
                if let Err(err) = router_tx.send(RuntimeRequest::Generate {
                    config_id,
                    request,
                    status,
                    connection_id: Some(connection_id),
                    sequence_number: Some(sequence_number),
                    response: Some(response),
                    reply: None,
                }) {
                    if let RuntimeRequest::Generate {
                        response,
                        connection_id,
                        sequence_number,
                        ..
                    } = err.into_inner()
                    {
                        if let (Some(response), Some(connection_id), Some(sequence_number)) =
                            (response, connection_id, sequence_number)
                        {
                            let _ = response.send(RuntimeResponseEvent::Error {
                                connection_id,
                                sequence_number,
                                error: RuntimeError::ChannelClosed,
                            });
                        }
                    }
                }
            }

            if let Some(reply) = disconnect {
                if let Err(err) = router_tx.send(RuntimeRequest::Disconnect {
                    connection_id,
                    reply,
                }) {
                    if let RuntimeRequest::Disconnect {
                        reply: Some(reply), ..
                    } = err.into_inner()
                    {
                        let _ = reply.send(Err(RuntimeError::ChannelClosed));
                    }
                }
            }
        }
    });
}

fn router_loop(
    registry: Arc<RuntimeRegistry>,
    rx: channel::Receiver<RuntimeRequest>,
    workers: Vec<WorkerRoute>,
) {
    let mut worker_map = HashMap::new();
    for worker in workers {
        worker_map.insert(worker.config_id, worker.tx);
    }
    let mut connections = ConnectionTracker::default();

    while let Ok(request) = rx.recv() {
        match request {
            RuntimeRequest::Connect {
                connection_id,
                config_id,
                reply,
            } => {
                let Some(config) = registry.get(&config_id) else {
                    let _ = reply.send(Err(RuntimeError::UnknownConfig(config_id)));
                    continue;
                };

                let _ = connections.connect(connection_id, config.id.clone());
                let _ = reply.send(Ok(()));
            }
            RuntimeRequest::Disconnect {
                connection_id,
                reply,
            } => {
                if let Some(ConnectionLifecycleEvent::Unload(config_id)) =
                    connections.disconnect(connection_id)
                {
                    tracing::info!(
                        connection_id = ?connection_id,
                        config_id = %config_id,
                        "unloading runtime model"
                    );
                    if let Some(worker_tx) = worker_map.get(&config_id) {
                        let _ = worker_tx.send(WorkerRequest::Unload { config_id });
                    }
                }
                if let Some(reply) = reply {
                    let _ = reply.send(Ok(()));
                }
            }
            RuntimeRequest::Generate {
                config_id,
                request,
                status,
                connection_id,
                sequence_number,
                response,
                reply,
            } => {
                route_generate(
                    &registry,
                    &worker_map,
                    config_id,
                    request,
                    status,
                    connection_id,
                    sequence_number,
                    response,
                    reply,
                );
            }
            RuntimeRequest::Shutdown => {
                for worker in worker_map.values() {
                    let _ = worker.send(WorkerRequest::Shutdown);
                }
                break;
            }
        }
    }
}

fn route_generate(
    registry: &RuntimeRegistry,
    worker_map: &HashMap<String, channel::Sender<WorkerRequest>>,
    config_id: String,
    request: RequestBuilder,
    status: Option<mpsc::UnboundedSender<RuntimeStatus>>,
    connection_id: Option<ConnectionId>,
    sequence_number: Option<u64>,
    response: Option<mpsc::UnboundedSender<RuntimeResponseEvent>>,
    reply: Option<oneshot::Sender<Result<String, RuntimeError>>>,
) {
    match registry.get(&config_id) {
        Some(config) => match worker_map.get(&config.id) {
            Some(worker_tx) => {
                send_status(
                    &status,
                    RuntimeStatus::Queued {
                        config_id: config.id.clone(),
                    },
                );
                let worker_request = WorkerRequest::Generate {
                    config: config.clone(),
                    request,
                    status,
                    connection_id,
                    sequence_number,
                    response,
                    reply,
                };

                if let Err(err) = worker_tx.send(worker_request) {
                    match err.into_inner() {
                        WorkerRequest::Generate {
                            connection_id,
                            sequence_number,
                            response,
                            reply,
                            ..
                        } => {
                            if let (Some(response), Some(connection_id), Some(sequence_number)) =
                                (response, connection_id, sequence_number)
                            {
                                let _ = response.send(RuntimeResponseEvent::Error {
                                    connection_id,
                                    sequence_number,
                                    error: RuntimeError::ChannelClosed,
                                });
                            }
                            if let Some(reply) = reply {
                                let _ = reply.send(Err(RuntimeError::ChannelClosed));
                            }
                        }
                        WorkerRequest::Load { .. }
                        | WorkerRequest::Unload { .. }
                        | WorkerRequest::Shutdown => {}
                    }
                }
            }
            None => {
                if let (Some(response), Some(connection_id), Some(sequence_number)) =
                    (response.as_ref(), connection_id, sequence_number)
                {
                    let _ = response.send(RuntimeResponseEvent::Error {
                        connection_id,
                        sequence_number,
                        error: RuntimeError::UnknownConfig(config_id.clone()),
                    });
                }
                if let Some(reply) = reply {
                    let _ = reply.send(Err(RuntimeError::UnknownConfig(config_id)));
                }
            }
        },
        None => {
            if let (Some(response), Some(connection_id), Some(sequence_number)) =
                (response.as_ref(), connection_id, sequence_number)
            {
                let _ = response.send(RuntimeResponseEvent::Error {
                    connection_id,
                    sequence_number,
                    error: RuntimeError::UnknownConfig(config_id.clone()),
                });
            }
            if let Some(reply) = reply {
                let _ = reply.send(Err(RuntimeError::UnknownConfig(config_id)));
            }
        }
    }
}

fn worker_loop(registry: Arc<RuntimeRegistry>, rx: channel::Receiver<WorkerRequest>) {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("runtime worker tokio runtime");
    let mut cache: HashMap<String, MistralRuntimeModel> = HashMap::new();

    while let Ok(request) = rx.recv() {
        match request {
            WorkerRequest::Load { config } => {
                let result = runtime.block_on(async {
                    if !cache.contains_key(&config.id) {
                        let model = load_model(&config).await?;
                        cache.insert(config.id.clone(), model);
                    }

                    Ok::<(), RuntimeError>(())
                });

                if let Err(err) = result {
                    tracing::error!(
                        config_id = %config.id,
                        repo_id = %config.model.repo_id(),
                        error = %err,
                        "model preload failed"
                    );
                }
            }
            WorkerRequest::Unload { config_id } => {
                if cache.remove(&config_id).is_some() {
                    tracing::info!(config_id = %config_id, "runtime model unloaded");
                } else {
                    tracing::info!(
                        config_id = %config_id,
                        "runtime model unload requested but cache was already empty"
                    );
                }
            }
            WorkerRequest::Generate {
                config,
                request,
                status,
                connection_id,
                sequence_number,
                response,
                reply,
            } => {
                let response = runtime.block_on(async {
                    run_generation(
                        &registry,
                        &mut cache,
                        config,
                        request,
                        status,
                        connection_id,
                        sequence_number,
                        response,
                    )
                    .await
                });

                if let Some(reply) = reply {
                    let _ = reply.send(response);
                }
            }
            WorkerRequest::Shutdown => break,
        }
    }
}

async fn run_generation(
    registry: &RuntimeRegistry,
    cache: &mut HashMap<String, MistralRuntimeModel>,
    config: RuntimeConfig,
    request: RequestBuilder,
    status: Option<mpsc::UnboundedSender<RuntimeStatus>>,
    connection_id: Option<ConnectionId>,
    sequence_number: Option<u64>,
    response: Option<mpsc::UnboundedSender<RuntimeResponseEvent>>,
) -> Result<String, RuntimeError> {
    let config = registry
        .get(&config.id)
        .ok_or_else(|| RuntimeError::UnknownConfig(config.id.clone()))?;

    let emit_error = |error: RuntimeError| {
        if let (Some(response), Some(connection_id), Some(sequence_number)) =
            (response.as_ref(), connection_id, sequence_number)
        {
            let _ = response.send(RuntimeResponseEvent::Error {
                connection_id,
                sequence_number,
                error,
            });
        }
    };

    if !cache.contains_key(&config.id) {
        let load_started = Instant::now();
        tracing::info!(
            config_id = %config.id,
            repo_id = %config.model.repo_id(),
            load_mode = ?config.model.load_mode(),
            quantization = ?config.model.quantization(),
            "runtime model cache miss; starting load"
        );
        send_status(
            &status,
            RuntimeStatus::Loading {
                config_id: config.id.clone(),
                repo_id: config.model.repo_id().to_string(),
            },
        );
        let model = match load_model(config).await {
            Ok(model) => model,
            Err(err) => {
                tracing::error!(
                    config_id = %config.id,
                    repo_id = %config.model.repo_id(),
                    error = %err,
                    "model load failed"
                );
                emit_error(err.clone());
                return Err(err);
            }
        };
        cache.insert(config.id.clone(), model);
        tracing::info!(
            config_id = %config.id,
            repo_id = %config.model.repo_id(),
            load_mode = ?config.model.load_mode(),
            elapsed_ms = load_started.elapsed().as_millis(),
            "runtime model cached"
        );
    } else {
        tracing::info!(
            config_id = %config.id,
            repo_id = %config.model.repo_id(),
            "runtime model cache hit"
        );
    }

    send_status(
        &status,
        RuntimeStatus::Generating {
            config_id: config.id.clone(),
        },
    );
    let model = cache.get(&config.id).expect("model cached after load");
    tracing::info!(
        config_id = %config.id,
        repo_id = %config.model.repo_id(),
        "generation started"
    );
    let mut stream = match model.stream_chat_request(request).await {
        Ok(stream) => stream,
        Err(err) => {
            let error = RuntimeError::Inference(err.to_string());
            tracing::error!(
                config_id = %config.id,
                repo_id = %config.model.repo_id(),
                error = %error,
                "generation stream failed to start"
            );
            emit_error(error.clone());
            return Err(error);
        }
    };
    let mut streamed_content = String::new();
    let mut chunk_count = 0usize;

    let emit_chunk = |content: &str| {
        if let (Some(response), Some(connection_id), Some(sequence_number)) =
            (response.as_ref(), connection_id, sequence_number)
        {
            let _ = response.send(RuntimeResponseEvent::Chunk {
                connection_id,
                sequence_number,
                content: content.to_string(),
            });
        }
    };

    let emit_complete = || {
        if let (Some(response), Some(connection_id), Some(sequence_number)) =
            (response.as_ref(), connection_id, sequence_number)
        {
            let _ = response.send(RuntimeResponseEvent::Complete {
                connection_id,
                sequence_number,
            });
        }
    };

    while let Some(response_event) = stream.next().await {
        match response_event {
            Response::Chunk(chunk) => {
                let model_name = chunk.model.clone();
                let usage = chunk.usage;
                let mut finish_reason = None;

                for choice in chunk.choices {
                    if choice.finish_reason.is_some() {
                        finish_reason = choice.finish_reason.clone();
                    }

                    if let Some(content) = choice.delta.content {
                        if content.is_empty() {
                            continue;
                        }

                        chunk_count += 1;
                        streamed_content.push_str(&content);
                        emit_chunk(&content);
                        tracing::info!(
                            config_id = %config.id,
                            repo_id = %config.model.repo_id(),
                            model = %model_name,
                            chunk_count,
                            chunk_chars = content.chars().count(),
                            response_chars = streamed_content.chars().count(),
                            delta = %content.escape_debug(),
                            "generation token chunk"
                        );
                    }
                }

                if let Some(usage) = usage {
                    tracing::info!(
                        config_id = %config.id,
                        repo_id = %config.model.repo_id(),
                        model = %model_name,
                        chunk_count,
                        finish_reason = finish_reason.as_deref().unwrap_or("unknown"),
                        prompt_tokens = usage.prompt_tokens,
                        completion_tokens = usage.completion_tokens,
                        total_tokens = usage.total_tokens,
                        avg_tok_per_sec = usage.avg_tok_per_sec,
                        avg_prompt_tok_per_sec = usage.avg_prompt_tok_per_sec,
                        avg_completion_tok_per_sec = usage.avg_compl_tok_per_sec,
                        total_time_sec = usage.total_time_sec,
                        prompt_time_sec = usage.total_prompt_time_sec,
                        completion_time_sec = usage.total_completion_time_sec,
                        response_chars = streamed_content.chars().count(),
                        "generation completed"
                    );

                    emit_complete();
                    return Ok(streamed_content);
                }

                if let Some(finish_reason) = finish_reason {
                    tracing::warn!(
                        config_id = %config.id,
                        repo_id = %config.model.repo_id(),
                        model = %model_name,
                        chunk_count,
                        finish_reason,
                        response_chars = streamed_content.chars().count(),
                        "generation stream finished without usage"
                    );

                    emit_complete();
                    return Ok(streamed_content);
                }
            }
            Response::Done(response) => {
                let usage = &response.usage;
                let final_content = response
                    .choices
                    .first()
                    .and_then(|choice| choice.message.content.as_ref())
                    .filter(|content| !content.is_empty())
                    .cloned()
                    .unwrap_or_else(|| streamed_content.clone());

                if streamed_content.is_empty() && !final_content.is_empty() {
                    emit_chunk(&final_content);
                }

                tracing::info!(
                    config_id = %config.id,
                    repo_id = %config.model.repo_id(),
                    model = %response.model,
                    chunk_count,
                    prompt_tokens = usage.prompt_tokens,
                    completion_tokens = usage.completion_tokens,
                    total_tokens = usage.total_tokens,
                    avg_tok_per_sec = usage.avg_tok_per_sec,
                    avg_prompt_tok_per_sec = usage.avg_prompt_tok_per_sec,
                    avg_completion_tok_per_sec = usage.avg_compl_tok_per_sec,
                    total_time_sec = usage.total_time_sec,
                    prompt_time_sec = usage.total_prompt_time_sec,
                    completion_time_sec = usage.total_completion_time_sec,
                    response_chars = final_content.chars().count(),
                    "generation completed"
                );

                emit_complete();
                return Ok(final_content);
            }
            Response::ModelError(message, response) => {
                let error = RuntimeError::Inference(format!("{message}: {:?}", response.choices));
                emit_error(error);
                return Err(RuntimeError::Inference(format!(
                    "{message}: {:?}",
                    response.choices
                )));
            }
            Response::InternalError(err) | Response::ValidationError(err) => {
                let error = RuntimeError::Inference(err.to_string());
                emit_error(error);
                return Err(RuntimeError::Inference(err.to_string()));
            }
            Response::CompletionModelError(message, _) => {
                let error = RuntimeError::Inference(format!(
                    "unexpected completion error while generating chat: {message}"
                ));
                emit_error(error);
                return Err(RuntimeError::Inference(format!(
                    "unexpected completion error while generating chat: {message}"
                )));
            }
            Response::CompletionDone(_)
            | Response::CompletionChunk(_)
            | Response::ImageGeneration(_)
            | Response::Speech { .. }
            | Response::Raw { .. }
            | Response::Embeddings { .. } => {
                let error =
                    RuntimeError::Inference("unexpected non-chat streaming response".into());
                emit_error(error);
                return Err(RuntimeError::Inference(
                    "unexpected non-chat streaming response".into(),
                ));
            }
        }
    }

    if streamed_content.is_empty() {
        let error = RuntimeError::Inference("stream ended before completion".into());
        emit_error(error);
        Err(RuntimeError::Inference(
            "stream ended before completion".into(),
        ))
    } else {
        tracing::warn!(
            config_id = %config.id,
            repo_id = %config.model.repo_id(),
            chunk_count,
            response_chars = streamed_content.chars().count(),
            "generation stream ended before final usage response"
        );
        emit_complete();
        Ok(streamed_content)
    }
}

fn send_status(status: &Option<mpsc::UnboundedSender<RuntimeStatus>>, value: RuntimeStatus) {
    if let Some(status) = status {
        let _ = status.send(value);
    }
}

fn model_load_lock(repo_id: &'static str) -> Arc<Mutex<()>> {
    static LOAD_LOCKS: OnceLock<Mutex<HashMap<&'static str, Arc<Mutex<()>>>>> = OnceLock::new();

    let mut locks = LOAD_LOCKS
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
        .expect("model load lock registry poisoned");

    Arc::clone(
        locks
            .entry(repo_id)
            .or_insert_with(|| Arc::new(Mutex::new(()))),
    )
}

async fn load_model(config: &RuntimeConfig) -> Result<MistralRuntimeModel, RuntimeError> {
    let repo_id = config.model.repo_id();
    let load_started = Instant::now();
    tracing::info!(
        config_id = %config.id,
        repo_id = %repo_id,
        backend = runtime_backend(),
        load_mode = ?config.model.load_mode(),
        quantization = ?config.model.quantization(),
        "loading runtime model"
    );
    tracing::info!(
        config_id = %config.id,
        repo_id = %repo_id,
        "waiting for model load lock"
    );
    let lock = model_load_lock(repo_id);
    let _guard = lock.lock().expect("model load lock poisoned");
    tracing::info!(
        config_id = %config.id,
        repo_id = %repo_id,
        elapsed_ms = load_started.elapsed().as_millis(),
        "model load lock acquired"
    );
    match config.model.load_mode() {
        LoadMode::AutoIsq => {
            tracing::info!(
                config_id = %config.id,
                repo_id = %repo_id,
                "building auto-isq model builder"
            );
            let builder = ModelBuilder::new(repo_id);
            let builder = config.model.quantization().apply(builder);

            tracing::info!(
                config_id = %config.id,
                repo_id = %repo_id,
                "starting model build"
            );
            builder
                .build()
                .await
                .map_err(|err| RuntimeError::ModelBuild(err.to_string()))
                .map(|model| {
                    tracing::info!(
                        config_id = %config.id,
                        repo_id = %repo_id,
                        elapsed_ms = load_started.elapsed().as_millis(),
                        "model build completed"
                    );
                    model
                })
        }
        LoadMode::Gguf {
            quantized_model_id,
            quantized_filenames,
        } => {
            tracing::info!(
                config_id = %config.id,
                repo_id = %repo_id,
                quantized_model_id = %quantized_model_id,
                "building gguf model builder"
            );
            let builder = GgufModelBuilder::new(
                quantized_model_id,
                quantized_filenames
                    .iter()
                    .map(|file| (*file).to_string())
                    .collect(),
            );

            tracing::info!(
                config_id = %config.id,
                repo_id = %repo_id,
                quantized_model_id = %quantized_model_id,
                "starting gguf model build"
            );
            builder
                .build()
                .await
                .map_err(|err| RuntimeError::ModelBuild(err.to_string()))
                .map(|model| {
                    tracing::info!(
                        config_id = %config.id,
                        repo_id = %repo_id,
                        quantized_model_id = %quantized_model_id,
                        elapsed_ms = load_started.elapsed().as_millis(),
                        "gguf model build completed"
                    );
                    model
                })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mistralrs::{RequestLike, TextMessageRole, TextMessages};
    use std::time::Duration;

    fn test_request(content: &str) -> RequestBuilder {
        TextMessages::new()
            .add_message(TextMessageRole::User, content)
            .into()
    }

    fn insert_connection(
        connections: &Arc<Mutex<HashMap<ConnectionId, ConnectionEntry>>>,
        connection_id: ConnectionId,
        config_id: &str,
    ) -> mpsc::UnboundedSender<ConnectionRequest> {
        let (tx, rx) = mpsc::unbounded_channel();
        connections
            .lock()
            .expect("connection registry poisoned")
            .insert(
                connection_id,
                ConnectionEntry {
                    config_id: config_id.to_string(),
                    rx,
                },
            );
        tx
    }

    fn spawn_test_forwarder(
        connections: Arc<Mutex<HashMap<ConnectionId, ConnectionEntry>>>,
    ) -> (
        mpsc::Sender<ForwarderSignal>,
        channel::Receiver<RuntimeRequest>,
        thread::JoinHandle<()>,
    ) {
        let (signal_tx, signal_rx) = mpsc::channel(16);
        let (router_tx, router_rx) = channel::unbounded();
        let join = thread::spawn(move || {
            connection_forwarder_loop(connections, signal_rx, router_tx);
        });
        (signal_tx, router_rx, join)
    }

    #[test]
    fn defaults_include_named_configs() {
        let registry = RuntimeRegistry::defaults();
        let ids: Vec<_> = registry
            .list()
            .iter()
            .map(|config| config.id.as_str())
            .collect();

        assert!(ids.contains(&"smollm2-360m"));
        assert!(ids.contains(&"smollm2-360m-isq4"));
        assert!(ids.contains(&"smollm2-360m-q4"));
        assert!(ids.contains(&"smollm2"));
        assert!(ids.contains(&"smollm2-q4"));
        assert!(ids.contains(&"smollm3-q4"));
        assert!(ids.contains(&"smollm3"));
        assert!(ids.contains(&"qwen3.5"));
        assert!(ids.contains(&"qwen3.5-q2k"));
    }

    #[test]
    fn quantized_defaults_map_to_expected_sources() {
        let registry = RuntimeRegistry::defaults();

        let smollm2 = registry.get("smollm2-q4").expect("smollm2-q4 registered");
        assert_eq!(smollm2.model.repo_id(), "nakue/SmolLM2-1.7B-W4A16-instruct");
        assert_eq!(smollm2.model.quantization(), Quantization::Unquantized);

        let smollm3 = registry.get("smollm3-q4").expect("smollm3-q4 registered");
        assert_eq!(smollm3.model.repo_id(), "AINovice2005/quantized-SmolLM3-3B");
        assert_eq!(smollm3.model.quantization(), Quantization::Unquantized);

        let qwen = registry.get("qwen3.5-q2k").expect("qwen3.5-q2k registered");
        assert_eq!(qwen.model.repo_id(), "unsloth/Qwen3-4B-GGUF");
        assert_eq!(qwen.model.quantization(), Quantization::Unquantized);
        assert!(matches!(
            qwen.model.load_mode(),
            LoadMode::Gguf {
                quantized_model_id: "unsloth/Qwen3-4B-GGUF",
                ..
            }
        ));

        let smollm2_360m = registry
            .get("smollm2-360m-isq4")
            .expect("smollm2-360m-isq4 registered");
        assert_eq!(
            smollm2_360m.model.repo_id(),
            "HuggingFaceTB/SmolLM2-360M-Instruct"
        );
        assert_eq!(smollm2_360m.model.quantization(), Quantization::AutoIsq4);

        let smollm2_360m_q4 = registry
            .get("smollm2-360m-q4")
            .expect("smollm2-360m-q4 registered");
        assert_eq!(
            smollm2_360m_q4.model.repo_id(),
            "HuggingFaceTB/SmolLM2-360M-Instruct"
        );
        assert_eq!(smollm2_360m_q4.model.quantization(), Quantization::AutoIsq4);
    }

    #[test]
    fn typed_messages_convert_into_request_builder() {
        let request: RequestBuilder = TextMessages::new()
            .add_message(TextMessageRole::System, "rules")
            .add_message(TextMessageRole::User, "hello")
            .into();

        assert_eq!(request.messages_ref().len(), 2);
    }

    #[test]
    fn connection_tracker_loads_on_first_connection_only() {
        let mut tracker = ConnectionTracker::default();

        assert_eq!(
            tracker.connect(ConnectionId(1), "smollm3".into()),
            Some(ConnectionLifecycleEvent::Load("smollm3".into()))
        );
        assert_eq!(tracker.connect(ConnectionId(2), "smollm3".into()), None);
    }

    #[test]
    fn connection_tracker_unloads_on_final_disconnect_only() {
        let mut tracker = ConnectionTracker::default();
        tracker.connect(ConnectionId(1), "smollm3".into());
        tracker.connect(ConnectionId(2), "smollm3".into());

        assert_eq!(tracker.disconnect(ConnectionId(1)), None);
        assert_eq!(
            tracker.disconnect(ConnectionId(2)),
            Some(ConnectionLifecycleEvent::Unload("smollm3".into()))
        );
    }

    #[test]
    fn connection_tracker_duplicate_disconnect_is_noop() {
        let mut tracker = ConnectionTracker::default();
        tracker.connect(ConnectionId(1), "smollm3".into());

        assert_eq!(
            tracker.disconnect(ConnectionId(1)),
            Some(ConnectionLifecycleEvent::Unload("smollm3".into()))
        );
        assert_eq!(tracker.disconnect(ConnectionId(1)), None);
    }

    #[tokio::test]
    async fn forwarder_ready_drains_only_signaled_connection() {
        let connections = Arc::new(Mutex::new(HashMap::new()));
        let first_tx = insert_connection(&connections, ConnectionId(1), "smollm3");
        let second_tx = insert_connection(&connections, ConnectionId(2), "qwen3.5");
        let (signal_tx, router_rx, join) = spawn_test_forwarder(Arc::clone(&connections));

        let (first_response_tx, _first_response_rx) = mpsc::unbounded_channel();
        first_tx
            .send(ConnectionRequest::Generate {
                request: test_request("one"),
                status: None,
                response: first_response_tx,
                sequence_number: 1,
            })
            .unwrap();
        let (second_response_tx, _second_response_rx) = mpsc::unbounded_channel();
        second_tx
            .send(ConnectionRequest::Generate {
                request: test_request("two"),
                status: None,
                response: second_response_tx,
                sequence_number: 1,
            })
            .unwrap();

        signal_tx
            .send(ForwarderSignal::Ready(ConnectionId(1)))
            .await
            .unwrap();

        match router_rx.recv_timeout(Duration::from_secs(1)).unwrap() {
            RuntimeRequest::Generate { config_id, .. } => {
                assert_eq!(config_id, "smollm3");
            }
            _ => panic!("expected generate request"),
        }
        assert!(router_rx.try_recv().is_err());

        signal_tx.send(ForwarderSignal::Shutdown).await.unwrap();
        join.join().unwrap();
    }

    #[tokio::test]
    async fn forwarder_preserves_config_for_independent_connections() {
        let connections = Arc::new(Mutex::new(HashMap::new()));
        let first_tx = insert_connection(&connections, ConnectionId(1), "smollm3");
        let second_tx = insert_connection(&connections, ConnectionId(2), "qwen3.5");
        let (signal_tx, router_rx, join) = spawn_test_forwarder(Arc::clone(&connections));

        let (first_response_tx, _first_response_rx) = mpsc::unbounded_channel();
        first_tx
            .send(ConnectionRequest::Generate {
                request: test_request("one"),
                status: None,
                response: first_response_tx,
                sequence_number: 1,
            })
            .unwrap();
        let (second_response_tx, _second_response_rx) = mpsc::unbounded_channel();
        second_tx
            .send(ConnectionRequest::Generate {
                request: test_request("two"),
                status: None,
                response: second_response_tx,
                sequence_number: 1,
            })
            .unwrap();

        signal_tx
            .send(ForwarderSignal::Ready(ConnectionId(2)))
            .await
            .unwrap();
        signal_tx
            .send(ForwarderSignal::Ready(ConnectionId(1)))
            .await
            .unwrap();

        let mut configs = Vec::new();
        for _ in 0..2 {
            match router_rx.recv_timeout(Duration::from_secs(1)).unwrap() {
                RuntimeRequest::Generate { config_id, .. } => configs.push(config_id),
                _ => panic!("expected generate request"),
            }
        }

        assert_eq!(configs, vec!["qwen3.5", "smollm3"]);

        signal_tx.send(ForwarderSignal::Shutdown).await.unwrap();
        join.join().unwrap();
    }

    #[tokio::test]
    async fn closed_agent_receiver_forwards_one_disconnect() {
        let connections = Arc::new(Mutex::new(HashMap::new()));
        let tx = insert_connection(&connections, ConnectionId(1), "smollm3");
        let (signal_tx, router_rx, join) = spawn_test_forwarder(Arc::clone(&connections));
        drop(tx);

        signal_tx
            .send(ForwarderSignal::Ready(ConnectionId(1)))
            .await
            .unwrap();
        signal_tx
            .send(ForwarderSignal::Ready(ConnectionId(1)))
            .await
            .unwrap();

        match router_rx.recv_timeout(Duration::from_secs(1)).unwrap() {
            RuntimeRequest::Disconnect {
                connection_id,
                reply,
            } => {
                assert_eq!(connection_id, ConnectionId(1));
                assert!(reply.is_none());
            }
            _ => panic!("expected disconnect request"),
        }
        assert!(router_rx.try_recv().is_err());

        signal_tx.send(ForwarderSignal::Shutdown).await.unwrap();
        join.join().unwrap();
    }

    #[tokio::test]
    async fn explicit_end_connection_removes_connection_and_blocks_later_sends() {
        let connections = Arc::new(Mutex::new(HashMap::new()));
        let tx = insert_connection(&connections, ConnectionId(1), "smollm3");
        let (signal_tx, router_rx, join) = spawn_test_forwarder(Arc::clone(&connections));
        let connection = RuntimeConnection {
            id: ConnectionId(1),
            config_id: "smollm3".into(),
            tx,
            signal: signal_tx.clone(),
            closed: Arc::new(AtomicBool::new(false)),
            next_sequence_number: Arc::new(AtomicU64::new(1)),
        };

        let end = connection.end_connection();
        tokio::pin!(end);

        tokio::select! {
            result = &mut end => panic!("end_connection returned before ack: {:?}", result),
            _ = tokio::time::sleep(Duration::from_millis(20)) => {}
        }

        let reply = match router_rx.recv_timeout(Duration::from_secs(1)).unwrap() {
            RuntimeRequest::Disconnect {
                connection_id,
                reply,
            } => {
                assert_eq!(connection_id, ConnectionId(1));
                reply.expect("explicit close should carry reply")
            }
            _ => panic!("expected disconnect request"),
        };
        reply.send(Ok(())).unwrap();
        end.await.unwrap();

        assert!(connections.lock().unwrap().get(&ConnectionId(1)).is_none());
        assert!(matches!(
            connection.send(test_request("after close")).await,
            Err(RuntimeError::ConnectionClosed)
        ));

        signal_tx.send(ForwarderSignal::Shutdown).await.unwrap();
        join.join().unwrap();
    }

    #[tokio::test]
    async fn response_stream_collects_chunk_events_in_order() {
        let (tx, rx) = mpsc::unbounded_channel();
        let stream = RuntimeResponseStream::new(ConnectionId(7), 3, rx);

        tx.send(RuntimeResponseEvent::Chunk {
            connection_id: ConnectionId(7),
            sequence_number: 3,
            content: "hello ".into(),
        })
        .unwrap();
        tx.send(RuntimeResponseEvent::Chunk {
            connection_id: ConnectionId(7),
            sequence_number: 3,
            content: "world".into(),
        })
        .unwrap();
        tx.send(RuntimeResponseEvent::Complete {
            connection_id: ConnectionId(7),
            sequence_number: 3,
        })
        .unwrap();

        assert_eq!(stream.collect_string().await.unwrap(), "hello world");
    }

    #[tokio::test]
    async fn response_stream_keeps_connection_and_sequence_metadata() {
        let (left_tx, left_rx) = mpsc::unbounded_channel();
        let (right_tx, right_rx) = mpsc::unbounded_channel();
        let mut left = RuntimeResponseStream::new(ConnectionId(1), 11, left_rx);
        let mut right = RuntimeResponseStream::new(ConnectionId(2), 22, right_rx);

        left_tx
            .send(RuntimeResponseEvent::Chunk {
                connection_id: ConnectionId(1),
                sequence_number: 11,
                content: "left".into(),
            })
            .unwrap();
        left_tx
            .send(RuntimeResponseEvent::Complete {
                connection_id: ConnectionId(1),
                sequence_number: 11,
            })
            .unwrap();

        right_tx
            .send(RuntimeResponseEvent::Chunk {
                connection_id: ConnectionId(2),
                sequence_number: 22,
                content: "right".into(),
            })
            .unwrap();
        right_tx
            .send(RuntimeResponseEvent::Complete {
                connection_id: ConnectionId(2),
                sequence_number: 22,
            })
            .unwrap();

        match left.next().await {
            Some(RuntimeResponseEvent::Chunk {
                connection_id,
                sequence_number,
                content,
            }) => {
                assert_eq!(connection_id, ConnectionId(1));
                assert_eq!(sequence_number, 11);
                assert_eq!(content, "left");
            }
            other => panic!("unexpected left event: {other:?}"),
        }
        assert!(matches!(
            left.next().await,
            Some(RuntimeResponseEvent::Complete {
                connection_id: ConnectionId(1),
                sequence_number: 11
            })
        ));

        match right.next().await {
            Some(RuntimeResponseEvent::Chunk {
                connection_id,
                sequence_number,
                content,
            }) => {
                assert_eq!(connection_id, ConnectionId(2));
                assert_eq!(sequence_number, 22);
                assert_eq!(content, "right");
            }
            other => panic!("unexpected right event: {other:?}"),
        }
        assert!(matches!(
            right.next().await,
            Some(RuntimeResponseEvent::Complete {
                connection_id: ConnectionId(2),
                sequence_number: 22
            })
        ));
    }

    #[tokio::test]
    async fn open_connection_waits_for_router_ack() {
        let registry = Arc::new(RuntimeRegistry::defaults());
        let (tx, rx) = channel::unbounded();
        let (forwarder_tx, _forwarder_rx) = mpsc::channel(16);
        let runtime = Runtime {
            registry,
            tx,
            next_connection_id: AtomicU64::new(1),
            connections: Arc::new(Mutex::new(HashMap::new())),
            forwarder_tx,
            router: None,
            forwarder: None,
            workers: Vec::new(),
        };

        let open = runtime.open_connection("qwen3.5-q2k");
        tokio::pin!(open);

        tokio::select! {
            result = &mut open => panic!(
                "open_connection returned before ack: {:?}",
                result.map(|connection| connection.id())
            ),
            _ = tokio::time::sleep(Duration::from_millis(20)) => {}
        }

        let reply = match rx.recv_timeout(Duration::from_secs(1)).unwrap() {
            RuntimeRequest::Connect {
                connection_id,
                config_id,
                reply,
            } => {
                assert_eq!(connection_id, ConnectionId(1));
                assert_eq!(config_id, "qwen3.5-q2k");
                reply
            }
            _ => panic!("expected connect request"),
        };
        reply.send(Ok(())).unwrap();

        let connection = open.await.unwrap();
        assert_eq!(connection.id(), ConnectionId(1));
    }
}
