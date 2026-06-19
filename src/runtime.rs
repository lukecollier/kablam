use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;
use std::thread;

use crossbeam::channel;
use mistralrs::{IsqBits, Model as MistralRuntimeModel, ModelBuilder, RequestBuilder};
use tokio::sync::oneshot;

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

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Model {
    SmolLM3(Quantization),
    Qwen35(Quantization),
}

impl Model {
    pub fn repo_id(&self) -> &'static str {
        match self {
            Self::SmolLM3(_) => "HuggingFaceTB/SmolLM3-3B",
            Self::Qwen35(_) => "Qwen/Qwen3-4B",
        }
    }

    pub fn quantization(&self) -> Quantization {
        match self {
            Self::SmolLM3(quantization) | Self::Qwen35(quantization) => *quantization,
        }
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
            RuntimeConfig::new("smollm3", Model::SmolLM3(Quantization::AutoIsq4)),
            RuntimeConfig::new("smollm3-fp16", Model::SmolLM3(Quantization::Unquantized)),
            RuntimeConfig::new("qwen3.5", Model::Qwen35(Quantization::AutoIsq4)),
            RuntimeConfig::new("qwen3.5-fp16", Model::Qwen35(Quantization::Unquantized)),
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
    Generate {
        config_id: String,
        request: RequestBuilder,
        reply: oneshot::Sender<Result<String, RuntimeError>>,
    },
    Shutdown,
}

#[derive(Debug)]
pub enum RuntimeError {
    UnknownConfig(String),
    ModelBuild(String),
    Inference(String),
    ChannelClosed,
}

impl fmt::Display for RuntimeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownConfig(id) => write!(f, "unknown runtime config: {id}"),
            Self::ModelBuild(message) => write!(f, "failed to build model: {message}"),
            Self::Inference(message) => write!(f, "inference failed: {message}"),
            Self::ChannelClosed => f.write_str("runtime request channel closed"),
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
    router: Option<thread::JoinHandle<()>>,
    workers: Vec<thread::JoinHandle<()>>,
}

impl Runtime {
    fn spawn(registry: RuntimeRegistry, worker_count: Option<usize>) -> Self {
        let registry = Arc::new(registry);
        let (tx, rx) = channel::unbounded();

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

        Self {
            registry,
            tx,
            router: Some(router),
            workers: worker_layout
                .into_iter()
                .map(|worker| worker.join)
                .collect(),
        }
    }

    pub fn list_configs(&self) -> &[RuntimeConfig] {
        self.registry.list()
    }

    pub fn request_sender(&self) -> channel::Sender<RuntimeRequest> {
        self.tx.clone()
    }

    pub async fn generate(
        &self,
        config_id: impl Into<String>,
        request: impl Into<RequestBuilder>,
    ) -> Result<String, RuntimeError> {
        let (reply_tx, reply_rx) = oneshot::channel();
        self.tx
            .send(RuntimeRequest::Generate {
                config_id: config_id.into(),
                request: request.into(),
                reply: reply_tx,
            })
            .map_err(|_| RuntimeError::ChannelClosed)?;

        reply_rx.await.map_err(|_| RuntimeError::ChannelClosed)?
    }
}

impl Drop for Runtime {
    fn drop(&mut self) {
        let _ = self.tx.send(RuntimeRequest::Shutdown);
        if let Some(router) = self.router.take() {
            let _ = router.join();
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

enum WorkerRequest {
    Generate {
        config: RuntimeConfig,
        request: RequestBuilder,
        reply: oneshot::Sender<Result<String, RuntimeError>>,
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

fn router_loop(
    registry: Arc<RuntimeRegistry>,
    rx: channel::Receiver<RuntimeRequest>,
    workers: Vec<WorkerRoute>,
) {
    let mut worker_map = HashMap::new();
    for worker in workers {
        worker_map.insert(worker.config_id, worker.tx);
    }

    while let Ok(request) = rx.recv() {
        match request {
            RuntimeRequest::Generate {
                config_id,
                request,
                reply,
            } => match registry.get(&config_id) {
                Some(config) => match worker_map.get(&config.id) {
                    Some(worker_tx) => {
                        let worker_request = WorkerRequest::Generate {
                            config: config.clone(),
                            request,
                            reply,
                        };

                        if let Err(err) = worker_tx.send(worker_request) {
                            match err.into_inner() {
                                WorkerRequest::Generate { reply, .. } => {
                                    let _ = reply.send(Err(RuntimeError::ChannelClosed));
                                }
                                WorkerRequest::Shutdown => {}
                            }
                        }
                    }
                    None => {
                        let _ = reply.send(Err(RuntimeError::UnknownConfig(config_id)));
                    }
                },
                None => {
                    let _ = reply.send(Err(RuntimeError::UnknownConfig(config_id)));
                }
            },
            RuntimeRequest::Shutdown => {
                for worker in worker_map.values() {
                    let _ = worker.send(WorkerRequest::Shutdown);
                }
                break;
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
            WorkerRequest::Generate {
                config,
                request,
                reply,
            } => {
                let response = runtime.block_on(async {
                    let config = registry
                        .get(&config.id)
                        .ok_or_else(|| RuntimeError::UnknownConfig(config.id.clone()))?;

                    if !cache.contains_key(&config.id) {
                        let model = load_model(config).await?;
                        cache.insert(config.id.clone(), model);
                    }

                    let model = cache.get(&config.id).expect("model cached after load");
                    let response = model
                        .send_chat_request(request)
                        .await
                        .map_err(|err| RuntimeError::Inference(err.to_string()))?;

                    response
                        .choices
                        .into_iter()
                        .next()
                        .and_then(|choice| choice.message.content)
                        .ok_or_else(|| RuntimeError::Inference("missing assistant content".into()))
                });

                let _ = reply.send(response);
            }
            WorkerRequest::Shutdown => break,
        }
    }
}

async fn load_model(config: &RuntimeConfig) -> Result<MistralRuntimeModel, RuntimeError> {
    let builder = ModelBuilder::new(config.model.repo_id());
    let builder = config.model.quantization().apply(builder);

    builder
        .build()
        .await
        .map_err(|err| RuntimeError::ModelBuild(err.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use mistralrs::{RequestLike, TextMessageRole, TextMessages};

    #[test]
    fn defaults_include_named_configs() {
        let registry = RuntimeRegistry::defaults();
        let ids: Vec<_> = registry
            .list()
            .iter()
            .map(|config| config.id.as_str())
            .collect();

        assert!(ids.contains(&"smollm3"));
        assert!(ids.contains(&"qwen3.5"));
    }

    #[test]
    fn typed_messages_convert_into_request_builder() {
        let request: RequestBuilder = TextMessages::new()
            .add_message(TextMessageRole::System, "rules")
            .add_message(TextMessageRole::User, "hello")
            .into();

        assert_eq!(request.messages_ref().len(), 2);
    }
}
