mod runtime;

use mistralrs::{TextMessageRole, TextMessages};
use runtime::RuntimeBuilder;

#[tokio::main(flavor = "multi_thread")]
async fn main() {
    let runtime = RuntimeBuilder::new().build();

    println!("registered runtimes:");
    for config in runtime.list_configs() {
        println!("- {}", config.id);
    }

    let messages = TextMessages::new()
        .add_message(TextMessageRole::System, "You are a concise assistant.")
        .add_message(TextMessageRole::User, "Reply with a single short sentence.");

    match runtime.generate("smollm3", messages).await {
        Ok(response) => println!("{response}"),
        Err(err) => eprintln!("runtime error: {err}"),
    }
}
