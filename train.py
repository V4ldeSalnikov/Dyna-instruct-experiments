from transformers import AutoTokenizer
from trl import DatasetMixtureConfig, ModelConfig, SFTConfig, SFTTrainer, TrlParser, get_dataset


def main() -> None:
    parser = TrlParser((SFTConfig, ModelConfig, DatasetMixtureConfig))
    training, model, data = parser.parse_args_and_config()

    training.model_init_kwargs = {
        "dtype": model.dtype,
        "revision": model.model_revision,
        "trust_remote_code": training.trust_remote_code,
        "attn_implementation": model.attn_implementation,
    }
    dataset = get_dataset(data)
    tokenizer = AutoTokenizer.from_pretrained(model.model_name_or_path)
    trainer = SFTTrainer(
        model=model.model_name_or_path,
        args=training,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("test"),
        processing_class=tokenizer,
    )
    trainer.train(resume_from_checkpoint=training.resume_from_checkpoint)
    trainer.save_model()


if __name__ == "__main__":
    main()
