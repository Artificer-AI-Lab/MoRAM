# Description: Factory for models (MoRAM entrypoint for this repository)


def get_model_loader(args):
    if args.method == "moram":
        from models.moram import get_moram

        return get_moram
    raise ValueError(f"Unknown method: {args.method}")


def get_trainer(args):
    if args.method == "moram":
        from models.moram import finetune

        return finetune
    raise ValueError(f"Unknown method: {args.method}")
