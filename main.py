from typing import List, Dict, Optional
from pathlib import Path

import typer

from project.tools.train import train as _train

app = typer.Typer()

@app.command(name='train')
def train(config: Path = typer.Argument(exists=True),
          command_opts: Optional[List[str]] = typer.Option(None)):
    typer.echo(config)
    typer.echo(command_opts)


if __name__ == '__main__':
    app()