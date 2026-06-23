"""Enable ``python -m mini_agent ...`` without installing the package."""

from .cli import main

raise SystemExit(main())
