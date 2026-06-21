"""
Run with: uv run python data_pipeline/verify_setup.py
Checks that all core dependencies and services are reachable.
"""

import sys
from rich.console import Console
from rich.table import Table

console = Console()


def check(label: str, fn) -> bool:
    try:
        fn()
        return True
    except Exception as e:
        console.print(f"  [red]✗[/red] {e}")
        return False


def main():
    table = Table(title="Environment Check", show_header=True)
    table.add_column("Component", style="cyan")
    table.add_column("Status", justify="center")

    results = {}

    # Python version
    ok = sys.version_info >= (3, 12)
    results["Python ≥ 3.12"] = ok
    table.add_row("Python ≥ 3.12", "[green]✓[/green]" if ok else "[red]✗[/red]")

    # Core imports
    for pkg, import_str in [
        ("httpx", "import httpx"),
        ("beautifulsoup4", "from bs4 import BeautifulSoup"),
        ("spaCy", "import spacy"),
        ("sentence-transformers", "from sentence_transformers import SentenceTransformer"),
        ("langchain", "import langchain"),
        ("qdrant-client", "from qdrant_client import QdrantClient"),
    ]:
        ok = check(pkg, lambda s=import_str: exec(s))
        results[pkg] = ok
        table.add_row(pkg, "[green]✓[/green]" if ok else "[red]✗[/red]")

    # spaCy model
    def check_spacy():
        import spacy
        spacy.load("en_core_web_sm")

    ok = check("spaCy en_core_web_sm", check_spacy)
    results["spaCy model"] = ok
    table.add_row("spaCy en_core_web_sm", "[green]✓[/green]" if ok else "[red]✗[/red]")

    # Qdrant connectivity
    def check_qdrant():
        from qdrant_client import QdrantClient
        client = QdrantClient(host="localhost", port=6333)
        client.get_collections()

    ok = check("Qdrant (localhost:6333)", check_qdrant)
    results["Qdrant"] = ok
    table.add_row("Qdrant (localhost:6333)", "[green]✓[/green]" if ok else "[red]✗[/red]")

    console.print(table)

    failures = [k for k, v in results.items() if not v]
    if failures:
        console.print(f"\n[red]Failed:[/red] {', '.join(failures)}")
        console.print("Fix the above before proceeding to the wiki scraper.")
        sys.exit(1)
    else:
        console.print("\n[green]All checks passed — ready to build the pipeline![/green]")


if __name__ == "__main__":
    main()