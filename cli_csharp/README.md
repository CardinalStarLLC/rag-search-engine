This is a small C# translation of the Python `keyword_search_cli.py`.

Requirements:

- .NET 7 SDK (or adjust the target framework in the .csproj)

Build & run examples (from repository root):

```bash
# build (formats: json|proto|both) - default is both
dotnet run --project cli_csharp/RagSearchEngine.csproj -- build
dotnet run --project cli_csharp/RagSearchEngine.csproj -- build proto
dotnet run --project cli_csharp/RagSearchEngine.csproj -- build json
# then search (loading defaults to protobuf; override with --load-format=json|proto|both)
dotnet run --project cli_csharp/RagSearchEngine.csproj -- search "star wars"
dotnet run --project cli_csharp/RagSearchEngine.csproj -- --load-format=json search "star wars"
# tf/idf/tfidf
dotnet run --project cli_csharp/RagSearchEngine.csproj -- tf 136 star
dotnet run --project cli_csharp/RagSearchEngine.csproj -- idf star
dotnet run --project cli_csharp/RagSearchEngine.csproj -- tfidf 136 star
```

Notes:

- The C# version uses a simple tokenizer and stopwords from `data/stopwords.txt`.
- No stemming is applied in this translation. If you need stemming, consider adding a Porter stemmer library or porting the algorithm.
- Stemming: this project now references the `Porter2Stemmer` NuGet package and attempts to use it at runtime. If the package is not present, a small fallback stemmer is used.

To restore packages and run:

```bash
dotnet restore cli_csharp
dotnet run --project cli_csharp/RagSearchEngine.csproj -- build
```
