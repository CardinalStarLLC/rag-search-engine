using System;
using System.IO;
using System.Text.Json;
using System.Text.RegularExpressions;
using System.Collections.Generic;
using System.Linq;
using Porter2Stemmer;
using ProtoBuf;

class InvertedIndex
{
    private readonly Dictionary<string, HashSet<int>> index = new();
    private readonly Dictionary<int, string> docmap = new();
    private readonly Dictionary<int, Dictionary<string, int>> termFrequencies = new();
    private readonly HashSet<string> stopWords = new();

    public void LoadStopWords(string path)
    {
        if (!File.Exists(path)) return;
        foreach (var line in File.ReadAllLines(path))
        {
            var w = line.Trim();
            if (w.Length > 0) stopWords.Add(w.ToLower());
        }
    }

    public List<string> Tokenize(string text)
    {
        if (string.IsNullOrEmpty(text)) return new List<string>();
        var cleaned = Regex.Replace(text.ToLowerInvariant(), "[\u0000-\u002F\u003A-\u0040\u005B-\u0060\u007B-\u007F]", " ");
        var parts = cleaned.Split(new[] { ' ', '\t', '\r', '\n' }, StringSplitOptions.RemoveEmptyEntries);
        var tokens = new List<string>();
        foreach (var p in parts)
        {
            if (stopWords.Contains(p)) continue;
            // apply stemming (Porter2) if available, otherwise use token as-is
            tokens.Add(StemWord(p));
        }
        return tokens;
    }

    private string Normalize(string term)
    {
        if (string.IsNullOrEmpty(term)) return term;
        var lowered = term.ToLowerInvariant();
        return StemWord(lowered);
    }

    private string StemWord(string word)
    {
        if (string.IsNullOrEmpty(word)) return word;

        // Use Porter2Stemmer directly (compile-time reference)
        try
        {
            return new EnglishPorter2Stemmer().Stem(word).Value;
        }
        catch
        {
            // fallback to small heuristic below when stemming fails
        }

        // fallback: naive suffix stripping for common endings
        if (word.Length > 4)
        {
            if (word.EndsWith("ing") || word.EndsWith("ed"))
                return word.Substring(0, word.Length - 3).TrimEnd('e');
        }
        if (word.EndsWith("ly") && word.Length > 3) return word.Substring(0, word.Length - 2);
        if (word.EndsWith("s") && word.Length > 2) return word.Substring(0, word.Length - 1);
        return word;
    }

    public void AddDocument(int docId, string text)
    {
        docmap[docId] = text;
        var tokens = Tokenize(text);
        var tf = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);
        foreach (var t in tokens)
        {
            if (!index.TryGetValue(t, out var set))
            {
                set = new HashSet<int>();
                index[t] = set;
            }
            set.Add(docId);
            tf[t] = tf.TryGetValue(t, out var c) ? c + 1 : 1;
        }
        termFrequencies[docId] = tf;
    }

    public List<int> GetDocuments(string term)
    {
        if (term == null) return new List<int>();
        var key = Normalize(term);
        if (!index.TryGetValue(key, out var set)) return new List<int>();
        return set.OrderBy(x => x).ToList();
    }

    public int GetTf(int docId, string term)
    {
        if (!termFrequencies.TryGetValue(docId, out var tf)) return 0;
        var key = Normalize(term);
        return tf.TryGetValue(key, out var count) ? count : 0;
    }

    public double GetIdf(string term)
    {
        var docs = GetDocuments(term);
        return Math.Log((docmap.Count + 1.0) / (docs.Count + 1.0));
    }

    public void BuildFromMoviesJson(string path)
    {
        if (!File.Exists(path)) throw new FileNotFoundException(path);
        using var doc = JsonDocument.Parse(File.ReadAllText(path));
        if (!doc.RootElement.TryGetProperty("movies", out var movies)) return;
        foreach (var m in movies.EnumerateArray())
        {
            int id = 0;
            string title = "";
            string description = "";
            if (m.TryGetProperty("id", out var idEl)) id = idEl.GetInt32();
            if (m.TryGetProperty("title", out var tEl)) title = tEl.GetString() ?? "";
            if (m.TryGetProperty("description", out var dEl)) description = dEl.GetString() ?? "";
            var concat = (title + " " + description).Trim();
            AddDocument(id, concat);
        }
    }

    public void Save(string cacheDir, bool writeJson = true, bool writeProto = true)
    {
        try
        {
            if (writeJson)
            {
                File.WriteAllText(Path.Combine(cacheDir, "docmap.json"), JsonSerializer.Serialize(docmap));
                File.WriteAllText(Path.Combine(cacheDir, "index.json"), JsonSerializer.Serialize(index.ToDictionary(kv => kv.Key, kv => kv.Value.ToList())));
                File.WriteAllText(Path.Combine(cacheDir, "term_frequencies.json"), JsonSerializer.Serialize(termFrequencies));
            }
            if (writeProto)
            {
                // Also write protobuf binary for faster load
                SaveProtoIndex(Path.Combine(cacheDir, "index.bin"));
                SaveProtoDocMap(Path.Combine(cacheDir, "docmap.bin"));
                SaveProtoTermFrequencies(Path.Combine(cacheDir, "term_frequencies.bin"));
            }
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine(ex.Message);
        }
    }

    public void Load(string cacheDir, string format = "proto")
    {
        try
        {
            // file paths
            var protoIndex = Path.Combine(cacheDir, "index.bin");
            var protoDocmap = Path.Combine(cacheDir, "docmap.bin");
            var protoTf = Path.Combine(cacheDir, "term_frequencies.bin");
            var idxJson = Path.Combine(cacheDir, "index.json");
            var dmJson = Path.Combine(cacheDir, "docmap.json");
            var tfJson = Path.Combine(cacheDir, "term_frequencies.json");

            if (format == "proto")
            {
                if (File.Exists(protoIndex) && File.Exists(protoDocmap) && File.Exists(protoTf))
                {
                    LoadProtoIndex(protoIndex);
                    LoadProtoDocMap(protoDocmap);
                    LoadProtoTermFrequencies(protoTf);

                    return;
                }
                // fall back to json if proto not available
            }
            if (format == "json")
            {
                if (File.Exists(idxJson))
                {
                    var idxText = File.ReadAllText(idxJson);
                    var loaded = JsonSerializer.Deserialize<Dictionary<string, List<int>>>(idxText);
                    if (loaded != null) foreach (var kv in loaded) index[kv.Key] = new HashSet<int>(kv.Value);
                }
                if (File.Exists(dmJson))
                {
                    var dmText = File.ReadAllText(dmJson);
                    var loaded = JsonSerializer.Deserialize<Dictionary<int, string>>(dmText);
                    if (loaded != null) foreach (var kv in loaded) docmap[kv.Key] = kv.Value;
                }
                if (File.Exists(tfJson))
                {
                    var tfText = File.ReadAllText(tfJson);
                    var loaded = JsonSerializer.Deserialize<Dictionary<int, Dictionary<string, int>>>(tfText);
                    if (loaded != null) foreach (var kv in loaded) termFrequencies[kv.Key] = kv.Value;
                }
                return;
            }
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine(ex.Message);
        }
    }

    [ProtoContract]
    public class ProtoIndex
    {
        [ProtoMember(1)]
        public Dictionary<int, string>? DocMap { get; set; }

        [ProtoMember(2)]
        public List<ProtoTermEntry>? Terms { get; set; }

        [ProtoMember(3)]
        public Dictionary<int, Dictionary<string, int>>? TermFrequencies { get; set; }
    }

    [ProtoContract]
    public class ProtoTermEntry
    {
        [ProtoMember(1)]
        public string? Term { get; set; }
        [ProtoMember(2)]
        public List<int>? Docs { get; set; }
    }

    public void SaveProtoIndex(string path)
    {
        var proto = new ProtoIndex
        {
            DocMap = new Dictionary<int, string>(docmap),
            TermFrequencies = new Dictionary<int, Dictionary<string, int>>(termFrequencies),
            Terms = index.Select(kv => new ProtoTermEntry { Term = kv.Key, Docs = kv.Value.ToList() }).ToList()
        };
        using var fs = File.Create(path);
        Serializer.Serialize(fs, proto);
    }

    public void LoadProtoIndex(string path)
    {
        using var fs = File.OpenRead(path);
        var proto = Serializer.Deserialize<ProtoIndex>(fs);
        if (proto?.DocMap != null)
        {
            docmap.Clear();
            foreach (var kv in proto.DocMap) docmap[kv.Key] = kv.Value;
        }
        if (proto?.Terms != null)
        {
            index.Clear();
            foreach (var t in proto.Terms) index[t.Term ?? string.Empty] = new HashSet<int>(t.Docs ?? new List<int>());
        }
        if (proto?.TermFrequencies != null)
        {
            termFrequencies.Clear();
            foreach (var kv in proto.TermFrequencies) termFrequencies[kv.Key] = kv.Value;
        }
    }

    public void SaveProtoDocMap(string path)
    {
        using var fs = File.Create(path);
        Serializer.Serialize(fs, docmap);
    }

    public void SaveProtoTermFrequencies(string path)
    {
        using var fs = File.Create(path);
        Serializer.Serialize(fs, termFrequencies);
    }

    public void LoadProtoDocMap(string path)
    {
        using var fs = File.OpenRead(path);
        var loaded = Serializer.Deserialize<Dictionary<int, string>>(fs);
        if (loaded != null)
        {
            docmap.Clear();
            foreach (var kv in loaded) docmap[kv.Key] = kv.Value;
        }
    }

    public void LoadProtoTermFrequencies(string path)
    {
        using var fs = File.OpenRead(path);
        var loaded = Serializer.Deserialize<Dictionary<int, Dictionary<string, int>>>(fs);
        if (loaded != null)
        {
            termFrequencies.Clear();
            foreach (var kv in loaded) termFrequencies[kv.Key] = kv.Value;
        }
    }

    public string? GetDocText(int docId)
    {
        return docmap.TryGetValue(docId, out var t) ? t : null;
    }
}

class Program
{
    static void PrintUsage()
    {
        Console.WriteLine("Usage: dotnet run -- <command> [args]\n");
        Console.WriteLine("Commands:");
        Console.WriteLine("  build                             Build index from data/movies.json");
        Console.WriteLine("  search <query>                    Search documents for query tokens");
        Console.WriteLine("  tf <docId> <term>                 Term frequency for term in document");
        Console.WriteLine("  idf <term>                        Inverse document frequency for term");
        Console.WriteLine("  tfidf <docId> <term>              TF-IDF score for term in document");
    }

    static int Main(string[] args)
    {
        if (args.Length == 0) { PrintUsage(); return 1; }
        var ii = new InvertedIndex();
        ii.LoadStopWords(Path.Combine("data", "stopwords.txt"));

        // Global option: --load-format=proto|json|both (default: proto)
        string loadFormat = "proto";
        foreach (var a in args)
        {
            if (a.StartsWith("--load-format="))
            {
                loadFormat = a.Substring("--load-format=".Length).ToLowerInvariant();
                break;
            }
        }
        ii.Load(Path.Combine("cache"), loadFormat);

        var cmd = args[0].ToLowerInvariant();
        try
        {
            switch (cmd)
            {
                case "build":
                    // build [json|proto|both]  - choose output format
                    ii.BuildFromMoviesJson(Path.Combine("data", "movies.json"));
                    string format = "both";
                    if (args.Length > 1)
                    {
                        var a1 = args[1];
                        if (a1.StartsWith("--format=")) format = a1.Substring("--format=".Length).ToLowerInvariant();
                        else format = a1.ToLowerInvariant();
                    }
                    bool writeJson = format == "json" || format == "both";
                    bool writeProto = format == "proto" || format == "protobuf" || format == "bin" || format == "both";
                    ii.Save("cache", writeJson, writeProto);
                    Console.WriteLine($"Index built and saved to cache/ (json={writeJson}, proto={writeProto})");
                    break;
                case "search":
                    if (args.Length < 2) { Console.WriteLine("Missing query"); return 2; }
                    var query = string.Join(' ', args.Skip(1));
                    var tokens = ii.Tokenize(query);
                    var results = new List<(int, string)>();
                    foreach (var token in tokens)
                    {
                        var docs = ii.GetDocuments(token);
                        foreach (var d in docs)
                        {
                            if (results.Count >= 10) break;
                            var text = ii.GetDocText(d) ?? "";
                            results.Add((d, text));
                        }
                    }
                    foreach (var r in results)
                    {
                        Console.WriteLine($"{r.Item1}: {r.Item2}");
                    }
                    break;
                case "tf":
                    if (args.Length < 3) { Console.WriteLine("Usage: tf <docId> <term>"); return 2; }
                    if (!int.TryParse(args[1], out var docId)) { Console.WriteLine("Invalid docId"); return 2; }
                    var termTf = args[2].ToLowerInvariant();
                    var tfVal = ii.GetTf(docId, termTf);
                    Console.WriteLine(tfVal);
                    break;
                case "idf":
                    if (args.Length < 2) { Console.WriteLine("Usage: idf <term>"); return 2; }
                    var termIdf = args[1].ToLowerInvariant();
                    var idf = ii.GetIdf(termIdf);
                    Console.WriteLine(idf);
                    break;
                case "tfidf":
                    if (args.Length < 3) { Console.WriteLine("Usage: tfidf <docId> <term>"); return 2; }
                    if (!int.TryParse(args[1], out var docId2)) { Console.WriteLine("Invalid docId"); return 2; }
                    var termTfidf = args[2].ToLowerInvariant();
                    var tfv = ii.GetTf(docId2, termTfidf);
                    var idfv = ii.GetIdf(termTfidf);
                    Console.WriteLine($"TF: {tfv}, IDF: {idfv:F4}, TF-IDF: {tfv * idfv:F4}");
                    break;
                default:
                    PrintUsage();
                    break;
            }
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine(ex.Message);
            return 3;
        }
        return 0;
    }
}
