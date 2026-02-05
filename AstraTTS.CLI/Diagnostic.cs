using System;
using System.IO;
using System.Linq;
using Microsoft.ML.Tokenizers;
using System.Reflection;
using Microsoft.ML.OnnxRuntime;

namespace AstraTTS.CLI
{
    public static class Diagnostic
    {
        public static void PrintMetadata(string modelPath)
        {
            if (!File.Exists(modelPath))
            {
                Console.WriteLine($"File not found: {modelPath}");
                return;
            }

            try
            {
                using var session = new InferenceSession(modelPath);
                Console.WriteLine($"\n=== Metadata for {Path.GetFileName(modelPath)} ===");
                Console.WriteLine($"Path: {Path.GetFullPath(modelPath)}");

                Console.WriteLine("Inputs:");
                foreach (var input in session.InputMetadata)
                {
                    var dims = string.Join(", ", input.Value.Dimensions.Select(d => d.ToString()));
                    Console.WriteLine($"  - {input.Key}: {input.Value.ElementType} [{dims}]");
                }

                Console.WriteLine("Outputs:");
                foreach (var output in session.OutputMetadata)
                {
                    var dims = string.Join(", ", output.Value.Dimensions.Select(d => d.ToString()));
                    Console.WriteLine($"  - {output.Key}: {output.Value.ElementType} [{dims}]");
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error reading {modelPath}: {ex.Message}");
            }
        }

        public static void PrintTypes()
        {
            var tokenizerType = typeof(Tokenizer);
            Console.WriteLine($"=== Methods of {tokenizerType.Name} ===");
            foreach (var method in tokenizerType.GetMethods().Select(m => m.Name).Distinct().OrderBy(s => s))
            {
                Console.WriteLine(method);
            }
        }
    }
}
