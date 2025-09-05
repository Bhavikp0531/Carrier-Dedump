using System.Collections.Generic;

namespace CarrierDeduperAPI.Model
{
    public class DedupeResult
    {
        public string MasterQuestion { get; set; } = string.Empty;

        // All the variants of this question
        public List<string> Variants { get; set; } = new List<string>();

        // Carriers that asked these questions
        public List<string> Carriers { get; set; } = new List<string>();

        // The embedding vector used internally for similarity checks
        public float[] Embedding { get; set; }
    }
}
