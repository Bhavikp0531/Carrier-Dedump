namespace CarrierDeduperAPI.Model
{
    public class EmbeddingResult
    {
        public long[] InputIds { get; set; } = Array.Empty<long>();
        public long[] AttentionMask { get; set; } = Array.Empty<long>();
        public long[] TokenTypeIds { get; set; } = Array.Empty<long>();
        public float[] Embeddings { get; set; } = Array.Empty<float>();
    }
}
