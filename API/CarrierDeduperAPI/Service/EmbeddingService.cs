using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using CarrierDeduperAPI.Model;
using CarrierDeduperAPI.Tokenizer;
using System.Linq;
using System.Collections.Generic;
using System;

namespace CarrierDeduperAPI.Service
{
    public class EmbeddingService
    {
        private readonly InferenceSession _session;
        private readonly BertTokenizer _tokenizer;

        public EmbeddingService(string modelPath, string vocabPath)
        {
            _session = new InferenceSession(modelPath);
            _tokenizer = new BertTokenizer(vocabPath);
        }

        public float[] GetEmbedding(string text, int maxLength = 128)
        {
            var result = GetEmbeddingResult(text, maxLength);
            return result.Embeddings;
        }

        public EmbeddingResult GetEmbeddingResult(string text, int maxLength = 128)
        {
            var (inputIds, attentionMask, tokenTypeIds) = _tokenizer.Encode(text, maxLength);

            // Convert to tensors
            var inputTensor = new DenseTensor<long>(new[] { 1, maxLength });
            var attentionTensor = new DenseTensor<long>(new[] { 1, maxLength });
            var tokenTypeTensor = new DenseTensor<long>(new[] { 1, maxLength });

            for (int i = 0; i < maxLength; i++)
            {
                inputTensor[0, i] = inputIds[i];
                attentionTensor[0, i] = attentionMask[i];
                tokenTypeTensor[0, i] = tokenTypeIds[i];
            }

            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("input_ids", inputTensor),
                NamedOnnxValue.CreateFromTensor("attention_mask", attentionTensor),
                NamedOnnxValue.CreateFromTensor("token_type_ids", tokenTypeTensor)
            };

            using var results = _session.Run(inputs);
            var output = results.First().AsEnumerable<float>().ToArray();

            return new EmbeddingResult
            {
                InputIds = inputIds,
                AttentionMask = attentionMask,
                TokenTypeIds = tokenTypeIds,
                Embeddings = output
            };
        }

        public static double CosineSimilarity(float[] v1, float[] v2)
        {
            double dot = 0, mag1 = 0, mag2 = 0;
            for (int i = 0; i < v1.Length; i++)
            {
                dot += v1[i] * v2[i];
                mag1 += v1[i] * v1[i];
                mag2 += v2[i] * v2[i];
            }
            return dot / (Math.Sqrt(mag1) * Math.Sqrt(mag2));
        }
    }
}
