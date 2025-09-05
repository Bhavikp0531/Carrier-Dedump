using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System.Text.RegularExpressions;

namespace CarrierDeduperAPI.Tokenizer
{
    public class BertTokenizer
    {
        private readonly Dictionary<string, int> _vocab;
        private readonly string _unkToken = "[UNK]";
        private readonly string _clsToken = "[CLS]";
        private readonly string _sepToken = "[SEP]";
        private readonly string _padToken = "[PAD]";

        public BertTokenizer(string vocabFilePath)
        {
            _vocab = File.ReadAllLines(vocabFilePath)
                         .Select((token, idx) => new { token, idx })
                         .ToDictionary(x => x.token, x => x.idx);
        }

        /// <summary>
        /// WordPiece tokenizer
        /// </summary>
        private List<string> WordPieceTokenize(string word)
        {
            var subTokens = new List<string>();

            if (_vocab.ContainsKey(word))
                return new List<string> { word };

            int start = 0;
            while (start < word.Length)
            {
                int end = word.Length;
                string curSubstr = null;

                while (start < end)
                {
                    string substr = word.Substring(start, end - start);
                    if (start > 0) substr = "##" + substr;

                    if (_vocab.ContainsKey(substr))
                    {
                        curSubstr = substr;
                        break;
                    }
                    end -= 1;
                }

                if (curSubstr == null)
                {
                    return new List<string> { _unkToken };
                }

                subTokens.Add(curSubstr);
                start = end;
            }

            return subTokens;
        }

        public (long[] inputIds, long[] attentionMask, long[] tokenTypeIds) Encode(
            string text, int maxLength = 128)
        {
            // 1. Basic tokenization: lowercase + split by punctuation/whitespace
            var words = Regex.Split(text.ToLower().Trim(), @"\W+")
                             .Where(w => !string.IsNullOrEmpty(w))
                             .ToList();

            // 2. WordPiece tokenize each word
            var tokens = new List<string>();
            foreach (var word in words)
                tokens.AddRange(WordPieceTokenize(word));

            // 3. Add special tokens
            var finalTokens = new List<string> { _clsToken };
            finalTokens.AddRange(tokens);
            finalTokens.Add(_sepToken);

            // 4. Convert to IDs
            var inputIds = finalTokens.Select(t => _vocab.ContainsKey(t) ? _vocab[t] : _vocab[_unkToken]).ToList();

            // 5. Pad/Truncate
            if (inputIds.Count > maxLength)
                inputIds = inputIds.Take(maxLength).ToList();
            else
                while (inputIds.Count < maxLength)
                    inputIds.Add(_vocab[_padToken]);

            // 6. Attention mask
            var attentionMask = inputIds.Select(id => id == _vocab[_padToken] ? 0L : 1L).ToList();

            // 7. Token type IDs (all 0 for single sentence)
            var tokenTypeIds = Enumerable.Repeat(0L, maxLength).ToList();

            return (inputIds.Select(x => (long)x).ToArray(),
                    attentionMask.ToArray(),
                    tokenTypeIds.ToArray());
        }
    }
}
