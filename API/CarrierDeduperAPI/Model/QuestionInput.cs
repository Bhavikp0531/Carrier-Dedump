using Newtonsoft.Json;

namespace CarrierDeduperAPI.Model
{
    public class QuestionInput
    {
        [JsonProperty("carrier")]
        public string Carrier { get; set; }

        [JsonProperty("question")]
        public string Question { get; set; }
    }
}
