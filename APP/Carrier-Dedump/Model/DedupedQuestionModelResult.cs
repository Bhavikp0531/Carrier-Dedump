namespace Carrier_Dedump.Model
{
   public class DedupedQuestionModelResult
   {
      public string MasterQuestion { get; set; }
      public List<string> Variants { get; set; }
      public List<string> Carriers { get; set; }
      public string Rationale { get; set; }
    }

}