using Microsoft.AspNetCore.Mvc;
using CarrierDeduperAPI.Model;
using CarrierDeduperAPI.Service;
using System.Collections.Generic;
using System.Linq;
using System;


namespace CarrierDeduperAPI.Controllers
{
    [ApiController]
[Route("api/[controller]")]
public class DeduperController : ControllerBase
{
    private readonly EmbeddingService _embedding;

    public DeduperController(EmbeddingService embedding)
    {
        _embedding = embedding;
    }

        [HttpPost("dedupe")]
    public IActionResult Dedupe([FromBody] List<QuestionInput> inputs)
    {
       var groups = new List<DedupeResult>();

       foreach (var input in inputs)
       {
           var embedding = _embedding.GetEmbedding(input.Question);
           bool found = false;

           foreach (var group in groups)
           {
               var sim = EmbeddingService.CosineSimilarity(embedding, group.Embedding);
               if (sim > 0.8)
               {
                  group.Variants.Add(input.Question);
                  group.Carriers.Add(input.Carrier);
                  found = true;
                  break;
               }
            }

            if (!found)
            {
                groups.Add(new DedupeResult
                {
                  MasterQuestion = input.Question,
                  Variants = new List<string> { input.Question },
                  Carriers = new List<string> { input.Carrier },
                  Embedding = embedding
                });
            }
        }

        var output = groups.Select(g => new
        {
          g.MasterQuestion,
          g.Variants,
          g.Carriers,
          Rationale = $"This question was asked by {string.Join(", ", g.Carriers)}."
        });

        return Ok(output);
    }

   [HttpPost("debug")]
   public IActionResult Debug([FromBody] List<QuestionInput> inputs)
   {
      var results = inputs.Select(input => new
      {
          input.Carrier,
          input.Question,
          Embedding = _embedding.GetEmbeddingResult(input.Question)
      });
      return Ok(results);
    }
}
}
