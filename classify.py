import boto3
import json

def classify_query(user_query: str):
  bedrock = boto3.client('bedrock-runtime', region_name='us-west-2')
  
  prompt = f"""
  You are a query classifier. Your job is to classify user queries as either "simple" or "complex".
  
  Simple queries are basic questions that can be answered quickly, like:
  - "What is the weather?"
  - "What time is it?"
  - "How old is the president?"
  
  Complex queries require more thought, research, or multiple steps, like:
  - "Explain quantum physics and its applications"
  - "Write a business plan for a startup"
  - "Compare different investment strategies"
  
  User query: "{user_query}"
  
  Classification (respond with only "simple" or "complex"):
  """
  
  request_body = {
    "anthropic_version": "bedrock-2023-05-31",
    "messages": [
      {
        "role": "user",
        "content": prompt
      }
    ],
    "max_tokens": 10
  }
  
  response = bedrock.invoke_model(
    modelId='anthropic.claude-3-haiku-20240307-v1:0',
    body=json.dumps(request_body),
    contentType='application/json'
  )

  response_body = json.loads(response['body'].read())
  classification = response_body['content'][0]['text'].strip().lower()
  
  return classification

def main() -> None:
  test_queries = [
    "What is 2+2?",
    "Explain the economic impact of artificial intelligence on job markets",
    "What color is the sky?",
    "Write a detailed marketing strategy for a new product launch"
  ]
  
  for query in test_queries:
    result = classify_query(query)
    print(f"Query: '{query}'")
    print(f"Classification: {result}")
    print("-" * 50)

if __name__ == "__main__":
  main()