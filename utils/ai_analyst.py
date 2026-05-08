import anthropic
from config import ANTHROPIC_API_KEY

def get_ai_insight(df, user_question):
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    
    # Send column names + sample data + user question to Claude
    columns = df.columns.tolist()
    sample = df.head(5).to_string()
    dtypes = df.dtypes.to_string()
    stats = df.describe().to_string()
    
    prompt = f"""
You are a data analyst. Here is the dataset information:

Columns: {columns}
Data types: {dtypes}
Sample data:
{sample}

Summary statistics:
{stats}

User question: {user_question}

Answer the question with specific numbers from the data. Be concise and clear. 
If relevant, suggest what chart would best show this insight.
"""
    
    message = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}]
    )
    
    return message.content[0].text
