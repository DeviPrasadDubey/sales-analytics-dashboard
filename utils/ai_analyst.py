from groq import Groq
import os
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
import json

def get_ai_insight(df, user_question):
    client = Groq(api_key=GROQ_API_KEY)
    
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

Respond ONLY in this exact JSON format, nothing else:
{{
    "insight": "Your text answer here with specific numbers",
    "chart": {{
        "type": "bar" or "line" or "pie" or "scatter" or "none",
        "x": "column name for x axis",
        "y": "column name for y axis",
        "title": "chart title",
        "agg": "sum" or "mean" or "count"
    }}
}}

Rules:
- insight must have specific numbers from data
- chart.x and chart.y must be exact column names from: {columns}
- if no chart is needed, set type to "none"
- return ONLY the JSON, no extra text
"""
    
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1024
    )
    
    raw = response.choices[0].message.content.strip()
    # Clean JSON if model adds backticks
    raw = raw.replace("```json", "").replace("```", "").strip()
    
    try:
        result = json.loads(raw)
    except:
        result = {"insight": raw, "chart": {"type": "none"}}
    
    return result