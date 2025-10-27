"""
NLQ to SQL Query Generator
A secure, locally-hosted web application for converting natural language to SQL queries
"""

from flask import Flask, render_template, request, jsonify, session
from flask_cors import CORS
from werkzeug.utils import secure_filename
import openai
import pandas as pd
import os
import re
import hashlib
import time
from datetime import datetime, timedelta
import json
import sqlite3
from dotenv import load_dotenv
import secrets
import logging
from functools import wraps

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
app.secret_key = secrets.token_hex(32)
CORS(app)
# --- Make Flask JSON-friendly with NumPy/pandas types ---
import numpy as np
from flask.json.provider import DefaultJSONProvider

class NumpyJSONProvider(DefaultJSONProvider):
    def default(self, o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.bool_,)):
            return bool(o)
        if isinstance(o, (np.ndarray,)):
            return o.tolist()
        # Handle datetime objects
        if isinstance(o, (datetime, pd.Timestamp)):
            return o.isoformat()
        if hasattr(o, 'isoformat'):  # For date and time objects
            return o.isoformat()
        return super().default(o)

app.json = NumpyJSONProvider(app)
# --------------------------------------------------------

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls'}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
RATE_LIMIT_REQUESTS = 30  # requests per minute
QUERY_CACHE_TIME = 3600  # 1 hour in seconds

# Create necessary directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs('cache', exist_ok=True)
os.makedirs('logs', exist_ok=True)

# OpenAI configuration
openai.api_key = os.getenv('OPENAI_API_KEY')

# Security: Forbidden SQL keywords for data modification
FORBIDDEN_KEYWORDS = [
    'DROP', 'DELETE', 'INSERT', 'UPDATE', 'ALTER', 'CREATE', 
    'TRUNCATE', 'EXEC', 'EXECUTE', 'GRANT', 'REVOKE'
]

# Rate limiting storage
rate_limit_storage = {}

class SecurityValidator:
    """Handles all security validations for SQL queries"""
    
    @staticmethod
    def validate_sql_query(query):
        """Validate SQL query for security threats"""
        if not query:
            return False, "Empty query"
        
        # Convert to uppercase for checking
        query_upper = query.upper()
        
        # Check for forbidden keywords
        for keyword in FORBIDDEN_KEYWORDS:
            if keyword in query_upper:
                return False, f"Forbidden operation: {keyword}"
        
        # Check for SQL injection patterns
        injection_patterns = [
            r';\s*DROP',
            r'--',
            r'/\*.*\*/',
            r'xp_cmdshell',
            r'sp_executesql'
        ]
        
        for pattern in injection_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                return False, "Potential SQL injection detected"
        
        # Ensure it's a SELECT query
        if not query_upper.strip().startswith('SELECT'):
            return False, "Only SELECT queries are allowed"
        
        return True, "Query is safe"
    
    @staticmethod
    def sanitize_input(text):
        """Sanitize user input"""
        if not text:
            return ""
        
        # Remove potentially harmful characters
        text = re.sub(r'[<>\"\'`;]', '', text)
        
        # Limit length
        max_length = 500
        if len(text) > max_length:
            text = text[:max_length]
        
        return text.strip()

class QueryGenerator:
    """Handles SQL query generation using OpenAI"""
    
    def __init__(self):
        self.client = openai.OpenAI()
        self.cache = {}
    
    def get_cache_key(self, prompt, schema):
        """Generate cache key for query"""
        return hashlib.md5(f"{prompt}{schema}".encode()).hexdigest()
    
    def generate_query(self, user_prompt, table_info, sample_data=None):
        """Generate SQL query from natural language"""
        
        # Check cache first
        cache_key = self.get_cache_key(user_prompt, table_info)
        if cache_key in self.cache:
            cached_time, cached_query = self.cache[cache_key]
            if time.time() - cached_time < QUERY_CACHE_TIME:
                return cached_query, None, True  # Return cached result
        
        try:
            # Build comprehensive system prompt
            system_prompt = """You are an expert SQL analyst. Generate ONLY valid SQL queries.
            
STRICT RULES:
1. ONLY generate SELECT queries - no data modification allowed
2. Use proper SQL syntax
3. Include column names in quotes if they contain spaces
4. Always use table name 'data' for the main table
5. Return ONLY the SQL query without any explanation or markdown
6. Make queries efficient and optimized
7. Handle NULL values appropriately
8. Use appropriate aggregate functions when needed"""

            if table_info:
                system_prompt += f"\n\nTable Schema:\nTable Name: data\n{table_info}"
            
            if sample_data:
                system_prompt += f"\n\nSample Data (first 3 rows):\n{sample_data}"
            
            # Add query examples for better context
            system_prompt += """

Example queries:
- "show all records" → SELECT * FROM data
- "count unique customers" → SELECT COUNT(DISTINCT customer_id) FROM data
- "average sales by region" → SELECT region, AVG(sales) as avg_sales FROM data GROUP BY region
- "top 10 products by revenue" → SELECT product, SUM(revenue) as total_revenue FROM data GROUP BY product ORDER BY total_revenue DESC LIMIT 10"""

            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,  # Lower temperature for more consistent SQL
                max_tokens=500
            )
            
            sql_query = response.choices[0].message.content.strip()
            
            # Remove any markdown formatting if present
            sql_query = re.sub(r'^```sql\n', '', sql_query)
            sql_query = re.sub(r'\n```$', '', sql_query)
            sql_query = re.sub(r'^```\n', '', sql_query)
            sql_query = re.sub(r'\n```$', '', sql_query)
            
            # Cache the result
            self.cache[cache_key] = (time.time(), sql_query)
            
            return sql_query, None, False
            
        except Exception as e:
            logger.error(f"Query generation error: {str(e)}")
            return None, str(e), False

class DataProcessor:
    """Handles data processing and schema extraction"""
    
    @staticmethod
    def process_csv(file_path):
        """Process CSV or Excel file and extract schema"""
        try:
            df = None

            # Check file extension to determine how to read it
            if file_path.lower().endswith(('.xlsx', '.xls')):
                # Read Excel file
                try:
                    df = pd.read_excel(file_path, engine='openpyxl')
                except Exception as e:
                    logger.error(f"Excel read error: {str(e)}")
                    raise ValueError(f"Could not read Excel file: {str(e)}")
            else:
                # Read CSV with various encoding attempts
                encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']

                for encoding in encodings:
                    try:
                        df = pd.read_csv(file_path, encoding=encoding)
                        break
                    except UnicodeDecodeError:
                        continue
                    except Exception as e:
                        logger.error(f"CSV read error with {encoding}: {str(e)}")

            if df is None:
                raise ValueError("Could not read file with any supported encoding")
            
            # Clean column names
            df.columns = [col.strip() for col in df.columns]
            
            # Get schema information
            schema_info = []
            for col in df.columns:
                dtype = str(df[col].dtype)
                null_count = df[col].isnull().sum()
                unique_count = df[col].nunique()
                
                # Determine SQL type
                if dtype.startswith('int'):
                    sql_type = 'INTEGER'
                elif dtype.startswith('float'):
                    sql_type = 'FLOAT'
                elif dtype == 'object':
                    sql_type = 'TEXT'
                elif dtype.startswith('datetime'):
                    sql_type = 'DATETIME'
                else:
                    sql_type = 'TEXT'
                
                schema_info.append({
                    'column': col,
                    'type': sql_type,
                    'nulls': null_count,
                    'unique': unique_count,
                    'total': len(df)
                })
            
            # Convert datetime/time columns to strings for JSON serialization
            for col in df.columns:
                if pd.api.types.is_datetime64_any_dtype(df[col]):
                    df[col] = df[col].astype(str)
                elif df[col].dtype == 'object':
                    # Check if column contains time/date objects
                    try:
                        sample = df[col].dropna().iloc[0]
                        if sample and hasattr(sample, '__class__'):
                            type_name = sample.__class__.__name__
                            if type_name in ['time', 'datetime', 'date', 'Timestamp']:
                                df[col] = df[col].astype(str)
                    except (IndexError, AttributeError, TypeError):
                        pass

            # Get sample data
            sample_data = df.head(3).to_dict('records')
            
            # Store in SQLite for querying
            db_path = f"cache/{hashlib.md5(file_path.encode()).hexdigest()}.db"
            conn = sqlite3.connect(db_path)
            df.to_sql('data', conn, if_exists='replace', index=False)
            conn.close()
            
            return {
                'success': True,
                'schema': schema_info,
                'sample_data': sample_data,
                'row_count': len(df),
                'column_count': len(df.columns),
                'db_path': db_path
            }
            
        except Exception as e:
            logger.error(f"Data processing error: {str(e)}")
            return {'success': False, 'error': str(e)}

def rate_limit(func):
    """Decorator for rate limiting"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        client_ip = request.remote_addr
        current_time = time.time()
        
        # Clean old entries
        cutoff_time = current_time - 60
        rate_limit_storage[client_ip] = [
            t for t in rate_limit_storage.get(client_ip, []) 
            if t > cutoff_time
        ]
        
        # Check rate limit
        if len(rate_limit_storage.get(client_ip, [])) >= RATE_LIMIT_REQUESTS:
            return jsonify({'error': 'Rate limit exceeded. Please wait a moment.'}), 429
        
        # Add current request
        if client_ip not in rate_limit_storage:
            rate_limit_storage[client_ip] = []
        rate_limit_storage[client_ip].append(current_time)
        
        return func(*args, **kwargs)
    return wrapper

# Initialize components
validator = SecurityValidator()
generator = QueryGenerator()
processor = DataProcessor()

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        # Validate file extension
        file_ext = file.filename.lower().split('.')[-1] if '.' in file.filename else ''
        if file_ext not in ALLOWED_EXTENSIONS:
            return jsonify({'error': f'Invalid file type. Allowed: {", ".join(ALLOWED_EXTENSIONS)}'}), 400

        # Save file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{timestamp}_{filename}"
        file_path = os.path.join(UPLOAD_FOLDER, filename)

        try:
            file.save(file_path)
        except Exception as e:
            logger.error(f"File save error: {str(e)}")
            return jsonify({'error': f'Failed to save file: {str(e)}'}), 500

        # Check file size
        if os.path.getsize(file_path) > MAX_FILE_SIZE:
            os.remove(file_path)
            return jsonify({'error': 'File too large (max 50MB)'}), 400

        # Process file
        result = processor.process_csv(file_path)

        if result['success']:
            # Store in session
            session['current_file'] = file_path
            session['current_db'] = result['db_path']
            session['schema'] = result['schema']

            logger.info(f"File uploaded successfully: {filename}")

            return jsonify({
                'success': True,
                'schema': result['schema'],
                'sample_data': result['sample_data'],
                'stats': {
                    'rows': result['row_count'],
                    'columns': result['column_count']
                }
            })
        else:
            # Clean up file if processing failed
            if os.path.exists(file_path):
                os.remove(file_path)
            return jsonify({'error': result['error']}), 400

    except Exception as e:
        logger.error(f"Upload error: {str(e)}", exc_info=True)
        return jsonify({'error': f'Failed to process file: {str(e)}'}), 500

@app.route('/generate', methods=['POST'])
@rate_limit
def generate_sql():
    """Generate SQL query from natural language"""
    try:
        data = request.json
        user_prompt = data.get('prompt', '')
        
        # Validate input
        user_prompt = validator.sanitize_input(user_prompt)
        if not user_prompt:
            return jsonify({'error': 'Please provide a query description'}), 400
        
        # Get schema from session
        schema = session.get('schema', [])
        if not schema:
            return jsonify({'error': 'Please upload a data file first'}), 400
        
        # Format schema for GPT
        table_info = "\n".join([
            f"{s['column']} ({s['type']}) - {s['unique']} unique values, {s['nulls']} nulls"
            for s in schema
        ])
        
        # Generate SQL query
        sql_query, error, from_cache = generator.generate_query(user_prompt, table_info)
        
        if error:
            return jsonify({'error': f'Generation failed: {error}'}), 500
        
        # Validate generated SQL
        is_safe, message = validator.validate_sql_query(sql_query)
        if not is_safe:
            return jsonify({'error': f'Security validation failed: {message}'}), 400
        
        # Log query generation
        logger.info(f"Generated query: {sql_query[:100]}... (from_cache: {from_cache})")
        
        return jsonify({
            'success': True,
            'query': sql_query,
            'from_cache': from_cache,
            'validation': message
        })
        
    except Exception as e:
        logger.error(f"Generation error: {str(e)}")
        return jsonify({'error': 'Failed to generate query'}), 500

@app.route('/execute', methods=['POST'])
@rate_limit
def execute_query():
    """Execute SQL query on uploaded data"""
    try:
        data = request.json
        sql_query = data.get('query', '')
        
        # Validate query
        is_safe, message = validator.validate_sql_query(sql_query)
        if not is_safe:
            return jsonify({'error': f'Security validation failed: {message}'}), 400
        
        # Get database path from session
        db_path = session.get('current_db')
        if not db_path or not os.path.exists(db_path):
            return jsonify({'error': 'No data available. Please upload a file first.'}), 400
        
        # Execute query
        conn = sqlite3.connect(db_path)
        try:
            # Limit results for safety
            limited_query = sql_query
            if 'LIMIT' not in sql_query.upper():
                limited_query = f"{sql_query} LIMIT 1000"
            
            df = pd.read_sql_query(limited_query, conn)
            
            # Convert to JSON-serializable format
            result = {
                'success': True,
                'columns': list(df.columns),
                'data': df.to_dict('records'),
                'row_count': len(df),
                'limited': len(df) == 1000
            }
            
            return jsonify(result)
            
        except Exception as e:
            logger.error(f"Query execution error: {str(e)}")
            return jsonify({'error': f'Query execution failed: {str(e)}'}), 400
        finally:
            conn.close()
            
    except Exception as e:
        logger.error(f"Execute error: {str(e)}")
        return jsonify({'error': 'Failed to execute query'}), 500

@app.route('/examples', methods=['GET'])
def get_examples():
    """Get example queries based on current schema"""
    schema = session.get('schema', [])
    
    if not schema:
        return jsonify({
            'examples': [
                "Upload a CSV file to get started",
                "Then describe what data you want to see"
            ]
        })
    
    # Generate relevant examples based on schema
    columns = [s['column'] for s in schema]
    examples = [
        "Show all data",
        "Show first 10 rows",
        f"Count total records",
    ]
    
    if len(columns) > 0:
        examples.append(f"Show unique values in {columns[0]}")
    
    if len(columns) > 1:
        examples.append(f"Group by {columns[0]} and count")
    
    # Add data type specific examples
    for s in schema:
        if s['type'] in ['INTEGER', 'FLOAT']:
            examples.append(f"Calculate average {s['column']}")
            examples.append(f"Find maximum {s['column']}")
            break
    
    return jsonify({'examples': examples})

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0'
    })

if __name__ == '__main__':
    print("\n" + "="*50)
    print("🚀 NLQ to SQL Query Generator")
    print("="*50)
    print(f"✅ Starting server on http://localhost:5001")
    print(f"📁 Upload folder: {UPLOAD_FOLDER}")
    print(f"🔒 Security features: Enabled")
    print(f"⚡ Rate limiting: {RATE_LIMIT_REQUESTS} requests/minute")
    print("="*50 + "\n")
    
    app.run(host='0.0.0.0', port=5001, debug=True)
