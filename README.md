# NLQ to SQL Query Generator

A secure, locally-hosted web application that converts natural language questions into SQL queries using GPT-4. Perfect for non-technical users who need to query data without knowing SQL.

## 🚀 Features

### Core Functionality
- **Natural Language Processing**: Convert plain English questions into SQL queries
- **GPT-4 Powered**: Advanced AI understanding of complex data questions
- **CSV/Excel Support**: Upload and query CSV or Excel files instantly
- **Schema Detection**: Automatic detection of columns, data types, and statistics
- **Query Execution**: Execute generated queries and see results immediately
- **Export Results**: Download query results as CSV files

### Security Features
- **Read-Only Queries**: Only SELECT statements allowed
- **SQL Injection Prevention**: Multiple layers of validation and sanitization
- **Rate Limiting**: 30 requests per minute to prevent abuse
- **Input Validation**: Strict validation of all user inputs
- **Secure File Handling**: File size limits and type validation

### User Experience
- **Modern UI**: Clean, responsive interface that works on all devices
- **Drag & Drop**: Easy file upload with drag and drop support
- **Query Examples**: Context-aware example queries based on your data
- **Syntax Highlighting**: Beautiful SQL syntax highlighting
- **Real-time Feedback**: Instant validation and error messages
- **Query Caching**: Lightning-fast repeated queries

## 📋 Prerequisites

- Python 3.8 or higher
- OpenAI API key (for GPT-4 access)
- 1GB RAM minimum
- Modern web browser

## 🔧 Installation

### 1. Clone or Download the Project

```bash
# If you have the files locally
cd /path/to/nlq-sql-generator
```

### 2. Create Virtual Environment (Recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

```bash
# Copy the example environment file
cp .env.example .env

# Edit .env and add your OpenAI API key
# Open .env in your text editor and update:
OPENAI_API_KEY=your_actual_openai_api_key_here
```

### 5. Create Required Directories

```bash
# The app will create these automatically, but you can create them manually
mkdir -p uploads cache logs
```

## 🚀 Running the Application

### Start the Server

```bash
python nlq_sql_app.py
```

You should see:
```
==================================================
🚀 NLQ to SQL Query Generator
==================================================
✅ Starting server on http://localhost:5001
📁 Upload folder: uploads
🔒 Security features: Enabled
⚡ Rate limiting: 30 requests/minute
==================================================
```

### Access the Application

Open your web browser and navigate to:
```
http://localhost:5001
```

## 📖 Usage Guide

### Step 1: Upload Your Data

1. Click "Choose File" or drag and drop a CSV/Excel file
2. The system will automatically detect the schema
3. Review the detected columns and data types

### Step 2: Ask Questions in Plain English

Examples:
- "Show all records"
- "Count unique customers"
- "Find top 10 products by sales"
- "Calculate average order value by month"
- "Show customers from New York with orders over $1000"

### Step 3: Review and Execute

1. Review the generated SQL query
2. Click "Execute" to run the query
3. View results in a formatted table
4. Export results as CSV if needed

## 🔒 Security Best Practices

1. **Keep your OpenAI API key secret** - Never commit it to version control
2. **Run locally only** - This is designed for local/intranet use
3. **Regular updates** - Keep dependencies updated for security patches
4. **Monitor usage** - Check logs regularly for suspicious activity
5. **Limit file sizes** - Default limit is 50MB per file

## 📁 Project Structure

```
nlq-sql-generator/
├── nlq_sql_app.py          # Main Flask application
├── templates/
│   └── index.html          # HTML template
├── static/
│   ├── css/
│   │   └── styles.css      # Stylesheet
│   └── js/
│       └── app.js          # Frontend JavaScript
├── uploads/                # Uploaded files (gitignored)
├── cache/                  # Query cache and temp databases
├── logs/                   # Application logs
├── requirements.txt        # Python dependencies
├── .env                    # Environment variables (create from .env.example)
└── README.md              # This file
```

## 🛠️ Configuration

Edit `nlq_sql_app.py` to customize:

```python
# File upload settings
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls'}

# Rate limiting
RATE_LIMIT_REQUESTS = 30  # requests per minute

# Query cache
QUERY_CACHE_TIME = 3600  # 1 hour in seconds
```

## 🐛 Troubleshooting

### Common Issues

**1. OpenAI API Key Error**
```
Error: Invalid API key
Solution: Check your .env file and ensure OPENAI_API_KEY is set correctly
```

**2. Port Already in Use**
```
Error: Address already in use
Solution: Change port in nlq_sql_app.py or stop other services on port 5001
```

**3. File Upload Fails**
```
Error: Failed to process file
Solution: Ensure file is CSV/Excel and under 50MB
```

**4. Module Import Errors**
```
Error: No module named 'flask'
Solution: Ensure virtual environment is activated and dependencies installed
```

### Debug Mode

For development, debug mode is enabled by default. For production:

```python
# In nlq_sql_app.py, change:
app.run(host='0.0.0.0', port=5001, debug=False)
```

## 📊 Example Queries

Based on a sales dataset with columns: `date`, `product`, `customer`, `amount`, `region`

| Natural Language | Generated SQL |
|-----------------|---------------|
| "Total sales" | `SELECT SUM(amount) as total_sales FROM data` |
| "Top 5 customers" | `SELECT customer, SUM(amount) as total FROM data GROUP BY customer ORDER BY total DESC LIMIT 5` |
| "Monthly sales trend" | `SELECT DATE_FORMAT(date, '%Y-%m') as month, SUM(amount) as sales FROM data GROUP BY month ORDER BY month` |
| "Products with no sales" | `SELECT DISTINCT product FROM data WHERE amount = 0 OR amount IS NULL` |

## 🔄 Updates and Maintenance

### Updating Dependencies
```bash
pip install --upgrade -r requirements.txt
```

### Clearing Cache
```bash
rm -rf cache/*.db
```

### Viewing Logs
```bash
tail -f logs/app.log  # Once logging is configured
```

## 🤝 Contributing

Feel free to enhance this tool with:
- Additional database connectors (PostgreSQL, MySQL, SQL Server)
- More advanced query validation
- Query history and saved queries
- User authentication and multi-tenancy
- Advanced visualizations
- Query optimization suggestions

## 📄 License

This is an MVP for demonstration purposes. Customize as needed for your use case.

## 🆘 Support

For issues or questions:
1. Check the troubleshooting section
2. Review error messages in the browser console
3. Check server logs for detailed error information

## 🎯 Roadmap

- [ ] SQL Server direct connection
- [ ] Multiple table support
- [ ] JOIN query generation
- [ ] Query history
- [ ] Advanced visualizations
- [ ] API endpoint for integration
- [ ] Docker containerization
- [ ] User authentication

---

**Built with ❤️ using Flask, GPT-4, and modern web technologies**
