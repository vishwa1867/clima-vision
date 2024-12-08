from flask import Flask, jsonify, request
from sqlalchemy import create_engine, MetaData
from sqlalchemy.orm import sessionmaker
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS

# Configure SQLite database connection
DATABASE_URI = 'sqlite:///C:/Users/vikra/OneDrive/Desktop/lalapeden/data.db'
engine = create_engine(DATABASE_URI)
metadata = MetaData()
metadata.reflect(bind=engine)

# Access the table
data_table = metadata.tables.get('data_table')

if data_table is None:
    raise ValueError("Table 'data_table' not found in the database.")

# Define a session
Session = sessionmaker(bind=engine)
session = Session()

@app.route('/get-column-names', methods=['GET'])
def get_column_names():
    columns = [column.name for column in data_table.columns]
    return jsonify(columns)

@app.route('/get-records', methods=['GET'])
def get_records():
    limit = request.args.get('limit')  # Get the limit parameter from the query string
    query = session.query(data_table)
    
    if limit:
        try:
            limit = int(limit)
            query = query.limit(limit)
        except ValueError:
            return jsonify({"error": "Invalid limit value"}), 400
    
    records = query.all()

    if not records:
        return jsonify({"message": "No records found."})

    # Manually convert records to a list of dictionaries
    records_list = []
    for record in records:
        record_dict = {column: getattr(record, column) for column in data_table.columns.keys()}
        records_list.append(record_dict)

    return jsonify(records_list)

@app.route('/get-latest-record', methods=['GET'])
def get_latest_record():
    # Query the latest record by ordering by a timestamp or primary key in descending order
    latest_record = session.query(data_table).order_by(data_table.c['Date & Time'].desc()).first()
    
    if not latest_record:
        return jsonify({"message": "No records found."})
    
    latest_record_dict = {column: getattr(latest_record, column) for column in data_table.columns.keys()}
    
    return jsonify(latest_record_dict)

if __name__ == '__main__':
    app.run(debug=True,port=5001)
