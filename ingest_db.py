import mysql.connector
import chromadb
from sentence_transformers import SentenceTransformer
import json
from typing import List, Dict
import time

class DatabaseIndexer:
    def __init__(self, db_config: Dict):
        # Connect to MySQL
        self.db = mysql.connector.connect(**db_config)
        self.cursor = self.db.cursor(dictionary=True)
        
        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(path="./chroma_db")
        
        # Create separate collections for tables and data
        for collection_name in ["table_metadata", "table_data"]:
            try:
                self.client.delete_collection(collection_name)
            except:
                pass
        
        self.table_collection = self.client.create_collection(
            name="table_metadata",
            metadata={"hnsw:space": "cosine"}
        )
        
        self.data_collection = self.client.create_collection(
            name="table_data",
            metadata={"hnsw:space": "cosine"}
        )
        
        # Load embedding model
        print("Loading embedding model...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
    def get_table_metadata(self) -> List[Dict]:
        """Extract table-level metadata for retrieval"""
        print("\n📊 Extracting table metadata...")
        start = time.time()
        
        self.cursor.execute("SHOW TABLES")
        tables = [list(t.values())[0] for t in self.cursor.fetchall()]
        
        table_docs = []
        
        for table in tables:
            # Get columns
            self.cursor.execute(f"DESCRIBE {table}")
            columns = self.cursor.fetchall()
            col_names = [c['Field'] for c in columns]
            
            # Get foreign keys
            self.cursor.execute(f"""
                SELECT COLUMN_NAME, REFERENCED_TABLE_NAME, REFERENCED_COLUMN_NAME
                FROM INFORMATION_SCHEMA.KEY_COLUMN_USAGE
                WHERE TABLE_NAME = '{table}' AND REFERENCED_TABLE_NAME IS NOT NULL
                AND TABLE_SCHEMA = DATABASE()
            """)
            fks = self.cursor.fetchall()
            
            # Create ENHANCED searchable summary with semantic context
            table_name_words = table.replace('_', ' ')
            summary = f"Table {table} ({table_name_words}). "
            summary += f"Stores information about {table_name_words}. "
            summary += f"Contains columns: {', '.join(col_names)}. "
            
            # Add semantic context based on table name patterns
            table_lower = table.lower()
            if any(word in table_lower for word in ['client', 'customer', 'user', 'passenger']):
                summary += "Used for client/customer/user/passenger information. "
            if any(word in table_lower for word in ['billet', 'ticket', 'reservation', 'booking']):
                summary += "Used for ticket/booking/reservation information. "
            if any(word in table_lower for word in ['voyage', 'trip', 'journey', 'trajet']):
                summary += "Used for journey/trip/travel information. "
            if any(word in table_lower for word in ['train', 'locomotive']):
                summary += "Used for train/locomotive information. "
            if any(word in table_lower for word in ['gare', 'station']):
                summary += "Used for station/depot information. IMPORTANT: Gares table has 'ville' column for filtering by city. "
            if any(word in table_lower for word in ['horaire', 'schedule', 'timetable']):
                summary += "Used for schedule/timetable/timing information. "
            if any(word in table_lower for word in ['prix', 'tarif', 'price', 'cost']):
                summary += "Used for pricing/tariff/cost information. "
            if any(word in table_lower for word in ['paiement', 'payment', 'transaction']):
                summary += "Used for payment/transaction information. "
                
            # Add column-based context
            col_names_lower = [c.lower() for c in col_names]
            if any('nom' in c or 'name' in c for c in col_names_lower):
                summary += "Contains name information. "
            if any('email' in c or 'mail' in c for c in col_names_lower):
                summary += "Contains email/contact information. "
            if any('date' in c for c in col_names_lower):
                summary += "Contains date/temporal information. "
            if any('prix' in c or 'price' in c or 'montant' in c for c in col_names_lower):
                summary += "Contains pricing/amount information. "
            if any('ville' in c or 'city' in c for c in col_names_lower):
                summary += "Contains city/location information for geographic filtering. "
            
            if fks:
                fk_info = [f"references {fk['REFERENCED_TABLE_NAME']}" for fk in fks]
                summary += f"Relationships: {', '.join(fk_info)}. "
            
            # Store full schema separately
            full_schema = {
                "table_name": table,
                "columns": [{"name": c['Field'], "type": c['Type'], "key": c['Key']} for c in columns],
                "foreign_keys": [{"column": fk['COLUMN_NAME'], 
                                "ref_table": fk['REFERENCED_TABLE_NAME'],
                                "ref_column": fk['REFERENCED_COLUMN_NAME']} for fk in fks]
            }
            
            table_docs.append({
                "text": summary,
                "metadata": {
                    "table": table,
                    "type": "table_metadata",
                    "full_schema": json.dumps(full_schema)
                }
            })
        
        print(f"✓ Extracted {len(tables)} tables in {time.time()-start:.2f}s")
        return table_docs
    
    def _create_gare_example(self, row: Dict) -> str:
        """Create concise Gare example: city -> station name (ID)"""
        return f"City {row['ville']} has station '{row['nom_gare']}' (ID: {row['gare_id']})"
    
    def _create_train_example(self, row: Dict) -> str:
        """Create concise Train example: train type (ID) with capacity"""
        return f"Train '{row['type_train']}' (ID: {row['train_id']}) has capacity {row['capacite']}"
    
    def _create_client_example(self, row: Dict) -> str:
        """Create concise Client example: client name (ID)"""
        return f"Client '{row['nom']}' (ID: {row['client_id']})"
    
    def _create_voyage_example(self, row: Dict) -> str:
        """Create minimal Voyage example with just IDs"""
        return f"Voyage ID: {row['voyage_id']}"
    
    def _create_billet_example(self, row: Dict) -> str:
        """Create minimal Billet example with just IDs"""
        return f"Billet ID: {row['billet_id']}"
    
    def get_table_data(self, limit: int = 100) -> List[Dict]:
        """Extract sample data with table-specific formatting"""
        print("\n📦 Extracting sample data...")
        start = time.time()
        
        self.cursor.execute("SHOW TABLES")
        tables = [list(t.values())[0] for t in self.cursor.fetchall()]
        
        data_docs = []
        
        for table in tables:
            # Fetch rows from this table
            self.cursor.execute(f"SELECT * FROM {table} LIMIT {limit}")
            rows = self.cursor.fetchall()
            
            if not rows:
                continue
            
            table_lower = table.lower()
            
            # Create searchable documents for each row with table-specific formatting
            for row in rows:
                # Format based on table type
                if 'gare' in table_lower:
                    row_text = self._create_gare_example(row)
                elif 'train' in table_lower:
                    row_text = self._create_train_example(row)
                elif 'client' in table_lower:
                    row_text = self._create_client_example(row)
                elif 'voyage' in table_lower:
                    row_text = self._create_voyage_example(row)
                elif 'billet' in table_lower:
                    row_text = self._create_billet_example(row)
                else:
                    # Fallback to generic format
                    row_parts = [f"{k}={v}" for k, v in row.items() if v is not None]
                    row_text = f"Table {table}: " + ", ".join(row_parts)
                
                data_docs.append({
                    "text": row_text,
                    "metadata": {
                        "table": table,
                        "type": "data",
                        "row_data": json.dumps(row, default=str)
                    }
                })
        
        print(f"✓ Extracted {len(data_docs)} rows in {time.time()-start:.2f}s")
        return data_docs
    
    def index_documents(self, collection, documents: List[Dict], desc: str):
        """Embed and store documents"""
        print(f"\n📝 Indexing {len(documents)} {desc}...")
        start = time.time()
        
        texts = [doc["text"] for doc in documents]
        metadatas = [doc["metadata"] for doc in documents]
        ids = [f"{desc}_{i}" for i in range(len(documents))]
        
        batch_size = 100
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            batch_ids = ids[i:i+batch_size]
            batch_metadata = metadatas[i:i+batch_size]
            
            embeddings = self.model.encode(batch_texts, show_progress_bar=False)
            
            collection.add(
                embeddings=embeddings.tolist(),
                documents=batch_texts,
                metadatas=batch_metadata,
                ids=batch_ids
            )
            
            print(f"  Indexed {min(i+batch_size, len(texts))}/{len(texts)}")
        
        print(f"✓ Complete in {time.time()-start:.2f}s")
    
    def build_index(self, sample_data_limit: int = 100):
        """Build complete two-stage index"""
        print("🚀 Starting two-stage database indexing...\n")
        total_start = time.time()
        
        # Stage 1: Index table metadata
        table_docs = self.get_table_metadata()
        self.index_documents(self.table_collection, table_docs, "table metadata")
        
        # Stage 2: Index sample data
        data_docs = self.get_table_data(sample_data_limit)
        self.index_documents(self.data_collection, data_docs, "data rows")
        
        print(f"\n✅ COMPLETE! Total time: {time.time()-total_start:.2f}s")
        print(f"📊 Created 2 collections:")
        print(f"   - table_metadata: {len(table_docs)} tables")
        print(f"   - table_data: {len(data_docs)} rows")
        
        self.cursor.close()
        self.db.close()


if __name__ == "__main__":
    DB_CONFIG = {
        "host": "localhost",
        "user": "root",
        "password": "",
        "database": "oncf_db"
    }
    
    indexer = DatabaseIndexer(DB_CONFIG)
    indexer.build_index(sample_data_limit=100)