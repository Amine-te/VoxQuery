"""
VoxQuery RAG Engine
Core two-stage retrieval-augmented generation for Text-to-SQL
"""

import time
import json
import requests
import mysql.connector
import chromadb
from sentence_transformers import SentenceTransformer
from typing import Dict, List, Optional
import torch


class RAGEngine:
    """
    Two-stage RAG pipeline for Text-to-SQL generation
    
    Stage 1: Retrieve relevant table schemas
    Stage 2: Retrieve relevant data examples from those tables
    """
    
    def __init__(
        self,
        db_config: Dict,
        ollama_model: str = "llama3:8b",
        ollama_url: str = "http://localhost:11434/api/generate",
        n_tables: int = 5,
        n_data_per_table: int = 5,
        chroma_path: str = "./chroma_db",
        embedding_model: str = "all-MiniLM-L6-v2",
        verbose: bool = False
    ):
        """
        Initialize RAG Engine
        
        Args:
            db_config: Database connection configuration
            ollama_model: Name of Ollama model to use
            ollama_url: URL of Ollama API endpoint
            n_tables: Number of tables to retrieve in Stage 1
            n_data_per_table: Number of data examples per table in Stage 2
            chroma_path: Path to ChromaDB persistent storage
            embedding_model: SentenceTransformer model name
            verbose: Enable verbose logging
        """
        self.verbose = verbose
        self.ollama_model = ollama_model
        self.ollama_url = ollama_url
        self.n_tables = n_tables
        self.n_data_per_table = n_data_per_table
        
        # Load embedding model with GPU support
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.embedder = SentenceTransformer(embedding_model, device=self.device)
        
        # Load ChromaDB collections
        self.client = chromadb.PersistentClient(path=chroma_path)
        self.table_collection = self.client.get_collection("table_metadata")
        self.data_collection = self.client.get_collection("table_data")
        
        # Connect to database
        self.db = mysql.connector.connect(**db_config)
        self.cursor = self.db.cursor(dictionary=True)
        
        # Debug information storage
        self.debug_info = self._init_debug_info()
    
    def _init_debug_info(self) -> Dict:
        """Initialize debug information dictionary"""
        return {
            "stage1_tables": [],
            "stage2_data": [],
            "final_prompt": "",
            "stage1_time": 0,
            "stage2_time": 0,
            "generation_time": 0,
            "execution_time": 0,
            "retrieved_table_count": 0,
            "retrieved_example_count": 0
        }
    
    def _expand_query(self, query: str) -> str:
        """Expand short queries with semantic context for better retrieval"""
        return query
    
    def _rerank_tables(self, tables_info: List[Dict], query: str) -> List[Dict]:
        """Rerank retrieved tables based on exact name matches"""
        query_words = set(query.lower().split())
        
        # Separate exact matches and non-matches
        exact_matches = []
        other_matches = []
        
        for table_info in tables_info:
            table_name = table_info['table_name'].lower()
            table_name_words = table_name.replace('_', ' ')
            
            # Check if any query word is in table name
            if any(word in table_name or word in table_name_words for word in query_words):
                exact_matches.append(table_info)
            else:
                other_matches.append(table_info)
        
        # Combine: exact matches first, then others
        reranked = exact_matches + other_matches
        
        # Remove duplicates while preserving order
        seen = set()
        unique_tables = []
        for t in reranked:
            if t['table_name'] not in seen:
                seen.add(t['table_name'])
                unique_tables.append(t)
        
        return unique_tables
    
    def stage1_retrieve_tables(self, query: str, n_tables: Optional[int] = None) -> Dict:
        """
        Stage 1: Retrieve relevant table schemas
        
        Args:
            query: User's natural language query
            n_tables: Override default number of tables to retrieve
            
        Returns:
            Dict with tables info and timing
        """
        n_tables = n_tables or self.n_tables
        start = time.time()
        
        # Use original query
        query_to_use = query
        
        # Encode query
        query_embedding = self.embedder.encode([query_to_use])
        
        # Query table metadata collection (retrieve more for reranking)
        results = self.table_collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=min(n_tables * 2, 15)
        )
        
        # Parse results
        tables_info = []
        for doc, metadata in zip(results['documents'][0], results['metadatas'][0]):
            schema = json.loads(metadata['full_schema'])
            tables_info.append({
                "table_name": schema['table_name'],
                "schema": schema,
                "summary": doc
            })
        
        # Rerank based on exact matches
        tables_info = self._rerank_tables(tables_info, query)
        
        # Keep only top n_tables
        tables_info = tables_info[:n_tables]
        
        elapsed = time.time() - start
        
        # Update debug info
        self.debug_info["stage1_time"] = elapsed
        self.debug_info["stage1_tables"] = tables_info
        self.debug_info["retrieved_table_count"] = len(tables_info)
        
        if self.verbose:
            print(f"Retrieved tables: {[t['table_name'] for t in tables_info]}")
        
        return {
            "tables": tables_info,
            "retrieval_time": elapsed,
            "table_names": [t['table_name'] for t in tables_info]
        }
    
    def stage2_retrieve_data(
        self,
        query: str,
        table_names: List[str],
        n_data_per_table: Optional[int] = None
    ) -> Dict:
        """
        Stage 2: Retrieve relevant data examples from specific tables
        Uses semantic search to find examples that match the query intent
        
        Args:
            query: User's natural language query
            table_names: List of table names to retrieve data from
            n_data_per_table: Override default number of examples per table
            
        Returns:
            Dict with data examples and timing
        """
        n_data_per_table = n_data_per_table or self.n_data_per_table
        start = time.time()
        
        # Encode query
        query_embedding = self.embedder.encode([query])
        
        # Query data collection for each table using semantic similarity
        all_data = []
        for table in table_names:
            table_lower = table.lower()
            
            # Skip transaction tables (billets, voyages) - they only have IDs, not useful reference data
            if 'billet' in table_lower or 'voyage' in table_lower:
                continue
            
            try:
                # Use semantic search to find most relevant examples
                results = self.data_collection.query(
                    query_embeddings=query_embedding.tolist(),
                    n_results=n_data_per_table * 2,  # Get more candidates
                    where={"table": table}
                )
                
                if results['documents'][0]:
                    # Take top semantically similar examples
                    table_data = results['documents'][0][:n_data_per_table]
                    all_data.extend(table_data)
            except:
                pass
        
        elapsed = time.time() - start
        
        # Update debug info
        self.debug_info["stage2_time"] = elapsed
        self.debug_info["stage2_data"] = all_data
        self.debug_info["retrieved_example_count"] = len(all_data)
        
        return {
            "data_examples": all_data,
            "retrieval_time": elapsed
        }
    
    def build_prompt(
        self,
        user_query: str,
        tables_info: List[Dict],
        data_examples: List[str]
    ) -> str:
        """
        Build LLM prompt from retrieved context
        
        Args:
            user_query: User's natural language query
            tables_info: Retrieved table schema information
            data_examples: Retrieved data examples
            
        Returns:
            Complete prompt string
        """
        # Build schema section
        schema_section = "DATABASE SCHEMA:\n\n"
        for table_info in tables_info:
            schema = table_info['schema']
            schema_section += f"Table: {schema['table_name']}\n"
            schema_section += "Columns:\n"
            
            for col in schema['columns']:
                key_info = f" [{col['key']}]" if col['key'] else ""
                schema_section += f"  - {col['name']} ({col['type']}){key_info}\n"
            
            if schema['foreign_keys']:
                schema_section += "Foreign Keys:\n"
                for fk in schema['foreign_keys']:
                    schema_section += f"  - {fk['column']} -> {fk['ref_table']}.{fk['ref_column']}\n"
            
            schema_section += "\n"
        
        # Build data examples section
        data_section = ""
        if data_examples:
            data_section = "REFERENCE DATA (shows available values and ID mappings):\n"
            data_section += "\n".join(data_examples)
            data_section += "\n\n"
        
        # Build complete prompt
        prompt = f"""{schema_section}{data_section}USER QUESTION: {user_query}

Write a MySQL query to answer this question.

IMPORTANT RULES:
1. Use ONLY tables and columns from the schema
2. Reference data shows what values exist - use these to understand the data structure, NOT as literal filter values unless the user specifically mentions them
3. When joining tables, always use the foreign key relationships shown in the schema
4. For geographic or categorical filtering, query the appropriate descriptive columns (like 'ville', 'nom', 'type') rather than hardcoding IDs
5. Output ONLY the SQL query with no explanation

SQL Query:"""
        
        # Store in debug info
        self.debug_info["final_prompt"] = prompt
        
        return prompt
    
    def generate_sql(self, prompt: str) -> Dict:
        """
        Generate SQL query using Ollama LLM
        
        Args:
            prompt: Complete prompt with context
            
        Returns:
            Dict with generated SQL and timing
        """
        start = time.time()
        
        payload = {
            "model": self.ollama_model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,
                "num_predict": 300
            }
        }
        
        try:
            response = requests.post(self.ollama_url, json=payload, timeout=60)
            response.raise_for_status()
            result = response.json()
            
            elapsed = time.time() - start
            sql_query = result['response'].strip()
            
            # Clean up SQL
            sql_query = self._clean_sql(sql_query)
            
            self.debug_info["generation_time"] = elapsed
            
            return {
                "query": sql_query,
                "generation_time": elapsed,
                "success": True
            }
            
        except Exception as e:
            elapsed = time.time() - start
            self.debug_info["generation_time"] = elapsed
            
            return {
                "query": "",
                "generation_time": elapsed,
                "success": False,
                "error": str(e)
            }
    
    def _clean_sql(self, sql_query: str) -> str:
        """Clean up generated SQL query"""
        sql_query = sql_query.replace("```sql", "").replace("```", "").strip()
        
        if ";" not in sql_query:
            sql_query += ";"
        
        while ";;" in sql_query:
            sql_query = sql_query.replace(";;", ";")
        
        return sql_query
    
    def execute_query(self, sql_query: str) -> Dict:
        """
        Execute SQL query on database
        
        Args:
            sql_query: SQL query string
            
        Returns:
            Dict with execution results and timing
        """
        start = time.time()
        
        try:
            self.cursor.execute(sql_query)
            results = self.cursor.fetchall()
            elapsed = time.time() - start
            
            self.debug_info["execution_time"] = elapsed
            
            return {
                "success": True,
                "results": results,
                "row_count": len(results),
                "execution_time": elapsed,
                "columns": list(results[0].keys()) if results else []
            }
            
        except Exception as e:
            elapsed = time.time() - start
            self.debug_info["execution_time"] = elapsed
            
            return {
                "success": False,
                "error": str(e),
                "execution_time": elapsed,
                "results": [],
                "row_count": 0,
                "columns": []
            }
    
    def process_query(self, user_query: str) -> Dict:
        """
        Complete end-to-end RAG pipeline
        
        Args:
            user_query: User's natural language query
            
        Returns:
            Dict with complete results and debug information
        """
        total_start = time.time()
        
        # Reset debug info
        self.debug_info = self._init_debug_info()
        
        # Stage 1: Retrieve relevant tables
        stage1_result = self.stage1_retrieve_tables(user_query)
        
        # Stage 2: Retrieve data examples
        stage2_result = self.stage2_retrieve_data(
            user_query,
            stage1_result['table_names']
        )
        
        # Build prompt
        prompt = self.build_prompt(
            user_query,
            stage1_result['tables'],
            stage2_result['data_examples']
        )
        
        # Generate SQL
        sql_result = self.generate_sql(prompt)
        
        if not sql_result['success']:
            return {
                "success": False,
                "error": sql_result.get('error', 'SQL generation failed'),
                "sql_query": "",
                "results": {"success": False, "results": [], "row_count": 0},
                "timings": self._get_timings(time.time() - total_start),
                "debug_info": self.debug_info
            }
        
        # Execute query
        exec_result = self.execute_query(sql_result['query'])
        
        total_time = time.time() - total_start
        
        return {
            "success": exec_result['success'],
            "sql_query": sql_result['query'],
            "results": exec_result,
            "timings": self._get_timings(total_time),
            "debug_info": self.debug_info
        }
    
    def _get_timings(self, total_time: float) -> Dict:
        """Get formatted timing information"""
        return {
            "stage1": self.debug_info["stage1_time"],
            "stage2": self.debug_info["stage2_time"],
            "generation": self.debug_info["generation_time"],
            "execution": self.debug_info["execution_time"],
            "total": total_time
        }
    
    def close(self):
        """Close database connection"""
        if self.cursor:
            self.cursor.close()
        if self.db:
            self.db.close()
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()


def create_rag_engine(
    db_config: Dict,
    ollama_model: str = "llama3:8b",
    n_tables: int = 5,
    n_data_per_table: int = 5,
    **kwargs
) -> RAGEngine:
    """
    Create and initialize a RAG engine
    
    Args:
        db_config: Database connection configuration
        ollama_model: Ollama model name
        n_tables: Number of tables to retrieve
        n_data_per_table: Examples per table
        **kwargs: Additional arguments for RAGEngine
        
    Returns:
        Initialized RAGEngine instance
    """
    return RAGEngine(
        db_config=db_config,
        ollama_model=ollama_model,
        n_tables=n_tables,
        n_data_per_table=n_data_per_table,
        **kwargs
    )