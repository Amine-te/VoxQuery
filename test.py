"""
VoxQuery Test Script
Command-line interface for testing Text-to-SQL generation with fancy output
"""

import sys
import time
from rag_engine import RAGEngine


def print_header(text: str):
    """Print a fancy header"""
    print("\n" + "=" * 70)
    print(f"🎯 {text}")
    print("=" * 70 + "\n")


def print_section(icon: str, text: str):
    """Print a section header"""
    print(f"\n{icon} {text}")


def print_metric(label: str, value: str, indent: int = 1):
    """Print a metric with proper indentation"""
    spaces = "   " * indent
    print(f"{spaces}⏱️  {label}: {value}")


def print_success(message: str, indent: int = 1):
    """Print a success message"""
    spaces = "   " * indent
    print(f"{spaces}✅ {message}")


def print_error(message: str, indent: int = 1):
    """Print an error message"""
    spaces = "   " * indent
    print(f"{spaces}❌ {message}")


def print_info(message: str, indent: int = 1):
    """Print an info message"""
    spaces = "   " * indent
    print(f"{spaces}ℹ️  {message}")


def print_stage1_results(stage1_result: dict):
    """Print Stage 1 results with formatting"""
    print_section("📊", "STAGE 1: Table Retrieval")
    print_metric("Retrieval Time", f"{stage1_result['retrieval_time']:.3f}s")
    print_success(f"Found {len(stage1_result['tables'])} tables")
    
    for i, table in enumerate(stage1_result['tables'], 1):
        print(f"      {i}. {table['table_name']}")
    print()


def print_stage2_results(stage2_result: dict):
    """Print Stage 2 results with formatting"""
    print_section("📦", "STAGE 2: Data Retrieval")
    print_metric("Retrieval Time", f"{stage2_result['retrieval_time']:.3f}s")
    print_success(f"Retrieved {len(stage2_result['data_examples'])} data examples")
    print()


def print_prompt_info(debug_info: dict):
    """Print prompt building information"""
    print_section("📝", "Prompt Construction")
    prompt_len = len(debug_info['final_prompt'])
    est_tokens = prompt_len // 4
    print_info(f"Prompt Length: {prompt_len} chars (~{est_tokens} tokens)")
    print_info(f"Context: {debug_info['retrieved_table_count']} tables, {debug_info['retrieved_example_count']} examples")
    print()


def print_sql_generation(sql_result: dict):
    """Print SQL generation results"""
    print_section("🤖", "SQL Generation")
    print_metric("Generation Time", f"{sql_result['generation_time']:.2f}s")
    
    if sql_result['success']:
        print_success("SQL generated successfully")
        print(f"\n      📜 SQL Query:")
        print(f"      {sql_result['query']}\n")
    else:
        print_error(f"Generation failed: {sql_result.get('error', 'Unknown error')}")
    print()


def print_execution_results(exec_result: dict):
    """Print query execution results"""
    print_section("💾", "Query Execution")
    print_metric("Execution Time", f"{exec_result['execution_time']:.3f}s")
    
    if exec_result['success']:
        print_success(f"Query executed successfully: {exec_result['row_count']} rows")
        
        if exec_result['results']:
            print("\n      🎯 Results (first 10 rows):")
            print("      " + "-" * 66)
            for i, row in enumerate(exec_result['results'][:10], 1):
                print(f"      {i}. {row}")
    else:
        print_error(f"Execution failed: {exec_result['error']}")
    print()


def print_summary(result: dict):
    """Print pipeline summary"""
    print("=" * 70)
    print("📊 PIPELINE SUMMARY")
    print("=" * 70)
    
    timings = result['timings']
    print_metric("Stage 1 (Tables)", f"{timings['stage1']:.3f}s", indent=0)
    print_metric("Stage 2 (Data)", f"{timings['stage2']:.3f}s", indent=0)
    print_metric("Generation", f"{timings['generation']:.2f}s", indent=0)
    print_metric("Execution", f"{timings['execution']:.3f}s", indent=0)
    print_metric("TOTAL", f"{timings['total']:.2f}s", indent=0)
    
    print()
    
    if result['success']:
        print_success(f"Pipeline completed: {result['results']['row_count']} rows retrieved", indent=0)
    else:
        print_error(f"Pipeline failed: {result.get('error', 'Unknown error')}", indent=0)
    
    print("=" * 70 + "\n")


def print_detailed_debug(debug_info: dict):
    """Print detailed debug information"""
    print("\n" + "=" * 70)
    print("🔍 DETAILED DEBUG INFORMATION")
    print("=" * 70)
    
    # Stage 1 details
    print("\n📊 Stage 1 - Retrieved Tables:")
    for i, table in enumerate(debug_info['stage1_tables'], 1):
        print(f"\n   {i}. Table: {table['table_name']}")
        print(f"      Summary: {table['summary']}")
        print(f"      Columns: {len(table['schema']['columns'])}")
        
        if table['schema']['foreign_keys']:
            print(f"      Foreign Keys: {len(table['schema']['foreign_keys'])}")
            for fk in table['schema']['foreign_keys']:
                print(f"         • {fk['column']} → {fk['ref_table']}.{fk['ref_column']}")
    
    # Stage 2 details
    print(f"\n📦 Stage 2 - Retrieved Data Examples ({len(debug_info['stage2_data'])}):")
    for i, example in enumerate(debug_info['stage2_data'][:5], 1):
        print(f"   {i}. {example}")
    
    if len(debug_info['stage2_data']) > 5:
        print(f"   ... and {len(debug_info['stage2_data']) - 5} more examples")
    
    # Prompt details
    print(f"\n📝 Prompt Statistics:")
    print(f"   • Total Length: {len(debug_info['final_prompt'])} characters")
    print(f"   • Estimated Tokens: ~{len(debug_info['final_prompt']) // 4}")
    print(f"   • Tables in Context: {debug_info['retrieved_table_count']}")
    print(f"   • Examples in Context: {debug_info['retrieved_example_count']}")
    
    print("\n" + "=" * 70 + "\n")


def main():
    """Main test function"""
    
    # Check for query argument
    if len(sys.argv) < 2:
        print("\n" + "=" * 70)
        print("🎤 VoxQuery - Voice to SQL Test Interface")
        print("=" * 70)
        print("\nUsage: python test.py \"your question here\" [options]")
        print("\nOptions:")
        print("  --verbose, -v    Show detailed debug information")
        print("  --quiet, -q      Minimal output (only results)")
        print("\nExamples:")
        print('  python test.py "Show me all trains departing from Casablanca"')
        print('  python test.py "List stations in Rabat" --verbose')
        print('  python test.py "Find departures today" -q')
        print("\n" + "=" * 70 + "\n")
        sys.exit(1)
    
    # Parse arguments
    verbose = '--verbose' in sys.argv or '-v' in sys.argv
    quiet = '--quiet' in sys.argv or '-q' in sys.argv
    
    # Remove flags from argv
    args = [arg for arg in sys.argv[1:] if not arg.startswith('-')]
    user_question = " ".join(args)
    
    # Database configuration
    DB_CONFIG = {
        "host": "localhost",
        "user": "root",
        "password": "",
        "database": "oncf_db"
    }
    
    # Print initialization
    if not quiet:
        print_header("VoxQuery Test Environment")
        print("🔧 Initializing RAG Engine...")
        print("   📱 Loading embedding model...")
        print("   🗄️  Connecting to ChromaDB...")
        print("   🔌 Connecting to MySQL database...")
    
    start_init = time.time()
    
    try:
        # Create RAG engine with context manager for automatic cleanup
        with RAGEngine(
            db_config=DB_CONFIG,
            ollama_model="llama3:8b",
            n_tables=5,
            n_data_per_table=5
        ) as rag_engine:
            
            init_time = time.time() - start_init
            
            if not quiet:
                print_success(f"Initialized in {init_time:.2f}s\n")
                print_header(f"QUERY: {user_question}")
            
            # Stage 1
            stage1_result = rag_engine.stage1_retrieve_tables(user_question)
            if not quiet:
                print_stage1_results(stage1_result)
            
            # Stage 2
            stage2_result = rag_engine.stage2_retrieve_data(
                user_question,
                stage1_result['table_names']
            )
            if not quiet:
                print_stage2_results(stage2_result)
            
            # Build prompt
            prompt = rag_engine.build_prompt(
                user_question,
                stage1_result['tables'],
                stage2_result['data_examples']
            )
            if not quiet:
                print_prompt_info(rag_engine.debug_info)
            
            # Generate SQL
            sql_result = rag_engine.generate_sql(prompt)
            if not quiet:
                print_sql_generation(sql_result)
            
            if not sql_result['success']:
                if quiet:
                    print(f"❌ SQL generation failed: {sql_result.get('error', 'Unknown error')}")
                sys.exit(1)
            
            # Execute query
            exec_result = rag_engine.execute_query(sql_result['query'])
            if not quiet:
                print_execution_results(exec_result)
            
            # Build complete result
            total_time = (
                stage1_result['retrieval_time'] +
                stage2_result['retrieval_time'] +
                sql_result['generation_time'] +
                exec_result['execution_time']
            )
            
            result = {
                "success": exec_result['success'],
                "sql_query": sql_result['query'],
                "results": exec_result,
                "timings": {
                    "stage1": stage1_result['retrieval_time'],
                    "stage2": stage2_result['retrieval_time'],
                    "generation": sql_result['generation_time'],
                    "execution": exec_result['execution_time'],
                    "total": total_time
                },
                "debug_info": rag_engine.debug_info
            }
            
            # Print summary
            if not quiet:
                print_summary(result)
            
            # Print detailed debug if requested
            if verbose:
                print_detailed_debug(result['debug_info'])
            
            # Quiet mode output
            if quiet:
                if result['success']:
                    print(f"✅ {result['results']['row_count']} rows | {total_time:.2f}s")
                    print(f"SQL: {result['sql_query']}")
                else:
                    print(f"❌ Failed: {result['results']['error']}")
            
            # Exit status
            if result['success']:
                if not quiet:
                    print("✅ Test completed successfully!\n")
                sys.exit(0)
            else:
                if not quiet:
                    print(f"❌ Test failed: {result.get('error', 'Unknown error')}\n")
                sys.exit(1)
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user\n")
        sys.exit(130)
    
    except Exception as e:
        print(f"\n❌ Fatal error: {str(e)}\n")
        import traceback
        if verbose:
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()