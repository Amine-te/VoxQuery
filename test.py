"""
VoxQuery Integrated Test Script
Complete end-to-end pipeline: Text → SQL → Database → NL Response → Audio
"""

import sys
import time
from pathlib import Path
from rag_engine import RAGEngine
from nl_response_generator import NLResponseGenerator
from tts_helper import generate_audio_xtts


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
    """Print Stage 1 results"""
    print_section("📊", "STAGE 1: Table Retrieval")
    print_metric("Retrieval Time", f"{stage1_result['retrieval_time']:.3f}s")
    print_success(f"Found {len(stage1_result['tables'])} tables")
    
    for i, table in enumerate(stage1_result['tables'], 1):
        print(f"      {i}. {table['table_name']}")
    print()


def print_stage2_results(stage2_result: dict):
    """Print Stage 2 results"""
    print_section("📦", "STAGE 2: Data Retrieval")
    print_metric("Retrieval Time", f"{stage2_result['retrieval_time']:.3f}s")
    print_success(f"Retrieved {len(stage2_result['data_examples'])} data examples")
    print()


def print_sql_generation(sql_result: dict):
    """Print SQL generation results"""
    print_section("🤖", "STAGE 3: SQL Generation")
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
    print_section("💾", "STAGE 4: Query Execution")
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


def print_nl_response(nl_result: dict):
    """Print natural language response"""
    print_section("💬", "STAGE 5: Natural Language Response")
    print_metric("Generation Time", f"{nl_result['generation_time']:.2f}s")
    print_info(f"Detected Language: {nl_result['language']}")
    
    if nl_result['success']:
        print_success(f"Response generated ({nl_result['results_shown']} of {nl_result['results_count']} results)")
        print(f"\n      📝 Response:")
        print(f"      {nl_result['response']}\n")
    else:
        print_error(f"Generation failed: {nl_result.get('error', 'Unknown error')}")
    print()


def print_tts_results(tts_success: bool, tts_error: str, tts_time: float, audio_file: str):
    """Print TTS generation results"""
    print_section("🔊", "STAGE 6: Audio Generation")
    
    if tts_success:
        print_metric("Generation Time", f"{tts_time:.2f}s")
        print_success(f"Audio saved to: {audio_file}")
    else:
        print_error(f"Audio generation failed: {tts_error}")
    print()


def print_summary(rag_result: dict, nl_result: dict, tts_enabled: bool = False):
    """Print pipeline summary"""
    print("=" * 70)
    print("📊 PIPELINE SUMMARY")
    print("=" * 70)
    
    timings = rag_result['timings']
    print_metric("Stage 1 (Tables)", f"{timings['stage1']:.3f}s", indent=0)
    print_metric("Stage 2 (Data)", f"{timings['stage2']:.3f}s", indent=0)
    print_metric("Stage 3 (SQL Gen)", f"{timings['generation']:.2f}s", indent=0)
    print_metric("Stage 4 (Execution)", f"{timings['execution']:.3f}s", indent=0)
    print_metric("Stage 5 (NL Gen)", f"{nl_result['generation_time']:.2f}s", indent=0)
    
    if tts_enabled:
        print_info(f"Stage 6 (TTS) timing included in total", indent=0)
    
    total_time = timings['total'] + nl_result['generation_time']
    print_metric("TOTAL", f"{total_time:.2f}s", indent=0)
    
    print()
    print_info(f"Language: {nl_result['language']}", indent=0)
    
    if rag_result['success'] and nl_result['success']:
        print_success(f"Pipeline completed: {rag_result['results']['row_count']} rows retrieved", indent=0)
    else:
        print_error(f"Pipeline failed", indent=0)
    
    print("=" * 70 + "\n")


def main():
    """Main test function"""
    
    # Check for query argument
    if len(sys.argv) < 2:
        print("\n" + "=" * 70)
        print("🎤 VoxQuery - Integrated Voice to SQL Test Interface")
        print("=" * 70)
        print("\nUsage: python test.py \"your question here\" [options]")
        print("\nOptions:")
        print("  --tts                Enable audio generation")
        print("  --speaker PATH       Speaker reference audio for voice cloning")
        print("  --no-nl              Skip natural language response generation")
        print("  --verbose, -v        Show detailed debug information")
        print("  --quiet, -q          Minimal output (only results)")
        print("\nExamples:")
        print('  python test.py "Show me all trains departing from Casablanca"')
        print('  python test.py "Affiche tous les clients" --tts')
        print('  python test.py "List stations in Rabat" --tts --speaker voix.wav')
        print('  python test.py "Find departures today" -q')
        print("\n" + "=" * 70 + "\n")
        sys.exit(1)
    
    # Parse arguments
    verbose = '--verbose' in sys.argv or '-v' in sys.argv
    quiet = '--quiet' in sys.argv or '-q' in sys.argv
    tts_enabled = '--tts' in sys.argv
    no_nl = '--no-nl' in sys.argv
    
    # Get speaker file if provided
    speaker_wav = None
    if '--speaker' in sys.argv:
        speaker_idx = sys.argv.index('--speaker')
        if speaker_idx + 1 < len(sys.argv):
            speaker_wav = sys.argv[speaker_idx + 1]
    
    # Remove flags from argv
    args = []
    skip_next = False
    for i, arg in enumerate(sys.argv[1:]):
        if skip_next:
            skip_next = False
            continue
        if arg.startswith('-'):
            if arg == '--speaker':
                skip_next = True
            continue
        args.append(arg)
    
    user_question = " ".join(args)
    
    # Database configuration
    DB_CONFIG = {
        "host": "localhost",
        "user": "root",
        "password": "",
        "database": "oncf_db"
    }
    
    # Output directory
    output_dir = Path("./outputs")
    output_dir.mkdir(exist_ok=True)
    
    # Print initialization
    if not quiet:
        print_header("VoxQuery Integrated Test Environment")
        print("🔧 Initializing RAG Engine...")
        print("   📱 Loading embedding model...")
        print("   🗄️  Connecting to ChromaDB...")
        print("   🔌 Connecting to MySQL database...")
    
    start_init = time.time()
    
    try:
        # Create RAG engine
        with RAGEngine(
            db_config=DB_CONFIG,
            ollama_model="llama3:8b",
            n_tables=5,
            n_data_per_table=5,
            verbose=verbose
        ) as rag_engine:
            
            # Create NL Response Generator
            nl_generator = NLResponseGenerator(
                ollama_model="llama3:8b",
                max_results=10,
                timeout=120,
                verbose=verbose
            )
            
            init_time = time.time() - start_init
            
            if not quiet:
                print_success(f"Initialized in {init_time:.2f}s\n")
                print_header(f"QUERY: {user_question}")
            
            # Run RAG pipeline (SQL generation + execution)
            rag_result = rag_engine.process_query(user_question)
            
            # Print RAG results if not quiet
            if not quiet:
                # Stage 1
                print_stage1_results({
                    'tables': rag_result['debug_info']['stage1_tables'],
                    'retrieval_time': rag_result['timings']['stage1']
                })
                
                # Stage 2
                print_stage2_results({
                    'data_examples': rag_result['debug_info']['stage2_data'],
                    'retrieval_time': rag_result['timings']['stage2']
                })
                
                # Stage 3 (SQL Generation)
                print_sql_generation({
                    'success': rag_result['success'],
                    'query': rag_result['sql_query'],
                    'generation_time': rag_result['timings']['generation']
                })
                
                # Stage 4 (Execution)
                print_execution_results(rag_result['results'])
            
            # Generate natural language response
            nl_result = {"success": False, "response": "", "language": "en", "generation_time": 0.0}
            
            if not no_nl:
                if rag_result['success'] and rag_result['results']['row_count'] > 0:
                    # Generate response from results
                    nl_result = nl_generator.generate_response(
                        user_query=user_question,
                        db_results=rag_result['results']['results']
                    )
                elif rag_result['success'] and rag_result['results']['row_count'] == 0:
                    # No results found
                    nl_result = nl_generator.generate_response(
                        user_query=user_question,
                        db_results=[]
                    )
                else:
                    # SQL generation or execution failed
                    nl_result = nl_generator.generate_error_response(
                        user_query=user_question,
                        error_message=rag_result.get('error', 'Unknown error')
                    )
                
                # Print NL response
                if not quiet:
                    print_nl_response(nl_result)
                
                # Save response to file
                if nl_result['success'] and nl_result['response']:
                    response_file = output_dir / "response.txt"
                    with open(response_file, "w", encoding="utf-8") as f:
                        f.write(nl_result['response'])
                    if not quiet:
                        print_info(f"Response saved to: {response_file}")
            
            # Generate audio if requested
            if tts_enabled and nl_result['success'] and nl_result['response']:
                if not quiet:
                    print_section("🔊", "STAGE 6: Audio Generation")
                    print("      Generating audio with XTTS v2...")
                
                audio_output = output_dir / "response_audio.wav"
                
                tts_ok, tts_err, tts_elapsed, audio_file = generate_audio_xtts(
                    text=nl_result['response'],
                    output_path=str(audio_output),
                    language=nl_result['language'],
                    speaker_wav=speaker_wav
                )
                
                if not quiet:
                    print_tts_results(tts_ok, tts_err, tts_elapsed, audio_file)
                
                if not tts_ok and not quiet:
                    print_error(f"TTS Error: {tts_err}")
            
            # Print summary
            if not quiet and not no_nl:
                print_summary(rag_result, nl_result, tts_enabled=tts_enabled)
            
            # Quiet mode output
            if quiet:
                if rag_result['success']:
                    print(f"✅ {rag_result['results']['row_count']} rows | {rag_result['timings']['total']:.2f}s")
                    print(f"SQL: {rag_result['sql_query']}")
                    if nl_result['success'] and nl_result['response']:
                        print(f"Response: {nl_result['response']}")
                else:
                    print(f"❌ Failed: {rag_result.get('error', 'Unknown error')}")
            
            # Verbose debug output
            if verbose:
                print("\n" + "=" * 70)
                print("🔍 DETAILED DEBUG INFORMATION")
                print("=" * 70)
                
                print("\n📝 SQL Prompt:")
                print(rag_result['debug_info']['final_prompt'])
                
                if nl_result.get('prompt'):
                    print("\n📝 NL Prompt:")
                    print(nl_result['prompt'])
                
                print("\n" + "=" * 70 + "\n")
            
            # Exit status
            if rag_result['success']:
                if not quiet:
                    print("✅ Test completed successfully!\n")
                sys.exit(0)
            else:
                if not quiet:
                    print(f"❌ Test failed: {rag_result.get('error', 'Unknown error')}\n")
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