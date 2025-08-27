"""Tab management for yQi evaluation platform."""

import streamlit as st
from typing import Dict, List, Any
import json
from datetime import datetime

from core.config import AppConfig
from core.services import ServiceManager
from core.exceptions import APIError, ConfigurationError, ValidationError
from ui.components import UIComponents
from api_client import send_prompts_to_chatgpt
from file_manager import save_responses_to_json, save_benchmark_data, load_responses_from_json
from batch_manager import batch_manager
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from simple_rating_interface import SimpleRatingInterface
from ui_components import display_responses, display_benchmark_summary, render_download_button


class TabManager:
    """Manages all application tabs."""
    
    def __init__(self, config: AppConfig, service_manager: ServiceManager):
        self.config = config
        self.service_manager = service_manager
        self.ui = UIComponents(config)
    
    def render_generate_tab(self, api_key: str):
        """Render the Generate Responses tab."""
        st.header("📝 Generate New Responses")
        
        # Model selection
        model_type = st.selectbox(
            "Select AI Model",
            ["ChatGPT-4o", "RAG (Local Documents)"],
            help="Choose between OpenAI's ChatGPT or local RAG system using TCM documents"
        )
        
        # Prompt configuration
        system_prompt, prompts = self.ui.render_prompt_configuration()
        self.ui.render_prompt_preview(prompts)
        
        # Processing mode selector (only for ChatGPT)
        if model_type == "ChatGPT-4o":
            processing_mode = self.ui.render_processing_mode_selector()
        else:
            processing_mode = "Real-time"  # RAG only supports real-time
            st.info("📚 RAG mode uses local TCM documents for knowledge-based responses")
        
        # Generate button
        if model_type == "ChatGPT-4o":
            button_text = "🚀 Generate Responses" if processing_mode == "Real-time" else "📦 Submit Batch Job"
        else:
            button_text = "🧠 Generate RAG Responses"
            
        if st.button(button_text, type="primary"):
            if model_type == "ChatGPT-4o":
                if processing_mode == "Real-time":
                    self._handle_generation(api_key, system_prompt, prompts)
                else:
                    self._handle_batch_submission(api_key, system_prompt, prompts)
            else:
                self._handle_rag_generation(system_prompt, prompts, api_key)
        
        # Batch job monitoring (only for ChatGPT mode)
        if model_type == "ChatGPT-4o":
            all_jobs = batch_manager.get_all_jobs()
            if processing_mode == "Batch" or all_jobs:
                st.divider()
                self._render_batch_monitoring(api_key)
    
    def _handle_generation(self, api_key: str, system_prompt: str, prompts: list):
        """Handle the response generation process."""
        # Validation
        if not api_key:
            self.ui.render_error_state("api_key", "")
            return
        
        if not prompts:
            self.ui.render_error_state("no_prompts", "")
            return
        
        # Generation process
        with self.ui.render_loading_state("Sending prompts to ChatGPT-4o..."):
            try:
                responses_data, benchmark_data = send_prompts_to_chatgpt(prompts, api_key, system_prompt)
                responses_file = save_responses_to_json(responses_data)
                benchmark_file = save_benchmark_data(benchmark_data)
                
                # Data is automatically saved to JSON files by the API client
                self.ui.render_generation_results(
                    responses_file, benchmark_file, benchmark_data
                )
                display_benchmark_summary(benchmark_data)
                
            except Exception as e:
                self.ui.render_error_state("generation", str(e))
    
    def _handle_rag_generation(self, system_prompt: str, prompts: list, api_key: str = None):
        """Handle RAG response generation process."""
        # Validation
        if not prompts:
            self.ui.render_error_state("no_prompts", "")
            return
        
        if not api_key:
            self.ui.render_error_state("api_key", "")
            return
        
        # Import RAG system
        try:
            import sys
            import os
            # Get the yQi root directory (parent of evaluation)
            evaluation_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            yqi_root = os.path.dirname(evaluation_dir)
            models_path = os.path.join(yqi_root, 'models')
            
            if models_path not in sys.path:
                sys.path.insert(0, models_path)
            
            from rag_system import RAGSystem
            from document_processor import DocumentProcessor
        except ImportError as e:
            st.error(f"RAG system not available: {e}")
            st.info("Please ensure the models folder is properly set up with RAG dependencies.")
            return
        
        # Initialize RAG system
        with st.spinner("Initializing RAG system..."):
            try:
                rag_system = RAGSystem(api_key=api_key)
                doc_processor = DocumentProcessor()
                
                # Check if vector database exists, if not build it
                if not rag_system.load_database():
                    st.info("Building knowledge base from documents...")
                    with st.spinner("Processing documents and building vector database..."):
                        try:
                            st.info(f"Processing documents from: {doc_processor.docs_directory}")
                            available_docs = doc_processor.get_available_documents()
                            st.info(f"Found documents: {available_docs}")
                            
                            chunks = doc_processor.process_documents()
                            st.info(f"Document processing completed. Found {len(chunks)} chunks.")
                            
                            # Show sample of what was processed
                            if chunks:
                                sample_sources = list(set([chunk['source'] for chunk in chunks[:10]]))
                                st.info(f"Successfully processed: {sample_sources}")
                            
                        except Exception as e:
                            st.error(f"Error during document processing: {e}")
                            import traceback
                            st.error(f"Full error: {traceback.format_exc()}")
                            return
                        
                        if not chunks:
                            st.error("No text chunks extracted from documents.")
                            st.info(f"Looking in: {doc_processor.docs_directory}")
                            available_docs = doc_processor.get_available_documents()
                            st.info(f"Available documents: {available_docs}")
                            return
                        
                        added_count = rag_system.build_database_from_chunks(chunks)
                        if added_count == 0:
                            st.error("Failed to build vector database. Check OpenAI API key and connection.")
                            return
                        
                        rag_system.save_database()
                        st.success(f"Built knowledge base with {added_count} document chunks")
                
                # Show database info
                db_info = rag_system.get_database_info()
                st.info(f"Knowledge base: {db_info['num_chunks']} chunks from {len(db_info['sources'])} documents")
                
            except Exception as e:
                st.error(f"Error initializing RAG system: {e}")
                return
        
        # Generate responses
        responses_data = []
        benchmark_data = {
            'run_id': f"rag_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'model': 'RAG_Local',
            'system_prompt': system_prompt or "Default TCM RAG system prompt",
            'total_prompts': len(prompts),
            'total_tokens': 0,
            'total_cost': 0.0,
            'responses': []
        }
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, prompt in enumerate(prompts):
            status_text.text(f"Processing prompt {i+1}/{len(prompts)}: {prompt[:50]}...")
            progress_bar.progress((i + 1) / len(prompts))
            
            try:
                # Query RAG system
                result = rag_system.query(prompt, top_n=3, system_prompt=system_prompt)
                
                response_data = {
                    'prompt': prompt,
                    'response': result['response'],
                    'model': 'RAG_Local',
                    'timestamp': datetime.now().isoformat(),
                    'prompt_number': i + 1,  # Add prompt_number for rating compatibility
                    'retrieved_chunks': result['retrieved_chunks'],
                    'num_chunks_used': result['num_chunks_used']
                }
                
                responses_data.append(response_data)
                benchmark_data['responses'].append(response_data)
                
            except Exception as e:
                st.error(f"Error processing prompt {i+1}: {e}")
                response_data = {
                    'prompt': prompt,
                    'response': f"Error: {e}",
                    'model': 'RAG_Local',
                    'timestamp': datetime.now().isoformat(),
                    'prompt_number': i + 1,  # Add prompt_number for rating compatibility
                    'error': str(e)
                }
                responses_data.append(response_data)
                benchmark_data['responses'].append(response_data)
        
        progress_bar.progress(1.0)
        status_text.text("Generation complete!")
        
        # Save results using the same format as ChatGPT responses
        try:
            from file_manager import save_responses_to_json, save_benchmark_data
            
            # Format responses_data as a dictionary like ChatGPT responses
            formatted_responses = {
                'run_id': benchmark_data['run_id'],
                'model': 'RAG_Local',
                'system_prompt': benchmark_data['system_prompt'],
                'timestamp': datetime.now().isoformat(),
                'total_prompts': len(responses_data),
                'responses': responses_data
            }
            
            responses_file = save_responses_to_json(formatted_responses, run_id=benchmark_data['run_id'])
            benchmark_file = save_benchmark_data(benchmark_data)
            
            # Display results
            self.ui.render_generation_results(
                responses_file, benchmark_file, benchmark_data
            )
            
            # Show RAG-specific summary
            st.subheader("📚 RAG Generation Summary")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Responses Generated", len(responses_data))
            with col2:
                st.metric("Knowledge Sources", len(db_info['sources']))
            with col3:
                st.metric("Document Chunks", db_info['num_chunks'])
            
            # Show sample retrieved context
            if responses_data and 'retrieved_chunks' in responses_data[0]:
                with st.expander("Sample Retrieved Context", expanded=False):
                    sample_chunks = responses_data[0]['retrieved_chunks'][:2]
                    for i, chunk in enumerate(sample_chunks):
                        # Ensure chunk is a dictionary with expected structure
                        if isinstance(chunk, dict):
                            metadata = chunk.get('metadata', {})
                            source = metadata.get('source', 'Unknown') if isinstance(metadata, dict) else 'Unknown'
                            similarity = chunk.get('similarity', 0.0)
                            text = chunk.get('text', '')
                            
                            st.write(f"**Source {i+1}:** {source}")
                            st.write(f"**Similarity:** {similarity:.3f}")
                            st.write(f"**Content:** {text[:200]}...")
                            st.divider()
                        else:
                            st.write(f"**Chunk {i+1}:** {str(chunk)[:200]}...")
                            st.divider()
            
        except Exception as e:
            st.error(f"Error saving results: {e}")
            st.json(responses_data)  # Show data even if saving fails
    
    def render_view_tab(self):
        """Render the View Responses tab."""
        st.header("📋 View Saved Responses")
        
        responses_data = load_responses_from_json()
        
        if responses_data:
            display_responses(responses_data)
            st.divider()
            render_download_button()
        else:
            st.info("No saved responses found. Generate some responses first!")
    
    def render_rating_tab(self):
        """Render the Rate Responses tab."""
        rating_interface = SimpleRatingInterface(self.config)
        rating_interface.render_rating_interface()
    
    def render_statistics_tab(self):
        """Render the Statistics tab."""
        rating_interface = SimpleRatingInterface(self.config)
        rating_interface.render_statistics_view()
    
    def _handle_batch_submission(self, api_key: str, system_prompt: str, prompts: list):
        """Handle batch job submission."""
        # Validation
        if not api_key:
            self.ui.render_error_state("api_key", "")
            return
        
        if not prompts:
            self.ui.render_error_state("no_prompts", "")
            return
        
        # Batch submission process
        with self.ui.render_loading_state("Submitting batch job to OpenAI..."):
            try:
                batch_job_id = batch_manager.submit_batch_job(
                    api_key=api_key,
                    system_prompt=system_prompt,
                    prompts=prompts
                )
                
                st.success(f"Batch job submitted successfully! Job ID: {batch_job_id[:8]}...")
                st.info("Your batch will be processed within 24 hours. Check the monitoring section below for updates.")
                
            except Exception as e:
                self.ui.render_error_state("batch_submission", str(e))
    
    def _render_batch_monitoring(self, api_key: str):
        """Render batch job monitoring interface."""
        active_jobs = batch_manager.get_active_jobs()
        completed_jobs = [job for job in batch_manager.get_all_jobs() if job.get('status') == 'completed']
        
        # Active jobs monitoring
        action = self.ui.render_batch_job_monitor(active_jobs)
        
        if action:
            if action['action'] == 'check_status':
                with st.spinner("Checking job status..."):
                    try:
                        status_info = batch_manager.update_job_status(
                            action['job']['batch_job_id'], api_key
                        )
                        st.success(f"Status updated: {status_info['status']}")
                        
                        # Auto-download if completed
                        if status_info['status'] == 'completed':
                            with st.spinner("Auto-downloading completed results..."):
                                try:
                                    responses_data, benchmark_data = batch_manager.download_results(
                                        action['job']['batch_job_id'], api_key
                                    )
                                    st.success("Results automatically downloaded and saved!")
                                    display_benchmark_summary(benchmark_data)
                                except Exception as e:
                                    st.warning(f"Auto-download failed: {e}. Use manual download button.")
                        
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error checking status: {e}")
            
            elif action['action'] == 'download':
                with st.spinner("Downloading and processing results..."):
                    try:
                        responses_data, benchmark_data = batch_manager.download_results(
                            action['job']['batch_job_id'], api_key
                        )
                        st.success("Results downloaded and saved successfully!")
                        
                        # Show results summary
                        display_benchmark_summary(benchmark_data)
                        
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error downloading results: {e}")
        
        # Completed jobs section
        if completed_jobs:
            st.subheader("Completed Jobs")
            for job in completed_jobs[:5]:  # Show last 5 completed jobs
                with st.expander(f"Job {job['batch_job_id'][:8]}... - Completed", expanded=False):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**Run ID:** {job['run_id']}")
                        st.write(f"**Model:** {job['model']}")
                        st.write(f"**Prompts:** {len(job['prompts'])}")
                    with col2:
                        completed_at = job.get('completed_at', 'Unknown')
                        if isinstance(completed_at, (int, float)):
                            # Convert timestamp to readable format
                            completed_at = datetime.fromtimestamp(completed_at).strftime('%Y-%m-%d %H:%M:%S')
                        elif isinstance(completed_at, str) and len(completed_at) > 19:
                            completed_at = completed_at[:19]
                        st.write(f"**Completed:** {completed_at}")
                        if job.get('responses_file'):
                            st.write("✅ Results saved to files")
                        else:
                            if st.button(f"Download Results", key=f"download_completed_{job['batch_job_id']}"):
                                with st.spinner("Downloading results..."):
                                    try:
                                        responses_data, benchmark_data = batch_manager.download_results(
                                            job['batch_job_id'], api_key
                                        )
                                        st.success("Results downloaded!")
                                        st.rerun()
                                    except Exception as e:
                                        st.error(f"Error: {e}")
