"""Simplified file-based rating interface for yQi evaluation platform."""

import streamlit as st
from typing import Dict, List, Any, Optional
from rating_manager import RatingManager
from core.config import AppConfig


class SimpleRatingInterface:
    """File-based rating interface without database dependencies."""
    
    def __init__(self, config: AppConfig):
        self.config = config
        self.rating_manager = RatingManager(config)
        
        # Initialize session state
        if 'current_response_index' not in st.session_state:
            st.session_state.current_response_index = 0
        if 'rater_name' not in st.session_state:
            st.session_state.rater_name = ""
    
    def render_rating_interface(self):
        """Render the main rating interface."""
        st.title("Response Rating Interface")
        
        # Sidebar for navigation and settings
        with st.sidebar:
            self._render_sidebar()
        
        # Main content area
        if 'selected_run' in st.session_state and st.session_state.selected_run:
            self._render_rating_content()
        else:
            self._render_run_selection()
    
    def _render_sidebar(self):
        """Render the sidebar with navigation and settings."""
        st.header("Settings")
        
        # Rater name input
        st.session_state.rater_name = st.text_input(
            "Rater Name", 
            value=st.session_state.rater_name,
            help="Enter your name for rating attribution"
        )
        
        st.divider()
        
        # Navigation
        if 'responses_to_rate' in st.session_state:
            st.header("Navigation")
            
            # Exit button
            if st.button("← Exit Rating", type="secondary"):
                self._exit_rating_session()
                st.rerun()
            
            st.divider()
            
            responses = st.session_state.responses_to_rate
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("← Previous", disabled=st.session_state.current_response_index == 0):
                    st.session_state.current_response_index -= 1
                    st.rerun()
            
            with col2:
                if st.button("Next →", disabled=st.session_state.current_response_index >= len(responses) - 1):
                    st.session_state.current_response_index += 1
                    st.rerun()
            
            # Progress indicator
            progress = (st.session_state.current_response_index + 1) / len(responses)
            st.progress(progress)
            st.caption(f"Response {st.session_state.current_response_index + 1} of {len(responses)}")
            
            # Show rating progress
            run_id = st.session_state.selected_run['run_id']
            existing_ratings = self.rating_manager.get_ratings_for_run(run_id)
            rated_count = len(existing_ratings)
            st.metric("Rated Responses", f"{rated_count}/{len(responses)}")
            
            # Check if all responses are rated
            if rated_count >= len(responses):
                st.success("All responses rated!")
                if st.button("Complete Rating Session", type="primary"):
                    self._exit_rating_session()
                    st.success("Rating session completed successfully!")
                    st.rerun()
    
    def _render_run_selection(self):
        """Render the evaluation run selection interface."""
        st.header("Select Evaluation Run to Rate")
        
        # Get available runs
        runs = self.rating_manager.get_available_runs()
        
        if not runs:
            st.warning("No evaluation runs found. Please generate some responses first.")
            return
        
        # Display runs in a table format
        for run in runs:
            with st.container():
                col1, col2, col3, col4, col5 = st.columns([3, 2, 2, 1, 1])
                
                with col1:
                    st.write(f"**Run ID:** {run['run_id']}")
                    st.caption(f"Total Prompts: {run['total_prompts']}")
                
                with col2:
                    if run['timestamp']:
                        st.write(f"**Created:** {run['timestamp'][:16]}")
                
                with col3:
                    st.write(f"**Rated:** {run['rated_count']}/{run['total_prompts']}")
                    if run['total_prompts'] > 0:
                        completion = min(1.0, run['rated_count'] / run['total_prompts'])
                        st.progress(completion)
                
                with col4:
                    if st.button("Rate", key=f"rate_{run['run_id']}"):
                        st.session_state.selected_run = run
                        st.session_state.responses_to_rate = run['responses']
                        st.session_state.current_response_index = 0
                        st.rerun()
                
                with col5:
                    if st.button("🗑️", key=f"delete_{run['run_id']}", help="Delete this run", type="secondary"):
                        # Show confirmation dialog
                        st.session_state[f"confirm_delete_{run['run_id']}"] = True
                        st.rerun()
                
                # Handle delete confirmation
                if st.session_state.get(f"confirm_delete_{run['run_id']}", False):
                    st.warning(f"⚠️ Are you sure you want to delete run **{run['run_id']}**?")
                    st.caption("This will permanently delete all responses, ratings, and benchmark data for this run.")
                    
                    col_confirm, col_cancel = st.columns(2)
                    with col_confirm:
                        if st.button("Yes, Delete", key=f"confirm_yes_{run['run_id']}", type="primary"):
                            if self.rating_manager.delete_run(run['run_id']):
                                st.success(f"Successfully deleted run {run['run_id']}")
                                # Clear confirmation state
                                del st.session_state[f"confirm_delete_{run['run_id']}"]
                                st.rerun()
                            else:
                                st.error(f"Failed to delete run {run['run_id']}")
                    
                    with col_cancel:
                        if st.button("Cancel", key=f"confirm_no_{run['run_id']}", type="secondary"):
                            # Clear confirmation state
                            del st.session_state[f"confirm_delete_{run['run_id']}"]
                            st.rerun()
                
                st.divider()
    
    def _render_rating_content(self):
        """Render the rating interface for responses."""
        responses = st.session_state.responses_to_rate
        
        if not responses:
            st.warning("No responses found for this evaluation run.")
            if st.button("← Back to Run Selection"):
                del st.session_state.selected_run
                st.rerun()
            return
        
        current_response = responses[st.session_state.current_response_index]
        
        # Display response content
        prompt_number = current_response.get('prompt_number', st.session_state.current_response_index + 1)
        st.header(f"Response {prompt_number}")
        
        # Show prompt
        with st.expander("Prompt", expanded=True):
            st.write(current_response['prompt'])
        
        # Show response
        with st.expander("AI Response", expanded=True):
            st.write(current_response['response'])
        
        # Show response metadata
        with st.expander("Response Metadata"):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Tokens Used", current_response.get('tokens_used', 0))
            with col2:
                st.metric("Timestamp", current_response.get('timestamp', 'N/A')[:16])
            with col3:
                prompt_number = current_response.get('prompt_number', st.session_state.current_response_index + 1)
                st.metric("Prompt Number", prompt_number)
        
        # Check if all responses are rated and auto-close
        run_id = st.session_state.selected_run['run_id']
        existing_ratings = self.rating_manager.get_ratings_for_run(run_id)
        total_responses = len(responses)
        rated_count = len(existing_ratings)
        
        # Auto-close if all responses are rated
        if rated_count >= total_responses:
            st.success("🎉 All responses have been rated!")
            st.info("Rating session completed.")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("View Statistics", type="primary"):
                    # Switch to statistics view
                    st.session_state.show_statistics = True
                    self._exit_rating_session()
                    st.rerun()
            
            with col2:
                if st.button("Return to Menu", type="secondary"):
                    self._exit_rating_session()
                    st.rerun()
            
            return
        
        # Show existing ratings for current response
        prompt_ratings = [r for r in existing_ratings if r['prompt_number'] == current_response['prompt_number']]
        
        if prompt_ratings:
            st.subheader("Existing Ratings")
            for i, rating in enumerate(prompt_ratings):
                with st.expander(f"Rating by {rating.get('rater_name', 'Anonymous')} - Overall: {rating.get('overall_rating', 'N/A')}/5"):
                    self._display_rating(rating)
        
        # Rating form
        st.subheader("Add Your Rating")
        self._render_rating_form(current_response)
    
    def _render_rating_form(self, response: Dict[str, Any]):
        """Render the rating form."""
        if not st.session_state.rater_name.strip():
            st.warning("Please enter your name in the sidebar before rating.")
            return
        
        with st.form(f"rating_form_{response['prompt_number']}"):
            st.write("**Rate this response on a scale of 1-5:**")
            
            # Overall rating (required)
            overall_rating = st.slider(
                "Overall Quality",
                min_value=1, max_value=5, value=3,
                help="Overall assessment of the response quality"
            )
            
            # Specific criteria ratings
            accuracy_rating = st.slider(
                "Medical Accuracy",
                min_value=1, max_value=5, value=3,
                help="Accuracy of TCM diagnosis and treatment recommendations"
            )
            
            completeness_rating = st.slider(
                "Completeness",
                min_value=1, max_value=5, value=3,
                help="How complete is the analysis and prescription"
            )
            
            clarity_rating = st.slider(
                "Clarity",
                min_value=1, max_value=5, value=3,
                help="Clarity and organization of the response"
            )
            
            clinical_relevance_rating = st.slider(
                "Clinical Relevance",
                min_value=1, max_value=5, value=3,
                help="Relevance to actual clinical practice"
            )
            
            safety_rating = st.slider(
                "Safety",
                min_value=1, max_value=5, value=3,
                help="Safety considerations in the recommendations"
            )
            
            # Comments
            comments = st.text_area(
                "Comments (Optional)",
                help="Additional comments about this response"
            )
            
            # Submit button
            submitted = st.form_submit_button("Submit Rating", type="primary")
            
            if submitted:
                try:
                    rating_data = {
                        'rater_name': st.session_state.rater_name,
                        'overall_rating': overall_rating,
                        'accuracy_rating': accuracy_rating,
                        'completeness_rating': completeness_rating,
                        'clarity_rating': clarity_rating,
                        'clinical_relevance_rating': clinical_relevance_rating,
                        'safety_rating': safety_rating,
                        'comments': comments if comments.strip() else None
                    }
                    
                    run_id = st.session_state.selected_run['run_id']
                    rating_id = self.rating_manager.save_rating(
                        run_id, response['prompt_number'], rating_data
                    )
                    
                    st.success("Rating saved successfully!")
                    
                    # Check if all responses are now rated
                    run_id = st.session_state.selected_run['run_id']
                    updated_ratings = self.rating_manager.get_ratings_for_run(run_id)
                    total_responses = len(st.session_state.responses_to_rate)
                    
                    if len(updated_ratings) >= total_responses:
                        # All responses rated - show completion message
                        st.success("🎉 All responses have been rated! Rating session complete.")
                        st.balloons()
                        # Will auto-close on next render
                    else:
                        # Auto-advance to next response
                        if st.session_state.current_response_index < len(st.session_state.responses_to_rate) - 1:
                            st.session_state.current_response_index += 1
                    
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Error saving rating: {e}")
    
    def _exit_rating_session(self):
        """Clean up rating session and return to menu."""
        # Clear rating session state
        if 'selected_run' in st.session_state:
            del st.session_state.selected_run
        if 'responses_to_rate' in st.session_state:
            del st.session_state.responses_to_rate
        if 'current_response_index' in st.session_state:
            st.session_state.current_response_index = 0
    
    def _display_rating(self, rating: Dict[str, Any]):
        """Display an existing rating."""
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**Overall:** {rating.get('overall_rating', 'N/A')}/5")
            st.write(f"**Accuracy:** {rating.get('accuracy_rating', 'N/A')}/5")
            st.write(f"**Completeness:** {rating.get('completeness_rating', 'N/A')}/5")
        
        with col2:
            st.write(f"**Clarity:** {rating.get('clarity_rating', 'N/A')}/5")
            st.write(f"**Clinical Relevance:** {rating.get('clinical_relevance_rating', 'N/A')}/5")
            st.write(f"**Safety:** {rating.get('safety_rating', 'N/A')}/5")
        
        if rating.get('comments'):
            st.write(f"**Comments:** {rating['comments']}")
        
        st.caption(f"Rated on: {rating.get('timestamp', 'Unknown')[:16]}")
    
    def render_statistics_view(self):
        """Render rating statistics and analytics."""
        st.title("Rating Statistics")
        
        # Run selection for statistics
        runs = self.rating_manager.get_available_runs()
        run_options = ["All Runs"] + [f"{run['run_id']} ({run['total_prompts']} prompts)" for run in runs]
        
        selected_option = st.selectbox("Select Run for Statistics", run_options)
        
        if selected_option == "All Runs":
            stats = self.rating_manager.get_rating_statistics()
        else:
            run_id = selected_option.split(" ")[0]
            stats = self.rating_manager.get_rating_statistics(run_id)
        
        if not stats:
            st.warning("No ratings found for the selected run(s).")
            return
        
        # Display statistics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Ratings", stats['total_ratings'])
        
        with col2:
            st.metric("Average Overall Rating", f"{stats['average_overall_rating']:.2f}/5")
        
        with col3:
            if stats.get('criteria_averages'):
                best_criteria = max(stats['criteria_averages'].items(), key=lambda x: x[1])
                st.metric("Best Criteria", f"{best_criteria[0].replace('_', ' ').title()}: {best_criteria[1]:.2f}")
        
        # Rating distribution
        if stats.get('rating_distribution'):
            st.subheader("Rating Distribution")
            
            # Simple bar chart using Streamlit's built-in chart
            import pandas as pd
            
            dist_data = pd.DataFrame([
                {'Rating': k, 'Count': v} 
                for k, v in stats['rating_distribution'].items()
            ])
            
            st.bar_chart(dist_data.set_index('Rating'))
        
        # Criteria averages
        if stats.get('criteria_averages'):
            st.subheader("Average Ratings by Criteria")
            
            criteria_data = pd.DataFrame([
                {'Criteria': k.replace('_', ' ').title(), 'Average': v}
                for k, v in stats['criteria_averages'].items()
            ])
            
            st.bar_chart(criteria_data.set_index('Criteria'))
