#!/usr/bin/env python3
"""
Three-Tier Ticket Classification Demo - Enhanced Streamlit App
Production-ready demonstration using our advanced three-tier system
"""

import streamlit as st
import pandas as pd
import time
import sys
from pathlib import Path

# Import our new three-tier demo engine
from three_tier_classification_engine import ThreeTierDemoEngine

# Page configuration
st.set_page_config(
    page_title="🎯 Advanced IT Ticket Classification Demo",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'classification_engine' not in st.session_state:
    st.session_state.classification_engine = None
if 'engine_ready' not in st.session_state:
    st.session_state.engine_ready = False

def main():
    """Main application"""
    
    st.title("🎯 Advanced IT Ticket Classification & Automation Analysis")
    st.markdown("*Production-ready three-tier classification system demonstration*")
    
    # Sidebar for system controls
    with st.sidebar:
        st.header("🚀 System Status")
        
        # System initialization
        st.subheader("⚙️ System Controls")
        
        if st.button("🔄 Initialize Three-Tier System", type="primary"):
            with st.spinner("Initializing advanced classification system..."):
                try:
                    st.session_state.classification_engine = ThreeTierDemoEngine(
                        use_embeddings=True, 
                        use_llm=False  # Disable LLM for demo stability
                    )
                    st.session_state.engine_ready = True
                    st.success("🎯 Three-tier classification system ready!")
                    
                except Exception as e:
                    st.error(f"❌ System initialization failed: {e}")
                    st.session_state.engine_ready = False
        
        # Show system status
        if st.session_state.engine_ready:
            st.success("✅ System Status: READY")
            
            st.markdown("**🔧 System Components:**")
            st.markdown("✅ Level 1: Business Classification")
            st.markdown("✅ Level 2: Semantic Problem Search") 
            st.markdown("✅ Level 3: Automation Analysis")
            st.markdown("✅ Manager Requirements Integration")
            
        else:
            st.warning("⚠️ System Status: NOT READY")
            st.info("👈 Click 'Initialize Three-Tier System' to begin")
        
        st.divider()
        
        # Advanced system information
        if st.session_state.engine_ready:
            st.subheader("📊 System Information")
            
            with st.expander("🔍 Technical Details"):
                st.markdown("""
                **Architecture:** Three-Tier Classification Pipeline
                
                **Level 1 - Business Routing:**
                - 10 business categories
                - Semantic embedding classification
                - 77% accuracy target
                - <50ms processing time
                
                **Level 2 - Problem Identification:**
                - 1,683 specific problems in database
                - Cosine similarity search
                - Business category context filtering
                - <5ms search time
                
                **Level 3 - Automation Analysis:**
                - 5-layer hybrid analysis approach
                - Historical database (209 + 1,203 mappings)
                - Manager requirement compliance
                - FULLY/PARTIALLY/NOT_AUTOMATABLE categories
                - Percentage effort savings estimates
                """)
    
    # Main content area
    if not st.session_state.engine_ready:
        st.info("👈 **Start by clicking 'Initialize Three-Tier System' in the sidebar**")
        
        st.markdown("## 🚀 Advanced Three-Tier Classification System")
        st.markdown("""
        This demo showcases our production-ready **three-tier classification pipeline** with comprehensive automation analysis:
        
        ### 🏗️ **System Architecture**
        """)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**📊 Level 1: Business Routing**")
            st.markdown("- 10 business categories")
            st.markdown("- Semantic classification")
            st.markdown("- Team routing & SLA assignment")
            st.markdown("- Priority determination")
            
        with col2:
            st.markdown("**🔍 Level 2: Problem Search**")
            st.markdown("- 1,683+ specific problems")
            st.markdown("- Context-aware semantic search") 
            st.markdown("- Similarity-based matching")
            st.markdown("- Business category filtering")
        
        with col3:
            st.markdown("**🤖 Level 3: Automation Analysis**")
            st.markdown("- 5-layer hybrid analysis")
            st.markdown("- Manager requirement compliance")
            st.markdown("- Percentage effort savings")
            st.markdown("- Implementation complexity")
        
        st.markdown("### 📋 **Example Test Cases**")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**🟢 Account Management**")
            st.code("unlock user account john.doe")
            st.caption("Expected: FULLY_AUTOMATABLE (95%)")
        
        with col2:
            st.markdown("**🟡 System Diagnostics**") 
            st.code("till crashed during busy period")
            st.caption("Expected: PARTIALLY_AUTOMATABLE (60%)")
        
        with col3:
            st.markdown("**🔴 Hardware Issues**")
            st.code("replace broken CPU on server")
            st.caption("Expected: NOT_AUTOMATABLE (10%)")
    
    else:
        # System is ready - show main demo interface
        show_three_tier_demo_interface()

def show_three_tier_demo_interface():
    """Show the advanced three-tier demo interface"""
    
    st.success("✅ Three-tier classification system ready for demonstration!")
    
    # Performance metrics display
    with st.expander("📈 Expected Performance Metrics"):
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Level 1 Accuracy", "77%", help="Business category classification accuracy")
        with col2:
            st.metric("Level 2 Coverage", "100%", help="Problem identification success rate")
        with col3:
            st.metric("Level 3 Coverage", "100%", help="Automation analysis coverage")
        with col4:
            st.metric("Total Response", "<100ms", help="End-to-end processing time")
    
    # Ticket classification section
    st.header("📝 Advanced Ticket Classification")
    
    # Input area with enhanced examples
    col1, col2 = st.columns([3, 1])
    
    with col1:
        ticket_description = st.text_area(
            "Enter IT ticket description:",
            height=120,
            placeholder="Example: Till 3 scanner not working, shows error message when trying to scan barcodes, customers waiting in queue",
            help="Enter a detailed ticket description to see the complete three-tier analysis"
        )
    
    with col2:
        st.markdown("**📋 Test Examples:**")
        
        # Enhanced test cases that demonstrate system capabilities
        enhanced_examples = [
            ("🔒 Account Issues", "cashier sarah locked on till 3 customers waiting"),
            ("📦 Vision Orders", "unable to amend urgent vision order quantities locked"),
            ("🖨️ Printer Problems", "till printer failed to initialize error code E1234"),
            ("🔧 Hardware Failure", "replace broken motherboard server room 2"),
            ("🔑 Access Control", "password reset needed urgent cannot login"),
            ("💳 Payment Issues", "chip pin reader offline customers cannot pay"),
            ("🏪 Store Operations", "back office manager reports system very slow"),
            ("🖥️ System Performance", "investigate network performance issues affecting all tills")
        ]
        
        for label, example in enhanced_examples:
            if st.button(label, help=f"Try: {example}", key=f"example_{label}"):
                st.session_state.example_text = example
                st.rerun()
        
        # Use example if button was clicked
        if 'example_text' in st.session_state:
            ticket_description = st.session_state.example_text
            del st.session_state.example_text
    
    # Enhanced classification processing with detailed progress
    if st.button("🚀 Analyze with Three-Tier System", type="primary", disabled=not ticket_description.strip()):
        
        # Enhanced progress display
        progress_container = st.container()
        with progress_container:
            progress_bar = st.progress(0)
            status_text = st.empty()
            time_display = st.empty()
            
            start_time = time.time()
            
            # Level 1: Business Classification
            progress_bar.progress(20)
            status_text.text("🎯 Level 1: Business category classification & routing...")
            time_display.text(f"Processing time: {(time.time() - start_time)*1000:.0f}ms")
            time.sleep(0.3)
            
            # Level 2: Problem Search
            progress_bar.progress(50)
            status_text.text("🔍 Level 2: Semantic problem search & matching...")
            time_display.text(f"Processing time: {(time.time() - start_time)*1000:.0f}ms")
            time.sleep(0.3)
            
            # Level 3: Automation Analysis
            progress_bar.progress(80)
            status_text.text("🤖 Level 3: Comprehensive automation analysis...")
            time_display.text(f"Processing time: {(time.time() - start_time)*1000:.0f}ms")
            time.sleep(0.2)
            
            # Final processing
            progress_bar.progress(100)
            status_text.text("✨ Finalizing three-tier analysis results...")
            
            # Perform actual classification
            result = st.session_state.classification_engine.classify_ticket(ticket_description)
            
            processing_time = (time.time() - start_time) * 1000
            time_display.text(f"Total processing time: {processing_time:.1f}ms")
            time.sleep(0.5)
            
            # Clear progress indicators
            progress_container.empty()
        
        # Display comprehensive results
        display_three_tier_results(result, ticket_description, processing_time)

def display_three_tier_results(result, original_input, processing_time):
    """Display comprehensive three-tier classification results"""
    
    st.success("🎉 Three-Tier Analysis Complete!")
    
    # Main results overview
    st.subheader("📊 Analysis Results Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        method_icon = {
            "Business Classification": "🎯",
            "Semantic Clustering": "🧠", 
            "Hardcoded Pattern": "📋",
            "Three-Tier Analysis": "⚡",
            "Error": "❌"
        }.get(result.method, "🔍")
        
        st.metric(
            f"{method_icon} Classification Method", 
            result.method,
            help="Primary method used for classification"
        )
    
    with col2:
        # Enhanced confidence display with color coding
        confidence_pct = result.confidence_score * 100
        confidence_color = "🟢" if confidence_pct >= 80 else "🟡" if confidence_pct >= 60 else "🔴"
        
        st.metric(
            f"{confidence_color} Overall Confidence",
            f"{confidence_pct:.1f}%",
            help="System confidence in the complete analysis"
        )
    
    with col3:
        # Enhanced automation display
        automation_colors = {
            'FULLY_AUTOMATABLE': '🟢',
            'PARTIALLY_AUTOMATABLE': '🟡', 
            'NOT_AUTOMATABLE': '🔴',
            'Unknown': '⚪'
        }
        
        automation_color = automation_colors.get(result.automation_potential, '⚪')
        automation_pct = result.automation_percentage or 0
        
        st.metric(
            f"{automation_color} Automation Potential",
            f"{result.automation_potential.replace('_', ' ').title()}",
            delta=f"{automation_pct}% effort savings",
            help="Automation feasibility and effort savings estimate"
        )
    
    with col4:
        st.metric(
            "⚡ Processing Speed",
            f"{processing_time:.1f}ms",
            delta="Target: <100ms",
            delta_color="normal" if processing_time < 100 else "inverse",
            help="Total system response time"
        )
    
    # Level-by-level analysis
    st.subheader("🏗️ Detailed Level-by-Level Analysis")
    
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Level 1: Business", "🔍 Level 2: Problem", "🤖 Level 3: Automation", "⚡ Performance"])
    
    with tab1:
        st.markdown("**🎯 Business Classification & Routing**")
        
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"**Business Category:** {result.business_category}")
            st.info(f"**Routing Team:** {result.routing_team}")
            st.info(f"**Priority Level:** {result.priority_level}")
            st.info(f"**SLA Requirement:** {result.sla_hours} hours")
        
        with col2:
            urgency = result.details.get('urgency_score', 0)
            st.metric("Urgency Score", f"{urgency:.1%}")
            st.metric("Level 1 Processing", f"{result.details.get('level1_time_ms', 0):.1f}ms")
            
            # Recommendation display
            recommendation = result.details.get('recommendation', 'ROUTE_TO_TEAM')
            st.write(f"**System Recommendation:** {recommendation.replace('_', ' ').title()}")
    
    with tab2:
        st.markdown("**🔍 Problem Identification & Matching**")
        
        if result.details.get('specific_problem'):
            st.success(f"**Identified Problem:** {result.details['specific_problem']}")
            
            # Problem matching details
            col1, col2 = st.columns(2)
            with col1:
                problem_confidence = result.details.get('problem_confidence', 0)
                st.metric("Problem Match Confidence", f"{problem_confidence:.1%}")
                st.metric("Level 2 Processing", f"{result.details.get('level2_time_ms', 0):.1f}ms")
            
            with col2:
                similar_count = result.details.get('similar_problems_count', 0)
                st.metric("Similar Problems Found", similar_count)
                
                # Show search method if available
                if 'automation_method' in result.details:
                    st.write(f"**Search Method:** {result.details['automation_method']}")
        else:
            st.warning("**Problem Identification:** Using business category classification")
    
    with tab3:
        st.markdown("**🤖 Automation Analysis & Implementation**")
        
        # Automation assessment
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Automation Assessment:**")
            st.write(result.automation_reasoning)
            
            # Implementation details
            if 'implementation_assessment' in result.details:
                st.write("**Implementation:**") 
                st.write(result.details['implementation_assessment'])
        
        with col2:
            # Automation metrics
            automation_confidence = result.automation_confidence
            st.metric("Automation Confidence", f"{automation_confidence:.1%}")
            st.metric("Level 3 Processing", f"{result.details.get('level3_time_ms', 0):.1f}ms")
            
            # Business priority and complexity
            business_priority = result.details.get('business_priority', 'MEDIUM')
            complexity = result.details.get('implementation_complexity', 'MODERATE')
            
            st.write(f"**Business Priority:** {business_priority}")
            st.write(f"**Implementation Complexity:** {complexity}")
        
        # ROI estimation
        if 'roi_estimate' in result.details and result.details['roi_estimate']:
            st.info(f"**💰 ROI Estimate:** {result.details['roi_estimate']}")
    
    with tab4:
        st.markdown("**⚡ Performance Analysis**")
        
        # Performance breakdown
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Processing Time Breakdown:**")
            level1_time = result.details.get('level1_time_ms', 0)
            level2_time = result.details.get('level2_time_ms', 0) 
            level3_time = result.details.get('level3_time_ms', 0)
            
            st.write(f"- Level 1 (Business): {level1_time:.1f}ms")
            st.write(f"- Level 2 (Problem): {level2_time:.1f}ms")
            st.write(f"- Level 3 (Automation): {level3_time:.1f}ms")
            st.write(f"- **Total: {processing_time:.1f}ms**")
        
        with col2:
            overall_confidence = result.details.get('overall_confidence', 0)
            st.metric("Overall System Confidence", f"{overall_confidence:.1%}")
            
            # Performance assessment
            if processing_time < 50:
                st.success("🚀 Excellent performance")
            elif processing_time < 100:
                st.info("✅ Good performance") 
            else:
                st.warning("⚠️ Performance optimization needed")
    
    # Technical details for advanced users
    with st.expander("🔬 Advanced Technical Details"):
        st.write("**Original Input:**")
        st.code(original_input)
        
        st.write("**Complete Analysis Details:**")
        
        # Create formatted details table
        tech_details = []
        for key, value in result.details.items():
            if isinstance(value, (list, tuple)):
                value = f"[{len(value)} items]"
            elif isinstance(value, dict):
                value = f"{{...{len(value)} fields...}}"
            elif isinstance(value, float):
                value = f"{value:.3f}"
            
            tech_details.append({
                "Field": key.replace('_', ' ').title(),
                "Value": str(value)[:100] + ("..." if len(str(value)) > 100 else "")
            })
        
        if tech_details:
            tech_df = pd.DataFrame(tech_details)
            st.dataframe(tech_df, hide_index=True, use_container_width=True)

if __name__ == "__main__":
    main()