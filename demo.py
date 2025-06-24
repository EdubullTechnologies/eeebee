import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import random
import time
from streamlit_autorefresh import st_autorefresh

# Set page config
st.set_page_config(
    page_title="AI on Wall - School Analytics Dashboard",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1e3d59;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 1rem;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        height: 100%;
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: bold;
        margin: 0.5rem 0;
    }
    .metric-label {
        font-size: 1rem;
        opacity: 0.9;
    }
    .alert-box {
        background-color: #fee;
        border-left: 4px solid #f66;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0.5rem;
    }
    .success-box {
        background-color: #efe;
        border-left: 4px solid #4a4;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0.5rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
        background-color: #f0f2f6;
        border-radius: 8px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1e3d59;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'last_update' not in st.session_state:
    st.session_state.last_update = datetime.now()

# Generate mock data
@st.cache_data
def generate_mock_data():
    # School structure
    grades = ['Grade 1', 'Grade 2', 'Grade 3', 'Grade 4', 'Grade 5', 
              'Grade 6', 'Grade 7', 'Grade 8', 'Grade 9', 'Grade 10']
    sections = ['A', 'B', 'C']
    subjects = ['Mathematics', 'Science', 'English', 'Social Studies', 'Hindi']
    
    # Generate student data
    students = []
    student_id = 1000
    for grade in grades:
        for section in sections:
            num_students = random.randint(35, 45)
            for i in range(num_students):
                student_id += 1
                students.append({
                    'StudentID': student_id,
                    'Name': f'Student {student_id}',
                    'Grade': grade,
                    'Section': section,
                    'Batch': f'{grade}-{section}',
                    'OverallScore': random.gauss(65, 15),
                    'AttendanceRate': random.gauss(85, 10),
                    'ConceptsClearedRate': random.gauss(70, 20),
                    'WeakConcepts': random.randint(2, 15),
                    'AIInteractions': random.randint(10, 200),
                    'LastActive': datetime.now() - timedelta(hours=random.randint(0, 72))
                })
    
    # Generate teacher data
    teachers = []
    teacher_id = 100
    for grade in grades:
        for subject in subjects:
            teacher_id += 1
            teachers.append({
                'TeacherID': teacher_id,
                'Name': f'Teacher {teacher_id}',
                'Subject': subject,
                'Grade': grade,
                'StudentsCount': random.randint(100, 150),
                'AIToolsUsage': random.randint(20, 100),
                'AssignmentsCreated': random.randint(5, 30),
                'AvgStudentImprovement': random.gauss(15, 5)
            })
    
    # Generate concept performance data
    concepts = []
    concept_id = 1
    for subject in subjects:
        num_topics = random.randint(8, 12)
        for topic_num in range(num_topics):
            num_concepts = random.randint(5, 10)
            for concept_num in range(num_concepts):
                concept_id += 1
                concepts.append({
                    'ConceptID': concept_id,
                    'Subject': subject,
                    'Topic': f'{subject} Topic {topic_num + 1}',
                    'ConceptName': f'Concept {concept_id}',
                    'AvgMastery': random.gauss(65, 20),
                    'StudentsAttempted': random.randint(200, 400),
                    'StudentsCleared': random.randint(100, 350),
                    'AvgTimeToMaster': random.gauss(45, 15)
                })
    
    # Time series data for trends
    dates = pd.date_range(start='2024-01-01', end='2024-01-20', freq='D')
    performance_trends = []
    for date in dates:
        for grade in grades:
            performance_trends.append({
                'Date': date,
                'Grade': grade,
                'AvgScore': random.gauss(65, 5) + (date.day * 0.3),  # Slight upward trend
                'ActiveStudents': random.randint(80, 120),
                'AIUsage': random.randint(100, 300)
            })
    
    return {
        'students': pd.DataFrame(students),
        'teachers': pd.DataFrame(teachers),
        'concepts': pd.DataFrame(concepts),
        'trends': pd.DataFrame(performance_trends)
    }

# Load mock data
data = generate_mock_data()

# Helper function to create metric cards
def create_metric_card(label, value, delta=None, delta_color="normal"):
    delta_html = ""
    if delta is not None:
        color = "green" if delta_color == "normal" and delta > 0 else "red"
        arrow = "↑" if delta > 0 else "↓"
        delta_html = f'<div style="color: {color}; font-size: 1rem;">{arrow} {abs(delta):.1f}%</div>'
    
    return f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        {delta_html}
    </div>
    """

# Header
st.markdown('<h1 class="main-header">🎓 AI on Wall - School Analytics Dashboard</h1>', unsafe_allow_html=True)

# Top metrics row
col1, col2, col3, col4, col5 = st.columns(5)

total_students = len(data['students'])
active_today = len(data['students'][data['students']['LastActive'] > datetime.now() - timedelta(days=1)])
avg_performance = data['students']['OverallScore'].mean()
at_risk_students = len(data['students'][data['students']['OverallScore'] < 40])
ai_interactions_today = data['students']['AIInteractions'].sum()

with col1:
    st.markdown(create_metric_card("Total Students", total_students, 2.5), unsafe_allow_html=True)

with col2:
    st.markdown(create_metric_card("Active Today", active_today, 5.2), unsafe_allow_html=True)

with col3:
    st.markdown(create_metric_card("Avg Performance", f"{avg_performance:.1f}%", 3.1), unsafe_allow_html=True)

with col4:
    st.markdown(create_metric_card("At Risk", at_risk_students, -2.3, "inverse"), unsafe_allow_html=True)

with col5:
    st.markdown(create_metric_card("AI Sessions", f"{ai_interactions_today:,}", 12.5), unsafe_allow_html=True)

# Alert section
st.markdown("---")
alert_col1, alert_col2 = st.columns(2)

with alert_col1:
    st.markdown("""
    <div class="alert-box">
        <strong>⚠️ Attention Required:</strong><br>
        • 23 students in Grade 8-B showing declining performance<br>
        • Mathematics concepts in Grade 7 need reinforcement<br>
        • 5 teachers haven't used AI tools this week
    </div>
    """, unsafe_allow_html=True)

with alert_col2:
    st.markdown("""
    <div class="success-box">
        <strong>✅ Achievements:</strong><br>
        • Grade 10-A achieved 85% mastery in Science<br>
        • Overall school performance improved by 3.1%<br>
        • AI adoption rate increased to 78%
    </div>
    """, unsafe_allow_html=True)

# Main content tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Performance Overview", 
    "👥 Student Analytics", 
    "👨‍🏫 Teacher Effectiveness",
    "🎯 Concept Mastery",
    "🤖 AI Insights",
    "🔍 Natural Language Query"
])

with tab1:
    st.subheader("School-wide Performance Trends")
    
    # Performance heatmap
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Create grade-subject performance matrix
        grade_subject_matrix = []
        for grade in data['students']['Grade'].unique():
            row = {'Grade': grade}
            for subject in ['Mathematics', 'Science', 'English', 'Social Studies', 'Hindi']:
                row[subject] = random.gauss(70, 15)
            grade_subject_matrix.append(row)
        
        matrix_df = pd.DataFrame(grade_subject_matrix)
        matrix_df = matrix_df.set_index('Grade')
        
        fig = px.imshow(matrix_df.values,
                       labels=dict(x="Subject", y="Grade", color="Performance %"),
                       x=matrix_df.columns,
                       y=matrix_df.index,
                       color_continuous_scale="RdYlGn",
                       aspect="auto")
        fig.update_layout(title="Performance Heatmap by Grade and Subject")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Grade distribution pie chart
        grade_distribution = data['students'].groupby('Grade').size()
        fig = px.pie(values=grade_distribution.values, 
                    names=grade_distribution.index,
                    title="Student Distribution by Grade")
        st.plotly_chart(fig, use_container_width=True)
    
    # Time series performance trend
    st.subheader("Performance Trend Over Time")
    
    trend_chart = alt.Chart(data['trends']).mark_line(point=True).encode(
        x='Date:T',
        y='AvgScore:Q',
        color='Grade:N',
        tooltip=['Date:T', 'Grade:N', 'AvgScore:Q']
    ).properties(
        height=400
    )
    
    st.altair_chart(trend_chart, use_container_width=True)

with tab2:
    st.subheader("Student Performance Analytics")
    
    # Student segmentation
    col1, col2 = st.columns(2)
    
    with col1:
        # Performance distribution
        fig = px.histogram(data['students'], x='OverallScore', nbins=20,
                          title="Student Performance Distribution",
                          labels={'OverallScore': 'Overall Score (%)', 'count': 'Number of Students'})
        fig.add_vline(x=40, line_dash="dash", line_color="red", annotation_text="At Risk Threshold")
        fig.add_vline(x=75, line_dash="dash", line_color="green", annotation_text="Excellence Threshold")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Student categorization
        def categorize_student(score):
            if score >= 75:
                return 'Outstanding'
            elif score >= 50:
                return 'Achiever'
            elif score >= 35:
                return 'Average'
            else:
                return 'Need Improvement'
        
        data['students']['Category'] = data['students']['OverallScore'].apply(categorize_student)
        category_counts = data['students']['Category'].value_counts()
        
        fig = px.bar(x=category_counts.index, y=category_counts.values,
                    title="Student Performance Categories",
                    labels={'x': 'Category', 'y': 'Number of Students'},
                    color=category_counts.index,
                    color_discrete_map={
                        'Outstanding': '#4CAF50',
                        'Achiever': '#8BC34A',
                        'Average': '#FFC107',
                        'Need Improvement': '#FF5722'
                    })
        st.plotly_chart(fig, use_container_width=True)
    
    # Student progress funnel
    st.subheader("Student Learning Journey Funnel")
    
    funnel_data = pd.DataFrame({
        'Stage': ['Total Enrolled', 'Active Learners', 'Concepts Attempted', 'Concepts Mastered', 'Excellence Achieved'],
        'Students': [1200, 980, 850, 620, 180]
    })
    
    fig = px.funnel(funnel_data, x='Students', y='Stage',
                   title="Student Progress Through Learning Stages")
    st.plotly_chart(fig, use_container_width=True)
    
    # At-risk students table
    st.subheader("Students Requiring Immediate Attention")
    
    at_risk_df = data['students'][data['students']['OverallScore'] < 40].sort_values('OverallScore')
    at_risk_display = at_risk_df[['Name', 'Grade', 'Section', 'OverallScore', 'WeakConcepts', 'AttendanceRate']].head(10)
    at_risk_display.columns = ['Student Name', 'Grade', 'Section', 'Score (%)', 'Weak Concepts', 'Attendance (%)']
    
    st.dataframe(
        at_risk_display.style.background_gradient(subset=['Score (%)', 'Attendance (%)'], cmap='RdYlGn'),
        use_container_width=True
    )

with tab3:
    st.subheader("Teacher Performance & Effectiveness")
    
    # Teacher effectiveness scatter plot
    fig = px.scatter(data['teachers'], 
                    x='AIToolsUsage', 
                    y='AvgStudentImprovement',
                    size='StudentsCount',
                    color='Subject',
                    title="Teacher Effectiveness: AI Usage vs Student Improvement",
                    labels={'AIToolsUsage': 'AI Tools Usage (Sessions)', 
                           'AvgStudentImprovement': 'Avg Student Improvement (%)'},
                    hover_data=['Name'])
    
    fig.add_hline(y=15, line_dash="dash", line_color="green", annotation_text="Target Improvement")
    st.plotly_chart(fig, use_container_width=True)
    
    # Teacher rankings
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Top Performing Teachers")
        top_teachers = data['teachers'].nlargest(5, 'AvgStudentImprovement')[['Name', 'Subject', 'AvgStudentImprovement']]
        top_teachers.columns = ['Teacher', 'Subject', 'Student Improvement (%)']
        st.dataframe(top_teachers, use_container_width=True)
    
    with col2:
        st.subheader("AI Tool Adoption by Subject")
        ai_adoption = data['teachers'].groupby('Subject')['AIToolsUsage'].mean().sort_values(ascending=True)
        
        fig = px.bar(x=ai_adoption.values, y=ai_adoption.index, orientation='h',
                    title="Average AI Tool Usage by Subject",
                    labels={'x': 'Average Sessions', 'y': 'Subject'})
        st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.subheader("Concept Mastery Analysis")
    
    # Subject-wise concept mastery
    subject_mastery = data['concepts'].groupby('Subject')['AvgMastery'].mean().sort_values(ascending=False)
    
    fig = px.bar(x=subject_mastery.index, y=subject_mastery.values,
                title="Average Concept Mastery by Subject",
                labels={'x': 'Subject', 'y': 'Average Mastery (%)'},
                color=subject_mastery.values,
                color_continuous_scale='RdYlGn')
    st.plotly_chart(fig, use_container_width=True)
    
    # Concept difficulty analysis
    st.subheader("Concept Difficulty Analysis")
    
    # Calculate clearance rate
    data['concepts']['ClearanceRate'] = (data['concepts']['StudentsCleared'] / data['concepts']['StudentsAttempted'] * 100)
    
    difficult_concepts = data['concepts'].nsmallest(10, 'ClearanceRate')[['ConceptName', 'Subject', 'Topic', 'ClearanceRate', 'AvgTimeToMaster']]
    difficult_concepts.columns = ['Concept', 'Subject', 'Topic', 'Clearance Rate (%)', 'Avg Time to Master (min)']
    
    st.dataframe(
        difficult_concepts.style.background_gradient(subset=['Clearance Rate (%)'], cmap='RdYlGn_r'),
        use_container_width=True
    )
    
    # Time to mastery distribution
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.box(data['concepts'], y='AvgTimeToMaster', x='Subject',
                    title="Time to Master Concepts by Subject",
                    labels={'AvgTimeToMaster': 'Time to Master (minutes)'})
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Concept progress over time (simulated)
        dates = pd.date_range(start='2024-01-01', end='2024-01-20', freq='D')
        mastery_progress = []
        for date in dates:
            mastery_progress.append({
                'Date': date,
                'Mastery': 45 + (date.day * 1.2) + random.gauss(0, 2)
            })
        
        progress_df = pd.DataFrame(mastery_progress)
        
        fig = px.line(progress_df, x='Date', y='Mastery',
                     title="Overall Concept Mastery Progress",
                     labels={'Mastery': 'Mastery Rate (%)'})
        fig.add_hline(y=70, line_dash="dash", line_color="green", annotation_text="Target")
        st.plotly_chart(fig, use_container_width=True)

with tab5:
    st.subheader("AI-Powered Insights & Predictions")
    
    # Predictive analytics
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🔮 Performance Predictions")
        
        # Generate prediction data
        future_dates = pd.date_range(start='2024-01-21', end='2024-02-20', freq='D')
        predictions = []
        for date in future_dates:
            predictions.append({
                'Date': date,
                'Predicted': 68 + (date.day * 0.15) + random.gauss(0, 1),
                'Upper': 72 + (date.day * 0.15) + random.gauss(0, 0.5),
                'Lower': 64 + (date.day * 0.15) + random.gauss(0, 0.5)
            })
        
        pred_df = pd.DataFrame(predictions)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=pred_df['Date'], y=pred_df['Predicted'],
                                mode='lines', name='Predicted Performance',
                                line=dict(color='blue', width=3)))
        fig.add_trace(go.Scatter(x=pred_df['Date'], y=pred_df['Upper'],
                                fill=None, mode='lines', line_color='rgba(0,100,80,0)',
                                showlegend=False))
        fig.add_trace(go.Scatter(x=pred_df['Date'], y=pred_df['Lower'],
                                fill='tonexty', mode='lines', line_color='rgba(0,100,80,0)',
                                name='Confidence Interval'))
        
        fig.update_layout(title="30-Day Performance Forecast",
                         xaxis_title="Date",
                         yaxis_title="Predicted Score (%)")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 🎯 Intervention Recommendations")
        
        recommendations = [
            {"Priority": "High", "Action": "Schedule remedial classes for Grade 8-B Mathematics", "Impact": "+8.5%"},
            {"Priority": "High", "Action": "Deploy AI tutoring for at-risk students in Science", "Impact": "+6.2%"},
            {"Priority": "Medium", "Action": "Increase teacher training on AI tools", "Impact": "+4.1%"},
            {"Priority": "Medium", "Action": "Implement peer learning groups for Grade 7", "Impact": "+3.7%"},
            {"Priority": "Low", "Action": "Update curriculum for advanced learners", "Impact": "+2.3%"}
        ]
        
        rec_df = pd.DataFrame(recommendations)
        
        # Color code by priority
        def color_priority(val):
            if val == "High":
                return 'background-color: #ffebee'
            elif val == "Medium":
                return 'background-color: #fff3e0'
            else:
                return 'background-color: #e8f5e9'
        
        st.dataframe(
            rec_df.style.applymap(color_priority, subset=['Priority']),
            use_container_width=True
        )
    
    # Resource optimization
    st.markdown("### 📚 Resource Allocation Optimization")
    
    # Create a Sankey diagram for resource flow
    fig = go.Figure(data=[go.Sankey(
        node = dict(
            pad = 15,
            thickness = 20,
            line = dict(color = "black", width = 0.5),
            label = ["Total Resources", "Human Resources", "Technology", "Learning Materials", 
                    "Teachers", "Support Staff", "AI Platform", "Digital Tools", 
                    "Textbooks", "Online Content"],
            color = ["blue", "green", "green", "green", 
                    "lightgreen", "lightgreen", "lightgreen", "lightgreen", 
                    "lightgreen", "lightgreen"]
        ),
        link = dict(
            source = [0, 0, 0, 1, 1, 2, 2, 3, 3],
            target = [1, 2, 3, 4, 5, 6, 7, 8, 9],
            value = [40, 35, 25, 30, 10, 20, 15, 15, 10]
        )
    )])
    
    fig.update_layout(title_text="Resource Allocation Flow", font_size=10)
    st.plotly_chart(fig, use_container_width=True)

with tab6:
    st.subheader("🔍 Natural Language Query Interface")
    st.markdown("Ask questions about your school's performance in natural language")
    
    # Query input
    query = st.text_input("Enter your question:", placeholder="e.g., Show me struggling students in Mathematics Grade 8")
    
    if st.button("🔍 Search", type="primary"):
        if query:
            with st.spinner("Analyzing your query..."):
                time.sleep(1)  # Simulate processing
                
                # Mock query responses based on keywords
                if "struggling" in query.lower() and "mathematics" in query.lower():
                    st.success("Query understood! Showing struggling students in Mathematics...")
                    
                    # Filter students with low math scores
                    struggling_math = data['students'][
                        (data['students']['OverallScore'] < 50) & 
                        (data['students']['Grade'].str.contains('8'))
                    ].head(10)
                    
                    if not struggling_math.empty:
                        st.dataframe(
                            struggling_math[['Name', 'Grade', 'Section', 'OverallScore', 'WeakConcepts']],
                            use_container_width=True
                        )
                        
                        # Visualization
                        fig = px.scatter(struggling_math, x='WeakConcepts', y='OverallScore',
                                       size='AIInteractions', color='Section',
                                       title="Struggling Students: Weak Concepts vs Performance")
                        st.plotly_chart(fig, use_container_width=True)
                
                elif "top performing" in query.lower():
                    st.success("Query understood! Showing top performing students...")
                    
                    top_students = data['students'].nlargest(10, 'OverallScore')
                    st.dataframe(
                        top_students[['Name', 'Grade', 'Section', 'OverallScore', 'ConceptsClearedRate']],
                        use_container_width=True
                    )
                
                elif "teacher" in query.lower() and "effective" in query.lower():
                    st.success("Query understood! Analyzing teacher effectiveness...")
                    
                    # Show teacher effectiveness metrics
                    effective_teachers = data['teachers'].nlargest(5, 'AvgStudentImprovement')
                    st.dataframe(effective_teachers[['Name', 'Subject', 'AvgStudentImprovement', 'AIToolsUsage']])
                
                else:
                    st.info("I can help you with queries about:")
                    st.markdown("""
                    - Struggling or top-performing students
                    - Teacher effectiveness
                    - Subject-wise performance
                    - Grade-level analytics
                    - AI usage statistics
                    """)
    
    # Quick query suggestions
    st.markdown("### 💡 Try these queries:")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Show at-risk students in Grade 9"):
            st.session_state.query = "Show at-risk students in Grade 9"
            st.rerun()
    
    with col2:
        if st.button("Which teachers use AI tools most?"):
            st.session_state.query = "Which teachers use AI tools most effectively?"
            st.rerun()
    
    with col3:
        if st.button("Grade 10 Science performance"):
            st.session_state.query = "What is the performance of Grade 10 in Science?"
            st.rerun()

# Footer with real-time updates
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f"**Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

with col2:
    st.markdown("**Data Status:** 🟢 All Systems Operational")

with col3:
    if st.button("🔄 Refresh Dashboard"):
        st.rerun()

# Add auto-refresh info
st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ Dashboard Settings")
st.sidebar.markdown("**Auto-refresh:** Every 5 minutes")
st.sidebar.markdown("**Data Range:** Last 30 days")
st.sidebar.markdown("**School:** EeeBee Demo School")

# Performance indicator in sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("### 📈 Quick Stats")
st.sidebar.metric("School Rank", "#3", "↑2")
st.sidebar.metric("District Average", "68.5%", "↑1.2%")
st.sidebar.metric("State Average", "62.3%", "↑0.8%")
