import faicons as fa
import polars as pl
import pandas as pd
from plotnine import *
import joblib

# Shiny Imports
from shiny import App, ui, render, reactive

# =============================================================================
# 1. SETUP & DATA LOADING
# =============================================================================

# Load Data
try:
    df = pl.read_csv("StudentPerformanceFactors.csv")
    df = df.filter(pl.col("Exam_Score") <= 100)
    print("Data loaded successfully.")
except Exception as e:
    print(f"Error loading data: {e}")
    # Create a dummy DF to prevent immediate crash if file missing
    df = pl.DataFrame({"Exam_Score": [], "Hours_Studied": [], "Attendance": []})

# Load Model
MODEL_FILENAME = 'performance_pipeline.joblib'
try:
    ml_pipeline = joblib.load(MODEL_FILENAME)
    model_loaded = True
except:
    ml_pipeline = None
    model_loaded = False
    print("Warning: Model file not found. Prediction tab will use dummy logic.")

# Constants & Helper Values
hours_range = (df.select("Hours_Studied").min().item(), df.select("Hours_Studied").max().item())
score_range = (df.select("Exam_Score").min().item(), df.select("Exam_Score").max().item())
attendance_range = (df.select("Attendance").min().item(), df.select("Attendance").max().item())
sleep_range = (df.select("Sleep_Hours").min().item(), df.select("Sleep_Hours").max().item())
tutoring_range = (df.select("Tutoring_Sessions").min().item(), df.select("Tutoring_Sessions").max().item())

yes_no = ["Yes", "No"]

ICONS = {
    "student": fa.icon_svg("graduation-cap"),
    "chart": fa.icon_svg("chart-simple"),
    "award": fa.icon_svg("award"),
    "bed": fa.icon_svg("bed"),
    "clock": fa.icon_svg("clock"),
    "calendar": fa.icon_svg("calendar"),
    "brain": fa.icon_svg("brain"),
    "warning": fa.icon_svg("triangle-exclamation"),
}

# =============================================================================
# 2. UI LAYOUT
# =============================================================================

app_ui = ui.page_navbar(
    
    # --- TAB 1: DASHBOARD (Analytics) ---
    ui.nav_panel("🏠 Dashboard", 
        ui.page_sidebar(
            ui.sidebar(
                ui.h2("Filter Data"),
                ui.input_slider("input_hours", "Hours Studied", min=hours_range[0], max=hours_range[1], value=hours_range),
                ui.input_slider("input_attendance", "Attendance (%)", min=attendance_range[0], max=attendance_range[1], value=attendance_range),
                ui.input_select("input_gender", "Gender", choices=["All"] + df["Gender"].unique().to_list(), selected="All"),
                ui.input_select("input_school_type", "School Type", choices=["All"] + df["School_Type"].unique().to_list(), selected="All"),
                ui.hr(),
                ui.input_action_button("reset_butt", "Reset Filters", class_="btn-secondary")
            ),
            
            # The "Safety Net" Output Wrapper
            ui.output_ui("main_dashboard_content")
        )
    ),

    # --- TAB 2: SCORE PREDICTOR (Your Original Logic) ---
    ui.nav_panel("🧑‍🎓 Score Predictor", 
        ui.page_sidebar(
            ui.sidebar(
                ui.h3("Student Profile"),
                ui.input_slider("pred_hours", "Hours Studied", 0, hours_range[1], 8),
                ui.input_slider("pred_attendance", "Attendance", 0, 100, 80),
                ui.input_slider("pred_sleep_hours", "Sleep Hours", 4, 10, 7),
                ui.input_slider("pred_tutoring_sessions", "Tutoring Sessions", 0, 5, 0),
                ui.hr(),
                ui.input_select("pred_gender", "Gender", df["Gender"].unique().to_list(), selected="Male"),
                ui.input_select("pred_school_type", "School Type", df["School_Type"].unique().to_list(), selected="Public"),
                ui.input_select("pred_ML", "Motivation Level", ["Low", "Medium", "High"], selected="Low"),
                ui.input_select("pred_IA", "Internet Access", yes_no, selected="Yes"),
                ui.input_select("pred_LD", "Learning Disabilities", yes_no, selected="No"),
                ui.input_action_button("reset_butt2", "Reset Inputs", class_="btn-secondary"),
            ),
            ui.div(
                ui.card(
                    ui.card_header("Predicted Exam Score"),
                    ui.div(
                        ui.output_text("predicted_score_ui"), # Large Text Output
                        style="text-align: center; font-size: 3rem; font-weight: bold; color: #4e79a7;"
                    ),
                    ui.card_footer("Based on the Machine Learning Pipeline")
                ),
                ui.card(
                    ui.card_header("💡 Improvement Tips"),
                    ui.markdown("""
                    - **Attendance Matters:** Students with >90% attendance score 15% higher on average.
                    - **Sleep:** 7-8 hours is the sweet spot for maximum retention.
                    - **Resources:** Access to internet and tutoring has a moderate positive correlation.
                    """)
                )
            )
        )
    ),

    # --- TAB 3: ABOUT (Your Original Content) ---
    ui.nav_panel("❓ About",
        ui.div(
            ui.layout_column_wrap(
                ui.card(
                    ui.card_header("📚 About Dashboard"),
                    ui.card_body(
                        ui.h5("Purpose"),
                        ui.p("A Business Intelligence tool to analyze student performance trends and predict outcomes based on lifestyle factors."),
                        ui.h5("Tech Stack"),
                        ui.p("Python Shiny • Polars • Plotnine • Scikit-learn")
                    )
                ),
                ui.card(
                    ui.card_header("📊 Model Info"),
                    ui.card_body(
                        ui.p("Data Source: Synthetic/Simulated Student Data (6,607 records)."),
                        ui.p("Model: Random Forest Regressor (R² ≈ 0.67)."),
                        ui.p("Limitations: Does not account for mental health or specific curriculum differences.")
                    )
                ),
            )
        )
    ),
    
    title="Student Performance Intelligence",
    inverse=True,
    fillable=True
)

# =============================================================================
# 3. SERVER LOGIC
# =============================================================================

def server(input, output, session):

    # --- REACTIVE FILTER ---
    @reactive.calc
    def filtered_df():
        hours = input.input_hours()
        attend = input.input_attendance()
        
        # Start with range conditions
        cond = [
            pl.col("Hours_Studied").is_between(hours[0], hours[1]),
            pl.col("Attendance").is_between(attend[0], attend[1])
        ]
        
        # Add categorical filters if not "All"
        if input.input_gender() != "All":
            cond.append(pl.col("Gender") == input.input_gender())
        if input.input_school_type() != "All":
            cond.append(pl.col("School_Type") == input.input_school_type())
            
        return df.filter(*cond)

    # --- SAFETY NET DASHBOARD ---
    @render.ui
    def main_dashboard_content():
        data = filtered_df()
        
        # 1. Safety Check: If no data, show error card
        if data.height == 0:
            return ui.div(
                ui.card(
                    ui.div(
                        ICONS["warning"],
                        ui.h3("No Data Found", style="color: #dc3545; margin-top: 10px;"),
                        ui.p("Your filters are too strict. No students matched these criteria."),
                        ui.p("Try widening the sliders or clicking 'Reset Filters'."),
                        style="text-align: center; padding: 40px;"
                    )
                ),
                style="margin-top: 20px;"
            )
        
        # 2. If data exists, render the full Dashboard
        return ui.div(
            # Row 1: KPI Cards
            ui.layout_column_wrap(
                ui.value_box("Student Count", f"{data.height}", showcase=ICONS["student"]),
                ui.value_box("Avg Score", f"{data['Exam_Score'].mean():.1f}", showcase=ICONS["chart"]),
                ui.value_box("Highest Score", f"{data['Exam_Score'].max()}", showcase=ICONS["award"]),
            ),
            
            # Row 2: Distributions & Factors
            ui.layout_columns(
                ui.card(
                    ui.card_header("Exam Score Distribution"),
                    ui.output_plot("plot_score_dist"),
                ),
                ui.card(
                    ui.card_header("Deep Dive: Factor Analysis"),
                    ui.input_select("factor_select", "Compare Scores By:", 
                                    choices=["Parental_Involvement", "Access_to_Resources", "Motivation_Level", "Family_Income", "School_Type"], 
                                    selected="Parental_Involvement"),
                    ui.output_plot("plot_factor_box"),
                ),
                col_widths=(5, 7)
            ),
            
            # Row 3: Correlations
            ui.layout_columns(
                 ui.card(
                    ui.card_header("Correlation Heatmap (What drives results?)"),
                    ui.output_plot("plot_heatmap"),
                )
            )
        )

    # --- PLOT RENDERERS ---
    @render.plot
    def plot_score_dist():
        return (
            ggplot(filtered_df(), aes(x="Exam_Score")) 
            + geom_histogram(fill="#4e79a7", bins=20, alpha=0.9) 
            + theme_minimal()
            + labs(x="Exam Score", y="Count")
        )

    @render.plot
    def plot_factor_box():
        factor = input.factor_select()
        # Ensure we work with Pandas for categorical plotting in plotnine if needed
        data_pd = filtered_df().to_pandas()
        return (
            ggplot(data_pd, aes(x=factor, y="Exam_Score", fill=factor))
            + geom_boxplot(show_legend=False)
            + theme_minimal()
            + theme(axis_text_x=element_text(rotation=45, ha="right"))
            + labs(y="Exam Score", title=f"Impact of {factor}")
        )

    @render.plot
    def plot_heatmap():
        # Calculate Correlation
        numeric_cols = ["Hours_Studied", "Attendance", "Sleep_Hours", "Previous_Scores", "Exam_Score"]
        corr = filtered_df().select(numeric_cols).to_pandas().corr().reset_index()
        corr_melted = corr.melt(id_vars='index', var_name='Variable', value_name='Correlation')
        
        return (
            ggplot(corr_melted, aes(x='index', y='Variable', fill='Correlation'))
            + geom_tile()
            + geom_text(aes(label='round(Correlation, 2)'), size=10, color="white")
            + scale_fill_cmap(cmap_name='coolwarm', limits=[-1, 1])
            + theme_minimal()
            + labs(x="", y="")
            + theme(axis_text_x=element_text(rotation=45, ha="right"))
        )

    # --- PREDICTION LOGIC (Your Robust Implementation) ---
    @render.text
    def predicted_score_ui():
        if filtered_df().height == 0: return "N/A"
        
        # Construct Input DF using User Inputs + Dataset Averages for missing fields
        input_data = {
            "Hours_Studied": [input.pred_hours()],
            "Attendance": [input.pred_attendance()],
            "Sleep_Hours": [input.pred_sleep_hours()],
            "Tutoring_Sessions": [input.pred_tutoring_sessions()],
            
            # Map Inputs to Feature Names
            "Gender": [input.pred_gender()],
            "School_Type": [input.pred_school_type()],
            "Motivation_Level": [input.pred_ML()],
            "Internet_Access": [input.pred_IA()],
            "Learning_Disabilities": [input.pred_LD()],
            
            # Defaults for un-inputted features (Critical for Model Stability)
            "Previous_Scores": [df["Previous_Scores"].mean()], 
            "Physical_Activity": [df["Physical_Activity"].mean()],
            "Parental_Involvement": [df["Parental_Involvement"].mode().first()],
            "Family_Income": [df["Family_Income"].mode().first()],
            "Teacher_Quality": [df["Teacher_Quality"].mode().first()],
            "Access_to_Resources": [df["Access_to_Resources"].mode().first()],
            "Peer_Influence": [df["Peer_Influence"].mode().first()],
            "Extracurricular_Activities": [df["Extracurricular_Activities"].mode().first()],
            "Distance_from_Home": [df["Distance_from_Home"].mode().first()],
            "Parental_Education_Level": [df["Parental_Education_Level"].mode().first()],
        }
        
        score = 0
        if model_loaded and ml_pipeline:
            try:
                score = ml_pipeline.predict(pd.DataFrame(input_data))[0]
            except Exception as e:
                return "Error"
        else:
            # Fallback Dummy Logic (if model file missing)
            base = 60
            score = base + (input.pred_hours() * 0.5) + (input.pred_attendance() * 0.2)
            
        return f"{score:.1f}"

    # --- RESET BUTTONS ---
    @reactive.effect
    @reactive.event(input.reset_butt)
    def _():
        ui.update_slider("input_hours", value=hours_range)
        ui.update_slider("input_attendance", value=attendance_range)
        ui.update_select("input_gender", selected="All")
        ui.update_select("input_school_type", selected="All")

    @reactive.effect
    @reactive.event(input.reset_butt2)
    def _():
        ui.update_slider("pred_hours", value=8)
        ui.update_slider("pred_attendance", value=80)
        ui.update_slider("pred_sleep_hours", value=7)
        ui.update_slider("pred_tutoring_sessions", value=0)
        ui.update_select("pred_gender", selected="Male")
        ui.update_select("pred_school_type", selected="Male")
        ui.update_select("pred_ML", selected="Low")
        ui.update_select("pred_IA", selected="Yes")
        ui.update_select("pred_LD", selected="No")

app = App(app_ui, server)