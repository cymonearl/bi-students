import faicons as fa
import polars as pl
import pandas as pd
from plotnine import ggplot, aes, geom_bar, labs
import matplotlib.pyplot as plt
import joblib

from shiny import App, ui, render, reactive

df = pl.read_csv("StudentPerformanceFactors.csv")
df = df.filter(pl.col("Exam_Score") <= 100)

MODEL_FILENAME = 'performance_pipeline.joblib'

ml_pipeline = joblib.load(MODEL_FILENAME)

score_amount = (
    df.select("Exam_Score").min().item(),
    df.select("Exam_Score").max().item(),
)

hours_studied_amount = (
    df.select("Hours_Studied").min().item(),
    df.select("Hours_Studied").max().item(),
)

attendance_amount = (
    df.select("Attendance").min().item(),
    df.select("Attendance").max().item(),
)

sleep_hours = (
    df.select("Sleep_Hours").min().item(),
    df.select("Sleep_Hours").max().item(),
)

tutoring_sessions = (
    df.select("Tutoring_Sessions").min().item(),
    df.select("Tutoring_Sessions").max().item(),
)

yes_no = [
    "Yes", "No"
]

# Icons
ICONS = {
    "student" : fa.icon_svg("graduation-cap"),
    "chart" : fa.icon_svg("chart-simple"),
    "award" : fa.icon_svg("award"),
    "bed" : fa.icon_svg("bed"),
    "clock" : fa.icon_svg("clock"),
    "calendar" : fa.icon_svg("calendar"),
    "triangle" : fa.icon_svg("triangle-exclamation"),
    "brain" : fa.icon_svg("brain"),
}

# App
app_ui = ui.page_navbar(
    ui.nav_panel("🏠 Home", 
        ui.page_sidebar(
            ui.sidebar(
                ui.h2("Filter Data"),
                ui.input_slider("input_hours",
                    "Hours Studied Range",
                    min=hours_studied_amount[0],
                    max=hours_studied_amount[1],
                    value=hours_studied_amount,
                    ),
                ui.input_slider("input_score",
                    "Exam Score Range",
                    min=score_amount[0],
                    max=score_amount[1],
                    value=score_amount,
                    ),
                ui.input_slider("input_attendance",
                    "Attendance Range",
                    min=attendance_amount[0],
                    max=attendance_amount[1],
                    value=attendance_amount,
                    ),
                ui.input_slider("input_sleep_hours",
                    "Sleep Hours Range",
                    min=sleep_hours[0],
                    max=sleep_hours[1],
                    value=sleep_hours,
                    ),
                ui.input_slider("input_tutoring_sessions",
                    "Tutoring Sessions Range",
                    min=tutoring_sessions[0],
                    max=tutoring_sessions[1],
                    value=tutoring_sessions,
                ),
                ui.hr(),
                ui.input_select(
                    "input_gender",
                    "Gender",
                    choices=["All"] + df["Gender"].unique().to_list(),
                    selected="All"
                ),
                ui.input_select(
                    "input_school_type",
                    "School Type",
                    choices=["All"] + df["School_Type"].unique().to_list(),
                    selected="All"
                ),
                ui.input_select(
                    "input_ML",
                    "Motivation Level",
                    choices=["All"] + ["Low", "Medium", "High"],
                    selected="All"
                ),
                ui.input_select(
                    "input_IA",
                    "Internet Access",
                    choices=["All"] + yes_no,
                    selected="All"
                ),
                ui.input_select(
                    "input_LD",
                    "Learning Disabilities ",
                    choices=["All"] + yes_no,
                    selected="All"
                ),
                ui.input_action_button(
                    "reset_butt",
                    "Reset All Filter"
                )
            ),
            ui.div(
                ui.layout_column_wrap(
                    ui.value_box(
                        ui.h4("Student Count"),
                        ui.output_text("student_count"),
                        showcase=ICONS["student"],
                    ),
                    ui.value_box(
                        ui.h4("Average Exam Score"),
                        ui.output_text("exam_score_mean"),
                        showcase=ICONS["chart"],
                    ),
                    ui.value_box(
                        ui.h4("Highest Exam Score"),
                        ui.output_text("exam_score_max"),
                        showcase=ICONS["award"],
                    ),
                    ui.value_box(
                        ui.h4("Lowest Exam Score"),
                        ui.output_text("exam_score_min"),
                        showcase=ICONS["triangle"],
                    ),
                    fill=True,
                ),
                ui.layout_columns(
                    ui.card(
                        ui.card_header("Top 20 Students"),
                        ui.output_data_frame("top_twenty"),
                    ),
                    ui.card(
                        ui.card_header("Exam Score Distribution"),
                        ui.output_plot("exam_score_dist"),
                    ),
                    fillable=True,
                ),
            ),
            ui.hr(),
            ui.div(
                ui.layout_column_wrap(
                    ui.value_box(
                        ui.h4("Average Sleep Hours"),
                        ui.output_text("sleep_hours_mean"),
                        showcase=ICONS["bed"],
                    ),
                    ui.value_box(
                        ui.h4("Average Studied Hours"),
                        ui.output_text("hours_studied_mean"),
                        showcase=ICONS["clock"],
                    ),
                    ui.value_box(
                        ui.h4("Average Attendance"),
                        ui.output_text("attendance_mean"),
                        showcase=ICONS["calendar"],
                    ),
                    fill=False,
                ),
                ui.layout_column_wrap(
                    ui.card(
                        ui.card_header("School Type Breakdown"),
                        ui.output_plot("school_type_dist"),
                    ),
                    ui.card(
                    ui.card_header("Studied Hours Distribution"),
                    ui.output_plot("studied_hours_dist"),
                    ),
                ),
                ui.layout_column_wrap(
                    ui.card(
                        ui.card_header("Gender Breakdown"),
                        ui.output_plot("gender_dist"),
                    ),
                    ui.card(
                        ui.card_header("Attendance Distribution"),
                        ui.output_plot("attendance_dist"),
                    ),
                ),
                ui.layout_column_wrap(
                    ui.card(
                        ui.card_header("Internet Access Breakdown"),
                        ui.output_plot("ia_dist")
                    ),
                    ui.card(
                        ui.card_header("Learning Disabilities Breakdown"),
                        ui.output_plot("ld_dist")
                    ),
                ),
                ui.layout_column_wrap(
                    ui.card(
                        ui.card_header("Motivation Level Breakdown"),
                        ui.output_plot("ml_dist")
                    )
                ),
            ),
        ),
    ),
    ui.nav_panel("🧑‍🎓 Score Predictor", 
        ui.page_sidebar(
            ui.sidebar(
                ui.h2("Data"),
                ui.input_slider("pred_hours",
                    "Hours Studied Range",
                    min=0,
                    max=hours_studied_amount[1],
                    value=8,
                    ),
                ui.input_slider("pred_attendance",
                    "Attendance Range",
                    min=0,
                    max=attendance_amount[1],
                    value=10,
                    ),
                ui.input_slider("pred_sleep_hours",
                    "Sleep Hours Range",
                    min=sleep_hours[0],
                    max=sleep_hours[1],
                    value=sleep_hours[0],
                    ),
                ui.input_slider("pred_tutoring_sessions",
                    "Tutoring Sessions Range",
                    min=0,
                    max=tutoring_sessions[1],
                    value=0,
                ),
                ui.hr(),
                ui.input_select(
                    "pred_gender",
                    "Gender",
                    df["Gender"].unique().to_list(),
                    selected="Male"
                ),
                ui.input_select(
                    "pred_school_type",
                    "School Type",
                    df["School_Type"].unique().to_list(),
                    selected="Public"
                ),
                ui.input_select(
                    "pred_ML",
                    "Motivation Level",
                    ["Low", "Medium", "High"],
                    selected="Low"
                ),
                ui.input_select(
                    "pred_IA",
                    "Internet Access",
                    yes_no,
                    selected="Yes"
                ),
                ui.input_select(
                    "pred_LD",
                    "Learning Disabilities ",
                    yes_no,
                    selected="Yes"
                ),
                ui.input_action_link(
                    "reset_butt2",
                    "Reset All Input",
                ),
            ),
            ui.value_box(
                ui.h4("Predicted Score"),
                ui.output_text("predicted_score"),
                showcase=ICONS["brain"],
            ),
        ),
    ),
    ui.nav_panel("❓ About",
        ui.div(
            ui.layout_column_wrap(
                ui.card(
                    ui.card_header("📚 About Dashboard"),
                    ui.card_body(
                        ui.h5("Purpose"),
                        ui.p(
                            "A Machine Problem for CSDS 312 that analyzes and predicts student exam scores "
                            "based on lifestyle and educational factors."
                        ),
                        ui.p(
                            "The School Director wanted to have a simple dashboard, solely purpose for Student Lifestyle, "
                            "without meddling into their personal information like Economic status and more."
                        ),
                        ui.br(),
                        ui.h5("Tech Stack"),
                        ui.p("Python Shiny • Scikit-learn • Polars • Matplotlib")
                    )
                ),
                ui.card(
                    ui.card_header("📊 Dataset & Model"),
                    ui.card_body(
                        ui.h5("Data Source"),
                        ui.p(
                            ui.tags.a(
                                "Kaggle Student Performance Dataset",
                                href="https://www.kaggle.com/datasets/minahilfatima12328/performance-trends-in-education",
                                target="_blank",
                                class_="text-primary"
                            ),
                            ui.br(),
                            "• 6,607 records (synthetic/simulated)", ui.br(),
                            "• 20 features → 1 target (Exam Score)"
                        ),
                        ui.br(),
                        ui.h5("Model Performance"),
                        ui.tags.ul(
                            ui.tags.li("Algorithm: Random Forest"),
                            ui.tags.li("Accuracy: R² = 0.67"),
                            ui.tags.li("Error: ± 1.11 points"),
                            ui.tags.li("Train/Test: 80/20 split"),
                        ),
                        ui.br(),
                        ui.h5("⚠️ Limitations"),
                        ui.tags.ul(
                            ui.tags.li("Synthetic data, not real students"),
                            ui.tags.li("No cultural/regional factors"),
                            ui.tags.li("Mental health not measured"),
                        )
                    )
                ),
            ),
        ),
        ui.div(
            ui.layout_column_wrap(
                ui.card(
                    ui.card_header("📖 How to Use"),
                    ui.card_body(
                        ui.h5("🏠 Home Tab"),
                        ui.tags.ul(
                            ui.tags.li("Filter data with sidebar controls"),
                            ui.tags.li("View statistics and distributions"),
                            ui.tags.li("Analyze top performer patterns"),
                        ),
                        ui.br(),
                        ui.h5("🧑‍🎓 Score Predictor"),
                        ui.tags.ul(
                            ui.tags.li("Input your study habits and context"),
                            ui.tags.li("Get predicted exam score"),
                            ui.tags.li("Experiment with 'what-if' scenarios"),
                        ),
                        ui.br(),
                        ui.h5("💡 Best Use Cases"),
                        ui.p(
                            "✅ Identifying improvement areas", ui.br(),
                            "✅ Understanding success patterns", ui.br(),
                            "❌ NOT for critical decisions alone", ui.br(),
                            "❌ NOT medical/professional advice"
                        )
                    )
                ),
                ui.card(
                    ui.card_header("🎓 Project Info"),
                    ui.card_body(
                        ui.h5("Developer"),
                        ui.p(
                            "Cymon Earl A. Galzote", ui.br(),
                            "CSDS 312 - Applied Data Science", ui.br(),
                            "November 2025", ui.br(),
                            "ceagalzote01323@usep.edu.ph"
                        ),
                    )
                ),
            ),
        ),
        ui.div(
            ui.card(
                ui.card_header("⚠️ Important Disclaimer"),
                ui.card_body(
                    ui.p(
                        "This dashboard is for educational exploration only. Predictions are statistical estimates "
                        "based on patterns in simulated data. Individual results vary."
                    )
                ),
                class_="bg-light"
            ),           
        ),
    ),
    title="Students Performance Dashboard",
    inverse=True,  # Dark navbar
    fillable=True,
)

def server(input, output, session):
    
    @reactive.calc
    def filtered_df():
        hours_studied = input.input_hours()
        exam_scores = input.input_score()
        attendance = input.input_attendance()
        sleep_hours = input.input_sleep_hours()
        tutoring_sessions = input.input_tutoring_sessions()
        gender = input.input_gender()
        school_type = input.input_school_type()
        motivation_level = input.input_ML()
        internet_access = input.input_IA()
        learning_disabilities = input.input_LD()

        conditions = [
                pl.col("Hours_Studied").is_between(hours_studied[0], hours_studied[1]),
                pl.col("Exam_Score").is_between(exam_scores[0], exam_scores[1]),
                pl.col("Attendance").is_between(attendance[0], attendance[1]),
                pl.col("Sleep_Hours").is_between(sleep_hours[0], sleep_hours[1]),
                pl.col("Tutoring_Sessions").is_between(tutoring_sessions[0], tutoring_sessions[1]),
        ]

        if input.input_gender() != "All":
            conditions.append(pl.col("Gender") == gender)
        
        if input.input_school_type() != "All":
            conditions.append(pl.col("School_Type") == school_type)

        if input.input_ML() != "All":
            conditions.append(pl.col("Motivation_Level") == motivation_level)

        if input.input_IA() != "All":
            conditions.append(pl.col("Internet_Access") == internet_access)

        if input.input_LD() != "All":
            conditions.append(pl.col("Learning_Disabilities") == learning_disabilities)

        return df.filter(*conditions)

    @render.text
    def predicted_score():
        if ml_pipeline is None or filtered_df().height == 0:
            return 0
        df = filtered_df()

        input_data = {
            "Hours_Studied": [input.pred_hours()],
            "Attendance": [input.pred_attendance()],
            "Sleep_Hours": [input.pred_sleep_hours()],
            "Tutoring_Sessions": [input.pred_tutoring_sessions()],
            "Previous_Scores": [df["Previous_Scores"].mean()],
            "Physical_Activity": [df["Physical_Activity"].mean()],
            
            # Categorical Inputs (strings)
            "Gender": [input.pred_gender()],
            "School_Type": [input.pred_school_type()],
            "Motivation_Level": [input.pred_ML()],
            "Internet_Access": [input.pred_IA()],
            "Learning_Disabilities": [input.pred_LD()],
            "Parental_Involvement": [df["Parental_Involvement"].mode().first()],
            "Family_Income": [df["Family_Income"].mode().first()],
            "Teacher_Quality": [df["Teacher_Quality"].mode().first()],
            "Access_to_Resources": [df["Access_to_Resources"].mode().first()],
            "Peer_Influence": [df["Peer_Influence"].mode().first()],
            "Extracurricular_Activities": [df["Extracurricular_Activities"].mode().first()],
            "Distance_from_Home": [df["Distance_from_Home"].mode().first()],
            "Parental_Education_Level": [df["Parental_Education_Level"].mode().first()],
        }

        score = ml_pipeline.predict(pd.DataFrame(input_data))
        return score

    @render.data_frame
    def table():
        return df.head(20)
    @render.data_frame
    def top_twenty():
        df = filtered_df()
        top15 = (
            df.sort("Exam_Score", descending=True)
            .head(20)
            .select("Exam_Score", 
                    "Hours_Studied",
                    "Attendance",
                    "Gender",
                    "School_Type",
                    "Motivation_Level",
                    "Internet_Access",
                    "Tutoring_Sessions",
                    "Learning_Disabilities",
                    "Sleep_Hours",
            )
            .with_row_index(name=" ", offset=1)
        )
        return top15

    @reactive.effect
    @reactive.event(input.reset_butt)
    def _():
        ui.update_slider("input_hours", value=hours_studied_amount)
        ui.update_slider("input_score", value=score_amount)
        ui.update_slider("input_attendance", value=attendance_amount)
        ui.update_slider("input_sleep_hours", value=sleep_hours)
        ui.update_slider("input_tutoring_sessions", value=tutoring_sessions)
        ui.update_select("input_gender", selected="All")
        ui.update_select("input_school_type", selected="All")
        ui.update_select("input_ML", selected="All")
        ui.update_select("input_IA", selected="All")
        ui.update_select("input_LD", selected="All")

    @reactive.effect
    @reactive.event(input.reset_butt2)
    def _():
        ui.update_slider("pred_hours", value=8)
        ui.update_slider("pred_attendance", value=10)
        ui.update_slider("pred_sleep_hours", value=4)
        ui.update_slider("pred_tutoring_sessions", value=0)
        ui.update_select("pred_gender", selected="Male")
        ui.update_select("pred_school_type", selected="Public")
        ui.update_select("pred_ML", selected="Low")
        ui.update_select("pred_IA", selected="Yes")
        ui.update_select("pred_LD", selected="Yes")

    @render.plot
    def exam_score_dist():
        df = filtered_df()
        plot = (
            ggplot(df, aes(x = "Exam_Score")) +
                geom_bar(fill="teal") +
                labs(
                    x = "Exam Scores",
                    y = "Student Count"
                )
        )
        return plot
    @render.plot
    def school_type_dist():
        df = filtered_df()
        summary_df = df.group_by("School_Type").len().rename({"len": "Count"})

        fig, plot = plt.subplots()
        plot.pie(
            x=summary_df["Count"],
            labels = summary_df["School_Type"],
            autopct="%1.1f%%"
        )
        return plot
    @render.plot
    def gender_dist():
        df = filtered_df()
        summary_df = df.group_by("Gender").len().rename({"len": "Count"})

        fig, plot = plt.subplots()
        plot.pie(
            x=summary_df["Count"],
            labels=summary_df["Gender"],
            autopct="%1.1f%%"
        )
        return plot
    @render.plot
    def ia_dist():
        df = filtered_df()
        summary_df = df.group_by("Internet_Access").len().rename({"len": "Count"})

        fig, plot = plt.subplots()
        plot.pie(
            x=summary_df["Count"],
            labels=summary_df["Internet_Access"],
            autopct="%1.1f%%"
        )
        return plot
    @render.plot
    def ml_dist():
        df = filtered_df()
        summary_df = df.group_by("Motivation_Level").len().rename({"len": "Count"})

        fig, plot = plt.subplots()
        plot.pie(
            x=summary_df["Count"],
            labels=summary_df["Motivation_Level"],
            autopct="%1.1f%%"
        )
        return plot
    @render.plot
    def ld_dist():
        df = filtered_df()
        summary_df = df.group_by("Learning_Disabilities").len().rename({"len": "Count"})

        fig, plot = plt.subplots()
        plot.pie(
            x=summary_df["Count"],
            labels=summary_df["Learning_Disabilities"],
            autopct="%1.1f%%",
        )
        return plot
    @render.plot
    def studied_hours_dist():
        df = filtered_df()
        plot = (
            ggplot(df, aes(x = "Hours_Studied")) +
                geom_bar(fill="teal") +
                labs(
                    x = "Exam Scores",
                    y = "Student Count"
                )
        )
        return plot
    @render.plot
    def attendance_dist():
        df = filtered_df()
        plot = (
                ggplot(df, aes(x = "Attendance")) +
                geom_bar(fill="teal") +
                labs(
                    x = "Attended",
                    y = "Student Count"
                )
        )
        return plot
    @render.plot
    def sleep_hours_dist():
        df = filtered_df()
        plot = (
            ggplot(df, aes(x = "Sleep_Hours")) +
                geom_bar(fill="teal") +
                labs(
                    x = "Hours",
                    y = "Student Count"
                )
        )
        return plot
        
    @render.text
    def student_count():
        df = filtered_df()
        return df["Attendance"].count()
    @render.text
    def exam_score_mean():
        df = filtered_df()
        return round(df["Exam_Score"].mean())
    @render.text
    def exam_score_max():
        df = filtered_df()
        return df["Exam_Score"].max()
    @render.text
    def sleep_hours_mean():
        df = filtered_df()
        return round(df["Sleep_Hours"].mean()) 
    @render.text
    def hours_studied_mean():
        df = filtered_df()
        return round(df["Hours_Studied"].mean())
    @render.text
    def attendance_mean():
        df = filtered_df()
        return round(df["Attendance"].mean())
    @render.text
    def exam_score_min():
        df = filtered_df()
        return df["Exam_Score"].min()

app = App(app_ui, server)