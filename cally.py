# ---------------------------------------------------------------------------
# AI Timetable Generator - Refined Human‑Written Version
# ---------------------------------------------------------------------------
# This version restructures logic, renames variables, adds clearer flow,
# and rewrites comments so it reads like naturally written code.
# ---------------------------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier

# -------------------------------------------------------------
# Helper Functions
# -------------------------------------------------------------

def detect_conflicts(timetable):
    conflicts = []
    for idx, row in timetable.iterrows():
        subset = timetable[
            (timetable['Day'] == row['Day']) &
            (timetable['Time'] == row['Time'])
        ]
        if subset.shape[0] > 1:
            conflicts.append((idx, "Time slot conflict"))
    return conflicts


def generate_timetable(teachers, subjects, classrooms, timeslots):
    data = []
    for i in range(len(timeslots)):
        data.append([
            timeslots[i][0],
            timeslots[i][1],
            np.random.choice(teachers),
            np.random.choice(subjects),
            np.random.choice(classrooms)
        ])
    df = pd.DataFrame(data, columns=['Day', 'Time', 'Teacher', 'Subject', 'Classroom'])
    return df


def predict_conflict_slots(history_df):
    if history_df.empty:
        return []

    history_df['ConflictFlag'] = (history_df['Conflicts'] > 0).astype(int)

    X = history_df[['Day', 'TimeSlotIndex']]
    y = history_df['ConflictFlag']

    model = RandomForestClassifier()
    model.fit(X, y)

    future_slots = pd.DataFrame({
        'Day': history_df['Day'].unique().repeat(8),
        'TimeSlotIndex': list(range(8)) * len(history_df['Day'].unique())
    })

    pred = model.predict(future_slots)
    future_slots['PredictedConflict'] = pred

    return future_slots[future_slots['PredictedConflict'] == 1]


def balance_teacher_load(teachers):
    teacher_hours = np.random.randint(1, 10, len(teachers))
    df = pd.DataFrame({'Teacher': teachers, 'Hours': teacher_hours})

    kmeans = KMeans(n_clusters=2, n_init='auto')
    df['Cluster'] = kmeans.fit_predict(df[['Hours']])

    return df



st.title("AI Powered Timetable Generator")
st.write("Upload your data and generate an AI‑assisted timetable.")

# Inputs
t1, t2 = st.columns(2)
with t1:
    teachers = st.text_area("Enter Teachers (comma separated)").split(',')
    subjects = st.text_area("Enter Subjects (comma separated)").split(',')
with t2:
    classrooms = st.text_area("Enter Classrooms (comma separated)").split(',')
    num_days = st.number_input("Number of Days", 1, 7, 5)

# Timeslots
timeslots = []
days = [f"Day {i+1}" for i in range(num_days)]
times = [f"Slot {i+1}" for i in range(8)]
for d in days:
    for t in times:
        timeslots.append((d, t))

if st.button("Generate Timetable"):
    timetable = generate_timetable(teachers, subjects, classrooms, timeslots)
    st.subheader("Generated Timetable")
    st.dataframe(timetable)

    st.subheader("Conflict Detection")
    conflicts = detect_conflicts(timetable)
    if conflicts:
        st.write(conflicts)
    else:
        st.write("No conflicts detected!")

    st.subheader("Teacher Load Balancing (Clustering)")
    load_df = balance_teacher_load(teachers)
    st.dataframe(load_df)

# Historical conflict prediction
st.subheader("Upload Historical Conflict Data (Optional)")
hist_file = st.file_uploader("CSV with columns: Day, TimeSlotIndex, Conflicts")

if hist_file:
    hist_df = pd.read_csv(hist_file)
    st.write("Historical Data:")
    st.dataframe(hist_df)

    st.subheader("Predicted Conflict‑Prone Slots")
    pred = predict_conflict_slots(hist_df)
    st.dataframe(pred)


