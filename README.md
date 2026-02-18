# 运行方法
## 1. Install dependencies
pip install -r requirements.txt

## 2. Build database (already done — skip if reviews.db exists)
python src/data_processing.py

## 3. Run notebooks in order
jupyter notebook notebooks/

## 4. Launch dashboard
streamlit run app/streamlit_app.py

# Assignment Overview

## Business Context
You are a group of Data Science Consultants hired by **HospitalityTech Solutions**, a SaaS company providing analytics platforms to hotels. They need you to build an intelligent analytics platform that helps hotel managers:

*   **Understand** customer satisfaction drivers
*   **Identify** improvement opportunities
*   **Benchmark** against competitors
*   **Predict** future trends
*   **Optimize** resource allocation

Your task is to develop a working product that hotel managers can use to make data-driven decisions.

---

## Assignment 1: Data Foundation & Exploratory Analytics (15%)
**Duration:** Weeks 2-6  
**Submission Deadline:** See Canvas  
**Weight:** 15% of final course grade

### Objectives
Build a foundational analytics system with:
1.  **Efficient data storage and retrieval:** SQLite database with 50,000–80,000 reviews.
2.  **Comprehensive exploratory analysis:** Statistical rigor in findings.
3.  **Performance optimization:** Through query and code profiling.
4.  **Competitive benchmarking strategy:** For hotel comparison.
5.  **User-friendly dashboard:** For non-technical users.

### Deliverables

#### 1. GitHub Repository Structure
```text
student-name-hotel-analytics/
├── README.md                   # Setup and usage instructions
├── requirements.txt            # Python dependencies
├── .gitignore                  
├── data/
│   ├── reviews_sample.db       # SQLite with 5000+ sample reviews
│   └── data_schema.sql         # (Optional) Schema documentation
├── notebooks/
│   ├── 01_data_preparation.ipynb
│   ├── 02_exploratory_analysis.ipynb
│   ├── 03_competitive_benchmarking.ipynb
│   └── 04_performance_profiling.ipynb
├── src/
│   ├── data_processing.py
│   ├── benchmarking.py
│   └── utils.py
├── app/
│   └── streamlit_app.py        # Dashboard application
├── profiling/
│   ├── query_results.txt       # Query profiling outputs
│   └── code_profiling.txt      # Code profiling results
└── reports/
    └── assignment1_report.pdf  # 8-10 pages (excluding cover, references, appendices)
```

#### 2. Data Requirements
*   **Timeframe:** Use latest 5 years available.
*   **Volume:** At least 50,000–80,000+ reviews (after filtering).
*   **Storage:** SQLite database with appropriate schema design.
*   **Sample Data:** Include 5,000+ reviews in repository (for TAs to test).

#### 3. Technical Report (8-10 pages; Submit on Canvas)
*   **Format:** PDF, excluding cover page, references, and appendices.

**Required Sections:**
1.  **Executive Summary:** Business problem and solution overview, key findings.
2.  **Data Foundation:** Data filtering rationale, schema design (ER diagram optional), indexing strategy, and data statistics.
3.  **Exploratory Data Analysis:** Key insights with business implications.
4.  **Performance Profiling & Optimization:** Query profiling and code profiling results.
5.  **Competitive Benchmarking Strategy:**
    *   *Business Context:* Hotel managers struggle to identify meaningful improvement opportunities ("Who are my real competitors?").
    *   Methodology for identifying comparable hotel groups.
    *   Performance analysis across groups and identification of best practices.
    *   Specific, actionable recommendations.
    *   Validation of approach.
    *   *Note: This is an open-ended problem. Propose and implement YOUR solution.*
6.  **System Architecture & Dashboard:** User interface rationale and key features.
7.  **Conclusion:** Key observations, deliverables, limitations, and future enhancements (2-3 sentences).
*   *For Group Submissions:* Include a member contribution summary table at the beginning.

#### 4. Working Dashboard (Streamlit)
*   Functional dashboard with web interface.
*   3-5 core features that solve business problems.
*   User documentation in `README.md` (GitHub).