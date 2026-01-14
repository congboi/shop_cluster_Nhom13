import papermill as pm
import os

os.makedirs("notebooks/runs", exist_ok=True)

# run_preprocessing_and_eda.py
pm.execute_notebook(
    "notebooks/preprocessing_and_eda.ipynb",
    "notebooks/runs/preprocessing_and_eda_run.ipynb",
    parameters=dict(
        DATA_PATH="data/raw/online_retail.csv",
        COUNTRY="United Kingdom",
        OUTPUT_DIR="data/processed",
        PLOT_REVENUE=True,         # tắt bớt plot khi chạy batch
        PLOT_TIME_PATTERNS=True,
        PLOT_PRODUCTS=True,
        PLOT_CUSTOMERS=True,
        PLOT_RFM=True,
    ),
    kernel_name="python3",
)

# run_basket_preparation.py

pm.execute_notebook(
    "notebooks/basket_preparation.ipynb",
    "notebooks/runs/basket_preparation_run.ipynb",
    parameters=dict(
        CLEANED_DATA_PATH="data/processed/cleaned_uk_data.csv",
        BASKET_BOOL_PATH="data/processed/basket_bool.parquet",
        INVOICE_COL="InvoiceNo",
        ITEM_COL="Description",
        QUANTITY_COL="Quantity",
        THRESHOLD=1,
    ),
    kernel_name="python3",
)

# Chạy Notebook Apriori Modelling
pm.execute_notebook(
    "notebooks/apriori_modelling.ipynb",
    "notebooks/runs/apriori_modelling_run.ipynb",
    parameters=dict(
        BASKET_BOOL_PATH="data/processed/basket_bool.parquet",
        RULES_OUTPUT_PATH="data/processed/rules_apriori_filtered.csv",

        # Tham số Apriori
        MIN_SUPPORT=0.03,
        MAX_LEN=4,

        # Generate rules
        METRIC="lift",
        MIN_THRESHOLD=1.0,

        # Lọc luật
        FILTER_MIN_SUPPORT=0.02,
        FILTER_MIN_CONF=0.4,
        FILTER_MIN_LIFT=1.2,
        FILTER_MAX_ANTECEDENTS=2,
        FILTER_MAX_CONSEQUENTS=1,

        # Số luật để vẽ
        TOP_N_RULES=20,

        # Tắt plot khi chạy batch (bật = True nếu muốn xem hình)
        PLOT_TOP_LIFT=True,
        PLOT_TOP_CONF=True,
        PLOT_SCATTER=True,
        PLOT_NETWORK=True,
        PLOT_PLOTLY_NETWORK=True,
        PLOT_PLOTLY_SCATTER=True,  
    ),
    kernel_name="python3",
)

# Chạy Notebook FP_Growth Modelling
pm.execute_notebook(
    "notebooks/fp_growth_modelling.ipynb",
    "notebooks/runs/fp_growth_modelling_run.ipynb",
    parameters=dict(
        BASKET_BOOL_PATH="data/processed/basket_bool.parquet",
        RULES_OUTPUT_PATH="data/processed/rules_fpgrowth_filtered.csv",

        MIN_SUPPORT=0.03,
        MAX_LEN=4,

        METRIC="lift",
        MIN_THRESHOLD=1.0,

        FILTER_MIN_SUPPORT=0.02,
        FILTER_MIN_CONF=0.4,
        FILTER_MIN_LIFT=1.2,
        FILTER_MAX_ANTECEDENTS=2,
        FILTER_MAX_CONSEQUENTS=1,

        TOP_N_RULES=20,

        PLOT_TOP_LIFT=True,
        PLOT_TOP_CONF=True,
        PLOT_SCATTER=True,
        PLOT_NETWORK=True,
        PLOT_PLOTLY_SCATTER=True,
    ),
    kernel_name="python3",
)

# Chạy Notebook So sánh Apriori và FP-Growth
pm.execute_notebook(
    "notebooks/compare_apriori_fpgrowth.ipynb",
    "notebooks/runs/compare_apriori_fpgrowth_run.ipynb",
    parameters=dict(
        BASKET_BOOL_PATH="data/processed/basket_bool.parquet",

        MIN_SUPPORT=0.03,
        MAX_LEN=4,

        METRIC="lift",
        MIN_THRESHOLD=1.0,
    ),
    kernel_name="python3",
)


# run_clustering_from_rules.py
pm.execute_notebook(
    "notebooks/clustering_from_rules.ipynb",
    "notebooks/runs/clustering_from_rules_run.ipynb",
    parameters=dict(
        CLEANED_DATA_PATH="data/processed/cleaned_uk_data.csv",
        RULES_INPUT_PATH="data/processed/rules_apriori_filtered.csv",

        TOP_K_RULES=30,
        SORT_RULES_BY="lift",
        WEIGHTING="lift",
        MIN_ANTECEDENT_LEN=1,
        USE_RFM=True,
        RFM_SCALE=True,
        RULE_SCALE=False,

        K_MIN=2,
        K_MAX=10,
        N_CLUSTERS=None,
        RANDOM_STATE=42,

        OUTPUT_CLUSTER_PATH="data/processed/customer_clusters_from_rules.csv",

        PROJECTION_METHOD="pca",
        PLOT_2D=True,
    ),
    kernel_name="python3",
)

# Trong run_papermill.py, sửa phần notebook 7:

pm.execute_notebook(
    "notebooks/cluster_profiling_and_interpretation.ipynb",
    "notebooks/runs/cluster_profiling_and_interpretation_run.ipynb",
    parameters=dict(
        CLUSTER_RESULT_PATH="data/processed/customer_clusters_from_rules.csv",
        RULES_INPUT_PATH="data/processed/rules_fpgrowth_filtered.csv",
        CLEANED_DATA_PATH="data/processed/cleaned_uk_data.csv",
        
        # Simple parameters only
        TOP_RULES_PER_CLUSTER=10,
        TOP_PRODUCTS_PER_CLUSTER=15,
        MIN_RULE_CONFIDENCE=0.3,
        
        PROFILING_OUTPUT_PATH="data/processed/cluster_profiles_detailed.csv",
        MARKETING_RECOMMENDATIONS_PATH="data/processed/marketing_recommendations.csv",
        
        # Set all plots to False for batch run
        PLOT_RFM_COMPARISON=False,
        PLOT_PRODUCT_HEATMAP=False,
        PLOT_CLUSTER_RADAR=False,
    ),
    kernel_name="python3",
)
print("Đã chạy xong pipeline")
