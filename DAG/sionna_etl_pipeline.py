from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from airflow.operators.empty import EmptyOperator
try:
    from airflow.providers.standard.operators.hitl import HITLOperator
    from airflow.providers.standard.operators.python import BranchPythonOperator
except ImportError:
    HITLOperator = None
    BranchPythonOperator = None
from datetime import datetime, timedelta
import os
import sys
import json
import shutil
import logging

# Add scripts to sys.path to allow importing modules
# We'll use the directory relative to this DAG file as a fallback
DAG_DIR = os.path.dirname(__file__)
PROJECT_ROOT_DEFAULT = os.path.dirname(DAG_DIR)
SCRIPTS_DIR = os.path.join(os.environ.get('PROJECT_ROOT', PROJECT_ROOT_DEFAULT), 'scripts')
if os.path.exists(SCRIPTS_DIR):
    sys.path.insert(0, SCRIPTS_DIR)

def _safe_run_id(run_id):
    """Windows-safe folder name (Airflow run_id can contain ':')."""
    return run_id.replace(":", "-")


def cleanup_artifacts(context):
    run_id = context.get('run_id')
    project_root = get_project_root()
    artifact_dir = os.path.join(project_root, 'artifacts', _safe_run_id(run_id))
    
    if os.path.exists(artifact_dir):
        logging.info(f"Cleaning up artifacts for failed run {run_id} at {artifact_dir}")
        try:
            shutil.rmtree(artifact_dir)
            logging.info(f"Successfully removed {artifact_dir}")
        except Exception as e:
            logging.error(f"Failed to remove {artifact_dir}: {e}")

# Default arguments for the DAG
default_args = {
    'owner': 'Xuewen SHAO & Xinyi LI',
    'depends_on_past': False,
    'start_date': datetime(2026, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'on_failure_callback': cleanup_artifacts,
}
DAG_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.dirname(DAG_DIR)

def get_project_root():
    return os.environ.get('PROJECT_ROOT', PROJECT_ROOT)


def run_simulations_callable(run_id, **kwargs):

    project_root = get_project_root()
    from sionna_simulator import main as sim_main
    
    plan_path = os.path.join(project_root, 'artifacts', _safe_run_id(run_id), 'run_plan.json')
    with open(plan_path, 'r', encoding='utf-8') as f:
        plan_data = json.load(f)
    
    plan_size = plan_data.get('plan_size', 0)
    print(f"Starting {plan_size} simulations for run_id: {run_id}")
    
    for i in range(plan_size):
        print(f"Running simulation {i+1}/{plan_size}...")
        sim_main(run_id, i)


def generate_plan_callable(**kwargs):
    """调用 run_plan_generator，从 dag_run.conf 读取 seed_id 或 reproduce_run_id。"""
    dag_run = kwargs.get('dag_run')
    run_id = dag_run.run_id if dag_run else kwargs.get('run_id', '')
    project_root = get_project_root()
    scripts_dir = os.path.join(project_root, 'scripts')
    config = getattr(dag_run, 'conf', None) or {}
    # seed_id 优先：自定义 ID，不依赖时间；reproduce_run_id 次之：复现某次 run
    seed_id = (config.get('seed_id') or config.get('reproduce_run_id') or '').strip() or None
    logging.info(f"generate_plan: run_id={run_id}, seed_id={seed_id!r} (from conf)")
    import subprocess
    cmd = [sys.executable, os.path.join(scripts_dir, 'run_plan_generator.py'), run_id]
    if seed_id:
        cmd.append(seed_id)
    subprocess.run(cmd, check=True, cwd=project_root)

# Define the DAG
with DAG(
    'sionna_etl_pipeline',
    default_args=default_args,
    description='Sionna Simulation ETL Pipeline',
    schedule=None,  # 修改这里：schedule_interval -> schedule
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=['sionna', 'etl'],
) as dag:

    # 1. Generate Plan（触发时 conf 传入 seed_id 或 reproduce_run_id 控制种子，不依赖时间）
    generate_plan = PythonOperator(
        task_id='generate_plan',
        python_callable=generate_plan_callable,
    )

    # 2. Simulation
    # Since there are multiple simulation points, we use a PythonOperator to loop through them.
    # Alternatively, we could use dynamic task mapping if Airflow version supports it.
    simulate = PythonOperator(
        task_id='simulate',
        python_callable=run_simulations_callable,
        op_kwargs={'run_id': '{{ run_id }}'},
    )

    # 3. Transform
    transform = BashOperator(
        task_id='transform',
        bash_command='python "{{ params.scripts_dir }}/transform_raw_to_table.py" "{{ run_id }}" "{{ params.project_root }}"',
        params={
            'scripts_dir': os.path.join(get_project_root(), 'scripts'),
            'project_root': get_project_root()
        },
    )

    # 4. Quality Check
    quality_check = BashOperator(
        task_id='quality_check',
        bash_command='python "{{ params.scripts_dir }}/data_quality_checks.py" "{{ run_id }}" "{{ params.project_root }}"',
        params={
            'scripts_dir': os.path.join(get_project_root(), 'scripts'),
            'project_root': get_project_root()
        },
    )

    # 5. KPI Calculation
    compute_kpis = BashOperator(
        task_id='compute_kpis',
        bash_command='python "{{ params.scripts_dir }}/compute_kpis.py" "{{ run_id }}" "{{ params.project_root }}"',
        params={
            'scripts_dir': os.path.join(get_project_root(), 'scripts'),
            'project_root': get_project_root()
        },
    )

    # 5.5 Human-in-the-loop: Review KPIs before generating report (需要 triggerer 服务)
    # Reject 时跳过 generate_report，需配合 BranchPythonOperator
    if HITLOperator and BranchPythonOperator:
        review_results = HITLOperator(
            task_id='review_results',
            subject="Review simulation KPIs before generating report",
            options=["Approve", "Reject"],
            body="Please review the KPIs in artifacts/<run_id>/kpis.json. Approve to proceed with report generation.",
        )
        skip_report = EmptyOperator(task_id='skip_report')

        def _branch_on_review(**context):
            payload = context['ti'].xcom_pull(task_ids='review_results')
            chosen = (payload or {}).get('chosen_options') or []
            if chosen and chosen[0] == "Approve":
                return "generate_report"
            return "skip_report"

        check_review = BranchPythonOperator(
            task_id='check_review',
            python_callable=_branch_on_review,
        )
    else:
        # Fallback for environments where HITLOperator is not available
        def review_kpis_callable(run_id, **kwargs):
            project_root = get_project_root()
            kpi_path = os.path.join(project_root, 'artifacts', _safe_run_id(run_id), 'kpis.json')
            logging.info(f"--- HITL Review Required ---")
            logging.info(f"Please review simulation KPIs at: {kpi_path}")
            if os.path.exists(kpi_path):
                with open(kpi_path, 'r') as f:
                    kpis = json.load(f)
                    logging.info(f"Overall Mean BLER: {kpis.get('overall', {}).get('mean_bler')}")
            logging.info(f"Proceeding to report generation...")

        review_results = PythonOperator(
            task_id='review_results',
            python_callable=review_kpis_callable,
            op_kwargs={'run_id': '{{ run_id }}'},
        )
        check_review = None
        skip_report = None

    # 6. Report Generation
    generate_report = BashOperator(
        task_id='generate_report',
        bash_command='python "{{ params.scripts_dir }}/report_generator.py" "{{ run_id }}" "{{ params.project_root }}"',
        params={
            'scripts_dir': os.path.join(get_project_root(), 'scripts'),
            'project_root': get_project_root()
        },
    )

    # Set dependencies
    generate_plan >> simulate >> transform >> quality_check >> compute_kpis >> review_results
    if check_review is not None:
        review_results >> check_review >> [generate_report, skip_report]
    else:
        review_results >> generate_report
