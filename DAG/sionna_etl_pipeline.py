from airflow import DAG
from airflow.providers.standard.operators.bash import BashOperator
from airflow.providers.standard.operators.python import PythonOperator
from datetime import datetime, timedelta
import os
import sys
import json

# Add scripts to sys.path to allow importing modules
SCRIPTS_DIR = os.path.join(os.environ.get('AIRFLOW_HOME', '/opt/airflow'), 'scripts')
if os.path.exists(SCRIPTS_DIR):
    sys.path.insert(0, SCRIPTS_DIR)

# Default arguments for the DAG
default_args = {
    'owner': 'cline',
    'depends_on_past': False,
    'start_date': datetime(2026, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

def get_project_root():
    # In Airflow context, we might need to be careful with paths
    # Assuming the current working directory or AIRFLOW_HOME is related to project root
    return os.environ.get('PROJECT_ROOT', 'd:/Project Computer Application/Airflow_ETL_Pipeline')

def run_simulations_callable(run_id, **kwargs):
    """
    Simulation step: Loads the run plan and executes all simulation rows.
    In a real production environment, this might be parallelized using Task Mapping.
    """
    project_root = get_project_root()
    from sionna_simulator import main as sim_main
    
    plan_path = os.path.join(project_root, 'artifacts', run_id, 'run_plan.json')
    with open(plan_path, 'r', encoding='utf-8') as f:
        plan_data = json.load(f)
    
    plan_size = plan_data.get('plan_size', 0)
    print(f"Starting {plan_size} simulations for run_id: {run_id}")
    
    for i in range(plan_size):
        print(f"Running simulation {i+1}/{plan_size}...")
        sim_main(run_id, i)

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

    # 1. Generate Plan
    generate_plan = BashOperator(
        task_id='generate_plan',
        bash_command='python "{{ params.scripts_dir }}/run_plan_generator.py" "{{ run_id }}"',
        params={'scripts_dir': os.path.join(get_project_root(), 'scripts')},
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
    generate_plan >> simulate >> transform >> quality_check >> compute_kpis >> generate_report
