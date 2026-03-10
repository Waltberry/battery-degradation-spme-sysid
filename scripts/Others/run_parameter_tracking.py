from battery_deg_spme.config.settings import get_default_settings
from battery_deg_spme.fitting.cycle_pipeline import run_single_cycle_pipeline
from battery_deg_spme.analysis.parameter_extraction import extract_monitorable_parameters


if __name__ == "__main__":
    settings = get_default_settings()
    result = run_single_cycle_pipeline(settings)

    params = extract_monitorable_parameters(
        cfg=result["cfg"],
        stage2_result=result["stage2"],
        stage3a_result=result["stage3a"],
        stage3b_result=result["stage3b"],
    )
    print(params)