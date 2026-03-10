from battery_deg_spme.config.settings import get_default_settings
from battery_deg_spme.fitting.cycle_pipeline import run_single_cycle_pipeline
from battery_deg_spme.evaluation.degree_comparison import run_degree_sweep


if __name__ == "__main__":
    settings = get_default_settings()
    result = run_single_cycle_pipeline(settings)

    sweep = run_degree_sweep(
        t_np=result["prep"]["t"],
        u_np=result["prep"]["u"],
        y_np=result["prep"]["y"],
        proxy=result["proxy"],
        cfg=result["cfg"],
        settings=settings,
    )
    for row in sweep:
        print(row["deg"], row["ct"]["rmse"], row["ls"]["rmse"], row["lsr"]["rmse"])