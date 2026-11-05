from coevobench.core.protocol.spec import Budget, RunSpec


def test_run_spec_matches_paper_eq1() -> None:
    budget = Budget(hours=24, gpus=(0, 1, 2, 3), teacher_spend_usd=100.0)
    spec = RunSpec(
        bench="tb",
        researcher_model="glm-5.3",
        researcher_cli="claude-code",
        initial_weights="base/",
        initial_harness="templates/tb/initial",
        budget=budget,
        replicate=2,
        seed=42,
    )
    assert spec.run_id == "glm-5.3_tb_2"


def test_budget_defaults() -> None:
    budget = Budget(hours=24, gpus=(0, 1, 2, 3), teacher_spend_usd=100.0)
    assert budget.hours == 24
    assert len(budget.gpus) == 4
