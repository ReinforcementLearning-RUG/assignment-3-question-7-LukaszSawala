from util import *
from model_free_prediction.monte_carlo_evaluator import MCEvaluator
from model_free_prediction.td_evaluator import TDEvaluator
from model_free_prediction.td_lambda_evaluator import TDLambdaEvaluator
def main() -> None:
    """
    Starting point of the program, you can instantiate any classes, run methods/functions here as needed.
    """
    mdp = create_mdp()
    policy_1 = create_policy_1()
    policy_2 = create_policy_2()

    print(f"Monte Carlo Evaluation:\n")
    evaluator_MCE = MCEvaluator(mdp)
    state_values_policy_1_MCE = evaluator_MCE.evaluate(policy_1, 1000)
    print(f"State values by MC for policy 1: {state_values_policy_1_MCE}")
    state_values_policy_2_MCE = evaluator_MCE.evaluate(policy_2, 1000)
    print(f"State values by MC for policy 2: {state_values_policy_2_MCE}\n")

    print(f"Temporal Differnce (0) Evaluation:\n")
    evaluator_TDE = TDEvaluator(mdp, alpha=0.1)
    state_values_policy_1_TDE = evaluator_TDE.evaluate(policy_1, 1000)
    print(f"State values by TD(0) for policy 1: {state_values_policy_1_TDE}")
    state_values_policy_2_TDE = evaluator_TDE.evaluate(policy_2, 1000)
    print(f"State values by TD(0) for policy 2: {state_values_policy_2_TDE}\n")

    print(f"Temporal Differnce (Lambda) Evaluation:\n")
    evaluator_TDLE = TDLambdaEvaluator(mdp, alpha=0.1, lambd=0.5)
    state_values_policy_1_TDLE = evaluator_TDLE.evaluate(policy_1, 1000)
    print(f"State values by TD(lambda) for policy 1: {state_values_policy_1_TDLE}")
    state_values_policy_2_TDLE = evaluator_TDLE.evaluate(policy_2, 1000)
    print(f"State values by TD(lambda) for policy 2: {state_values_policy_2_TDLE}")

    
if __name__ == "__main__":
    main()
