import os
import warnings
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols, mixedlm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import jarque_bera
warnings.filterwarnings("ignore")


OUTPUT_DIR = os.path.join("Synthetic Experiments", "ANOVA", "Outputs")

# Include quadratic terms in the extended model.
INCLUDE_QUADRATIC_MODEL = True

# Run mixed-effects sensitivity analysis when repeated configurations exist.
RUN_MIXED_MODEL = True

# Minimum repetitions required before considering a configuration repeated.
MIN_REPEATS_FOR_MIXED_MODEL = 2


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def print_separator(char="=", width=78):
    print(char * width)


def calculate_partial_eta_squared(anova_table):
    """
    Partial eta squared:

        eta_p^2 = SS_effect / (SS_effect + SS_error)

    This is appropriate for interpreting the magnitude of an ANOVA effect.
    """
    if "Residual" not in anova_table.index:
        return anova_table

    residual_ss = anova_table.loc["Residual", "sum_sq"]

    result = anova_table.copy()

    result["partial_eta_sq"] = (
        result["sum_sq"] /
        (result["sum_sq"] + residual_ss)
    )

    return result


def calculate_vif(df, predictors):
    """
    Calculate VIF for the supplied predictor matrix.
    """
    X = df[predictors].copy()
    X = sm.add_constant(X, has_constant="add")

    vif_rows = []

    for i, column in enumerate(X.columns):
        if column == "const":
            continue

        try:
            vif_value = variance_inflation_factor(X.values, i)
        except Exception:
            vif_value = np.inf

        vif_rows.append({
            "variable": column,
            "VIF": vif_value
        })

    return pd.DataFrame(vif_rows)


def standardized_coefficients(df, formula):
    """
    Fit an OLS model after standardizing numeric predictors and response.
    Useful for comparing relative linear effect magnitude.
    """

    import re

    # Extract variables from the formula.
    rhs = formula.split("~", 1)[1]

    variable_names = set(
        re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", rhs)
    )

    # Remove formula operators / Python-like tokens.
    excluded = {
        "I",
        "np",
        "log",
        "sqrt",
        "abs"
    }

    predictors = [
        x for x in variable_names
        if x not in excluded and x in df.columns
    ]

    if "f1_score" not in df.columns or not predictors:
        return pd.DataFrame()

    temp = df[["f1_score"] + predictors].copy()

    # Standardize all numeric variables.
    for col in temp.columns:
        std = temp[col].std(ddof=0)

        if std == 0 or pd.isna(std):
            temp[col] = 0.0
        else:
            temp[col] = (temp[col] - temp[col].mean()) / std

    try:
        model = ols(formula, data=temp).fit()
    except Exception:
        return pd.DataFrame()

    coefficients = pd.DataFrame({
        "term": model.params.index,
        "standardized_coefficient": model.params.values,
        "p_value": model.pvalues.values
    })

    return coefficients


def print_data_diagnostics(df, required_columns):
    """
    Print and return numeric conversion diagnostics.
    """
    print("\nColumn diagnostics:")
    print_separator("-")

    diagnostics = []

    for col in required_columns:

        if col not in df.columns:
            diagnostics.append({
                "column": col,
                "dtype_before": "MISSING",
                "valid": 0,
                "invalid": len(df),
                "missing": len(df)
            })

            print(
                f"{col:25s} "
                f"MISSING"
            )
            continue

        original_dtype = str(df[col].dtype)

        numeric = pd.to_numeric(
            df[col],
            errors="coerce"
        )

        invalid = int(numeric.isna().sum())
        valid = int(numeric.notna().sum())

        diagnostics.append({
            "column": col,
            "dtype_before": original_dtype,
            "valid": valid,
            "invalid": invalid,
            "missing": invalid
        })

        print(
            f"{col:25s} "
            f"dtype={original_dtype:12s} "
            f"valid={valid:8,d} "
            f"invalid={invalid:8,d}"
        )

    return pd.DataFrame(diagnostics)


def create_configuration_id(df, predictors):
    """
    Create an identifier for each exact hyperparameter configuration.

    Rows sharing the same configuration ID are stochastic repetitions.
    """
    config_cols = [
        col for col in predictors
        if col in df.columns
    ]

    if not config_cols:
        return pd.Series(
            np.arange(len(df)),
            index=df.index,
            name="configuration_id"
        )

    # Convert values to strings to make grouping robust.
    return (
        df[config_cols]
        .astype(str)
        .agg("|".join, axis=1)
    )


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def run_n_way_anova(csv_file, is_dual=False):

    ensure_output_dir()

    print_separator()
    print(f"Loading data from {csv_file}...")
    print_separator()

    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"ERROR: {csv_file} not found.")
        return

    print(f"Raw rows: {len(df):,}")
    print(f"Raw columns: {len(df.columns)}")

    # -------------------------------------------------------------------------
    # CONTAMINATION RATE
    # -------------------------------------------------------------------------

    if "contamination_rate" not in df.columns:
        print(
            "ERROR: 'contamination_rate' column not found in the CSV."
        )
        return

    df["contamination_rate"] = pd.to_numeric(
        df["contamination_rate"],
        errors="coerce"
    )

    valid_contamination = df["contamination_rate"].notna().sum()

    print(
        f"Using contamination_rate: "
        f"{valid_contamination:,} valid values."
    )

    # -------------------------------------------------------------------------
    # MODEL VARIABLES
    # -------------------------------------------------------------------------

    if is_dual:

        title = (
            "THE MASTER N-WAY ANOVA RESULTS "
            "(DUAL MAGIC NUMBERS)"
        )

        predictors = [
            "std",
            "center_distance",
            "num_samples",
            "gauss_threshold",
            "overlap_threshold",
            "gap_distance",
            "contamination_rate"
        ]

    else:

        title = (
            "THE MASTER N-WAY ANOVA RESULTS "
            "(SINGLE MAGIC NUMBER)"
        )

        predictors = [
            "std",
            "center_distance",
            "num_samples",
            "gauss_threshold",
            "overlap_threshold",
            "contamination_rate"
        ]

    response = "f1_score"

    required_columns = predictors + [response]

    # -------------------------------------------------------------------------
    # CHECK MISSING COLUMNS
    # -------------------------------------------------------------------------

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:

        print("\nERROR: Missing required columns:")

        for col in missing_columns:
            print(f"  - {col}")

        return

    # -------------------------------------------------------------------------
    # NUMERIC CONVERSION
    # -------------------------------------------------------------------------

    diagnostics = print_data_diagnostics(
        df,
        required_columns
    )

    diagnostics.to_csv(
        os.path.join(
            OUTPUT_DIR,
            f"{os.path.basename(csv_file).replace('.csv', '')}"
            "_column_diagnostics.csv"
        ),
        index=False
    )

    print("\nConverting analysis variables to numeric...")

    for col in required_columns:

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    # -------------------------------------------------------------------------
    # INFINITY HANDLING
    # -------------------------------------------------------------------------

    df[required_columns] = df[
        required_columns
    ].replace(
        [np.inf, -np.inf],
        np.nan
    )

    # -------------------------------------------------------------------------
    # ROW-LEVEL VALIDITY
    # -------------------------------------------------------------------------

    print("\nRows available for analysis:")
    print_separator("-")

    valid_mask = df[required_columns].notna().all(axis=1)

    print(f"Rows with all required variables: {valid_mask.sum():,}")
    print(f"Rows with at least one invalid value: {(~valid_mask).sum():,}")

    # Show which variables are responsible.
    print("\nInvalid values by variable:")

    for col in required_columns:

        invalid_count = int(df[col].isna().sum())

        print(
            f"  {col:25s}: "
            f"{invalid_count:8,d}"
        )

    # -------------------------------------------------------------------------
    # IMPORTANT:
    # DO NOT SILENTLY DESTROY THE DATASET.
    #
    # We only remove rows that actually cannot participate in THIS model.
    # -------------------------------------------------------------------------

    before_cleaning = len(df)

    df = df.loc[
        valid_mask,
        required_columns
    ].copy()

    after_cleaning = len(df)

    print("\nCleaning summary:")
    print_separator("-")
    print(f"Rows before cleaning: {before_cleaning:,}")
    print(f"Rows after cleaning:  {after_cleaning:,}")
    print(f"Rows removed:         {before_cleaning - after_cleaning:,}")

    if after_cleaning == 0:

        print("\nERROR: No valid rows remain after cleaning.")
        print(
            "The column diagnostics above show exactly which "
            "variable caused the problem."
        )
        return

    # -------------------------------------------------------------------------
    # BASIC RANGE CHECKS
    # -------------------------------------------------------------------------

    print("\nBasic variable ranges:")
    print_separator("-")

    for col in predictors + [response]:

        print(
            f"{col:25s} "
            f"min={df[col].min():.8g} "
            f"max={df[col].max():.8g} "
            f"mean={df[col].mean():.8g} "
            f"std={df[col].std():.8g}"
        )

    # -------------------------------------------------------------------------
    # CONFIGURATION ID
    # -------------------------------------------------------------------------

    df["configuration_id"] = create_configuration_id(
        df,
        predictors
    )

    configuration_counts = (
        df["configuration_id"]
        .value_counts()
    )

    repeated_configurations = (
        configuration_counts[
            configuration_counts >= MIN_REPEATS_FOR_MIXED_MODEL
        ]
    )

    print("\nStochastic configuration structure:")
    print_separator("-")

    print(
        f"Unique configurations: "
        f"{configuration_counts.size:,}"
    )

    print(
        f"Repeated configurations: "
        f"{len(repeated_configurations):,}"
    )

    if len(repeated_configurations) > 0:
        print(
            f"Maximum repetitions of one configuration: "
            f"{configuration_counts.max():,}"
        )

    # -------------------------------------------------------------------------
    # BASE OLS FORMULA
    # -------------------------------------------------------------------------

    predictor_string = " + ".join(predictors)

    formula_linear = (
        f"{response} ~ ({predictor_string}) ** 2"
    )

    print("\nFitting linear + 2-way interaction model:")
    print(formula_linear)

    # -------------------------------------------------------------------------
    # FIT MODEL
    # -------------------------------------------------------------------------

    try:

        model = ols(
            formula_linear,
            data=df
        ).fit()

    except Exception as exc:

        print(f"\nERROR fitting OLS model: {exc}")
        return

    # -------------------------------------------------------------------------
    # TYPE II ANOVA
    # -------------------------------------------------------------------------

    print("\nRunning Type II ANOVA...")

    try:

        anova_type2 = sm.stats.anova_lm(
            model,
            typ=2
        )

        anova_type2 = calculate_partial_eta_squared(
            anova_type2
        )

        anova_type2 = anova_type2.sort_values(
            by="sum_sq",
            ascending=False
        )

    except Exception as exc:

        print(f"ERROR calculating Type II ANOVA: {exc}")
        anova_type2 = pd.DataFrame()

    # -------------------------------------------------------------------------
    # TYPE III ANOVA
    # -------------------------------------------------------------------------

    print("Running Type III ANOVA...")

    try:

        anova_type3 = sm.stats.anova_lm(
            model,
            typ=3
        )

        anova_type3 = calculate_partial_eta_squared(
            anova_type3
        )

        anova_type3 = anova_type3.sort_values(
            by="sum_sq",
            ascending=False
        )

    except Exception as exc:

        print(f"ERROR calculating Type III ANOVA: {exc}")
        anova_type3 = pd.DataFrame()

    # -------------------------------------------------------------------------
    # PRINT MAIN RESULTS
    # -------------------------------------------------------------------------

    print("\n")
    print_separator()

    print(
        f" {title}"
    )

    print_separator()

    if not anova_type2.empty:

        display_columns = [
            col for col in [
                "sum_sq",
                "df",
                "F",
                "PR(>F)",
                "partial_eta_sq"
            ]
            if col in anova_type2.columns
        ]

        print(
            anova_type2[
                display_columns
            ].to_string()
        )

    print_separator()

    # -------------------------------------------------------------------------
    # MODEL FIT STATISTICS
    # -------------------------------------------------------------------------

    fit_statistics = pd.DataFrame([{
        "n_observations": int(model.nobs),
        "r_squared": model.rsquared,
        "adjusted_r_squared": model.rsquared_adj,
        "AIC": model.aic,
        "BIC": model.bic,
        "F_statistic": model.fvalue,
        "F_p_value": model.f_pvalue
    }])

    print("\nModel fit:")
    print_separator("-")
    print(
        fit_statistics.to_string(index=False)
    )

    # -------------------------------------------------------------------------
    # ROBUST COEFFICIENTS — HC3
    # -------------------------------------------------------------------------

    print("\nHC3 robust coefficient inference...")

    robust_model = model.get_robustcov_results(
        cov_type="HC3"
    )

    robust_coefficients = pd.DataFrame({
        "term": model.params.index,
        "coefficient": robust_model.params,
        "standard_error_HC3": robust_model.bse,
        "t_HC3": robust_model.tvalues,
        "p_value_HC3": robust_model.pvalues
    })

    # -------------------------------------------------------------------------
    # STANDARDIZED COEFFICIENTS
    # -------------------------------------------------------------------------

    print("Calculating standardized coefficients...")

    standardized = standardized_coefficients(
        df,
        formula_linear
    )

    # -------------------------------------------------------------------------
    # VIF
    # -------------------------------------------------------------------------

    print("Calculating VIF...")

    vif = calculate_vif(
        df,
        predictors
    )

    # -------------------------------------------------------------------------
    # RESIDUAL DIAGNOSTICS
    # -------------------------------------------------------------------------

    residuals = model.resid
    fitted = model.fittedvalues

    # Breusch-Pagan
    try:

        bp_test = het_breuschpagan(
            residuals,
            model.model.exog
        )

        bp_results = pd.DataFrame([{
            "LM_statistic": bp_test[0],
            "LM_p_value": bp_test[1],
            "F_statistic": bp_test[2],
            "F_p_value": bp_test[3]
        }])

    except Exception:

        bp_results = pd.DataFrame()

    # Jarque-Bera
    try:

        jb = jarque_bera(
            residuals
        )

        jb_results = pd.DataFrame([{
            "JB_statistic": jb[0],
            "JB_p_value": jb[1],
            "skewness": jb[2],
            "kurtosis": jb[3]
        }])

    except Exception:

        jb_results = pd.DataFrame()

    # -------------------------------------------------------------------------
    # QUADRATIC MODEL
    # -------------------------------------------------------------------------

    quadratic_model = None
    quadratic_comparison = pd.DataFrame()

    if INCLUDE_QUADRATIC_MODEL:

        print("\nFitting quadratic model...")

        quadratic_terms = [
            f"I({x} ** 2)"
            for x in predictors
        ]

        formula_quadratic = (
            f"{response} ~ "
            f"({predictor_string}) ** 2 + "
            + " + ".join(quadratic_terms)
        )

        try:

            quadratic_model = ols(
                formula_quadratic,
                data=df
            ).fit()

            # Compare nested models with an F-test.
            comparison = sm.stats.anova_lm(
                model,
                quadratic_model
            )

            quadratic_comparison = comparison

            print("\nLinear vs quadratic model comparison:")
            print_separator("-")
            print(
                comparison.to_string()
            )

        except Exception as exc:

            print(
                f"Quadratic model failed: {exc}"
            )

    # -------------------------------------------------------------------------
    # MIXED-EFFECTS MODEL
    # -------------------------------------------------------------------------

    mixed_results = None

    if RUN_MIXED_MODEL and len(repeated_configurations) > 0:

        print(
            "\nRunning mixed-effects sensitivity analysis..."
        )

        try:

            mixed_formula = (
                f"{response} ~ "
                + predictor_string
            )

            mixed_model = mixedlm(
                mixed_formula,
                data=df,
                groups=df["configuration_id"]
            )

            mixed_results = mixed_model.fit(
                reml=True,
                method="lbfgs",
                disp=False
            )

            print_separator("-")
            print(
                mixed_results.summary()
            )

        except Exception as exc:

            print(
                f"Mixed-effects model could not be fitted: {exc}"
            )
            mixed_results = None

    # -------------------------------------------------------------------------
    # SAVE RESULTS
    # -------------------------------------------------------------------------

    base_name = os.path.splitext(
        os.path.basename(csv_file)
    )[0]

    suffix = (
        "_dual"
        if is_dual
        else "_single"
    )

    if not anova_type2.empty:

        anova_type2.to_csv(
            os.path.join(
                OUTPUT_DIR,
                f"{base_name}{suffix}_ANOVA_TypeII.csv"
            )
        )

    if not anova_type3.empty:

        anova_type3.to_csv(
            os.path.join(
                OUTPUT_DIR,
                f"{base_name}{suffix}_ANOVA_TypeIII.csv"
            )
        )

    robust_coefficients.to_csv(
        os.path.join(
            OUTPUT_DIR,
            f"{base_name}{suffix}_HC3_coefficients.csv"
        ),
        index=False
    )

    standardized.to_csv(
        os.path.join(
            OUTPUT_DIR,
            f"{base_name}{suffix}_standardized_coefficients.csv"
        ),
        index=False
    )

    vif.to_csv(
        os.path.join(
            OUTPUT_DIR,
            f"{base_name}{suffix}_VIF.csv"
        ),
        index=False
    )

    fit_statistics.to_csv(
        os.path.join(
            OUTPUT_DIR,
            f"{base_name}{suffix}_model_fit.csv"
        ),
        index=False
    )

    bp_results.to_csv(
        os.path.join(
            OUTPUT_DIR,
            f"{base_name}{suffix}_Breusch_Pagan.csv"
        ),
        index=False
    )

    jb_results.to_csv(
        os.path.join(
            OUTPUT_DIR,
            f"{base_name}{suffix}_Jarque_Bera.csv"
        ),
        index=False
    )

    if not quadratic_comparison.empty:

        quadratic_comparison.to_csv(
            os.path.join(
                OUTPUT_DIR,
                f"{base_name}{suffix}_quadratic_comparison.csv"
            )
        )

    # Save configuration repetition information.
    configuration_counts.rename(
        "repetitions"
    ).to_csv(
        os.path.join(
            OUTPUT_DIR,
            f"{base_name}{suffix}_configuration_repetitions.csv"
        )
    )

    # Save mixed model summary.
    if mixed_results is not None:

        with open(
            os.path.join(
                OUTPUT_DIR,
                f"{base_name}{suffix}_mixed_model.txt"
            ),
            "w",
            encoding="utf-8"
        ) as f:

            f.write(
                str(
                    mixed_results.summary()
                )
            )

    # -------------------------------------------------------------------------
    # FINAL SUMMARY
    # -------------------------------------------------------------------------

    print("\n")
    print_separator()

    print("ANALYSIS COMPLETE")

    print_separator()

    print(
        f"Dataset:               {csv_file}"
    )

    print(
        f"Observations analyzed: {len(df):,}"
    )

    print(
        f"Predictors:             {len(predictors)}"
    )

    print(
        f"R²:                     {model.rsquared:.6f}"
    )

    print(
        f"Adjusted R²:            {model.rsquared_adj:.6f}"
    )

    if not anova_type2.empty:

        # Exclude residual row.
        effects = anova_type2.drop(
            index="Residual",
            errors="ignore"
        )

        if not effects.empty:

            strongest_effect = effects.iloc[0]

            print(
                "\nLargest Type-II ANOVA effect:"
            )

            print(
                f"  {effects.index[0]}"
            )

            print(
                f"  Sum of squares: "
                f"{strongest_effect['sum_sq']:.8g}"
            )

            if "partial_eta_sq" in strongest_effect:

                print(
                    f"  Partial eta²: "
                    f"{strongest_effect['partial_eta_sq']:.6f}"
                )

    print_separator()

    print(
        f"Results saved in: {os.path.abspath(OUTPUT_DIR)}"
    )

    print_separator()
    print()


# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    
    data_dir = os.path.join("Synthetic Experiments", "Data")

    run_n_way_anova(
        os.path.join(data_dir, "1_magic_numbers_results.csv"),
        is_dual=False
    )

    run_n_way_anova(
        os.path.join(data_dir, "2_magic_numbers_results.csv"),
        is_dual=True
    )

    print("\nAll analyses finished.")