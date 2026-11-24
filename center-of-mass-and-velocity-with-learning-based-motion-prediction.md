# Portfolio Project Data — Center of Mass and Velocity with Learning-Based Motion Prediction

- **Category:** `machine-learning` (`Machine Learning`)
- **Accent Color:** `indigo`
- **Target Assets Folder:** `public/images/projects/center_of_mass_and/`
- **Required Local Images:** `banner.webp`, `image01.webp` .. `image15.webp`

## TypeScript Data (`ProjectCaseStudy`)

```typescript
{
  "id": "center-of-mass-and-velocity-with-learning-based-motion-prediction",
  "code": "PRO-16 // ML-03",
  "title": "Center of Mass and Velocity with Learning-Based Motion Prediction",
  "category": "machine-learning",
  "categoryLabel": "Supervised Learning",
  "year": "2023",
  "status": "Completed",
  "accentColor": "indigo",
  "repoUrl": "https://github.com/weztcy/Center-of-Mass-and-Velocity-with-Learning-Based-Motion-Prediction",
  "license": "MIT License",
  "banner": "/images/projects/center_of_mass_and/banner.webp",
  "summary": "Multi-target linear regression pipeline predicting 3D center-of-mass motion trajectories from noisy multi-particle physics simulations.",
  "projectDescription": "A computational kinematics and supervised learning system combining 3D center-of-mass physics calculations with multi-target Linear Regression trained on 3,000 noisy synthetic motion scenarios, paired with 300 DPI diagnostics and an interactive dark-mode Streamlit studio.",
  "architectureSummary": "Multi-Particle Physics Simulator -> Gaussian Noise Injection (3,000 CSV Scenarios) -> 80/20 Train-Test Split -> Multi-Target LinearRegression -> Physics vs. ML 3D Comparator & Streamlit Studio",
  "techStack": [
    "Python",
    "Scikit-Learn",
    "NumPy",
    "Pandas",
    "Matplotlib",
    "Streamlit",
    "Jupyter Notebook"
  ],
  "telemetryMatrix": {
    "projectType": "Supervised Multi-Target Regression",
    "targetPlatform": "Python CLI, Notebook & Streamlit Web",
    "primaryDomain": "3D Kinematics & Motion Prediction",
    "architecturePipeline": "Physics Sim -> OLS Regression -> 3D Eval",
    "testingAndAudit": "Verified",
    "projectStatus": "Completed",
    "projectYear": "2023"
  },
  "mlProjectInfo": {
    "projectType": "Multi-Target Motion Regression Pipeline",
    "learningParadigm": "Supervised Learning",
    "modelDomain": "3D Multi-Particle Kinematics",
    "projectStatus": "Completed",
    "developmentYear": "2023",
    "capabilityBadges": [
      "3D Motion Prediction",
      "Synthetic Physics Simulation",
      "Interactive Streamlit Studio"
    ]
  },
  "mlProblemContext": "Tracking multi-particle center-of-mass motion in real sensor environments is degraded by measurement uncertainty on initial positions and displacements. This project investigates how supervised regression recovers exact constant-velocity 3D kinematics directly from noise-perturbed observations without hardcoded rules.",
  "mlDatasetOverview": {
    "profileBadge": "3K Synthetic 3D Scenarios",
    "source": "Self-generated via buat_dataset_sintetis() in program.py (dataset_center_of_mass_ml.csv)",
    "modality": "Numerical Tabular Kinematics",
    "scale": "3,000 Multi-Particle Scenarios (14 Columns)",
    "inputShape": "N x 6 Continuous Features",
    "dataFormat": "CSV (Float64 Precision)",
    "description": "Each row simulates 2 to 6 randomized objects with masses (0.5-10.0 kg), 3D positions (-20 to 20 m), 3D velocities (-5 to 5 m/s), and time horizons (0.1-5.0 s). Six noisy input features drive three noiseless future coordinate targets across a 2,400/600 split.",
    "provenanceNote": "100% synthetic physics simulation generated deterministically with NumPy default_rng(42); no external dataset dependencies or licensing restrictions."
  },
  "mlFeaturesAndDataCharacteristics": {
    "spaceBadge": "6D Input -> 3D Target Space",
    "spaceNote": "Input features inject sigma = 0.05 m Gaussian noise while target coordinates preserve exact analytical ground truth.",
    "objectiveOutput": {
      "label": "Target Output",
      "name": "cm_x_masa_depan, cm_y_masa_depan, cm_z_masa_depan",
      "type": "Multi-Target Continuous Vector",
      "distribution": "X: [-34.87, 33.77] m | Y: [-37.24, 30.43] m | Z: [-31.46, 34.29] m"
    },
    "inputSignals": [
      {
        "name": "cm_x_awal_terukur",
        "type": "Numerical Continuous",
        "importanceOrRole": "Primary Weight (~1.0000)",
        "description": "Measured initial X-axis center of mass perturbed by zero-mean Gaussian noise (sigma = 0.05 m)."
      },
      {
        "name": "cm_y_awal_terukur",
        "type": "Numerical Continuous",
        "importanceOrRole": "Primary Weight (~0.9998)",
        "description": "Measured initial Y-axis center of mass perturbed by zero-mean Gaussian noise (sigma = 0.05 m)."
      },
      {
        "name": "cm_z_awal_terukur",
        "type": "Numerical Continuous",
        "importanceOrRole": "Primary Weight (~0.9999)",
        "description": "Measured initial Z-axis center of mass perturbed by zero-mean Gaussian noise (sigma = 0.05 m)."
      },
      {
        "name": "perpindahan_x_terukur",
        "type": "Numerical Continuous",
        "importanceOrRole": "Primary Weight (~1.0001)",
        "description": "Measured X-axis center-of-mass displacement (V_cm_x * t) with independent Gaussian measurement noise."
      },
      {
        "name": "perpindahan_y_terukur",
        "type": "Numerical Continuous",
        "importanceOrRole": "Primary Weight (~0.9996)",
        "description": "Measured Y-axis center-of-mass displacement (V_cm_y * t) with independent Gaussian measurement noise."
      },
      {
        "name": "perpindahan_z_terukur",
        "type": "Numerical Continuous",
        "importanceOrRole": "Primary Weight (~0.9993)",
        "description": "Measured Z-axis center-of-mass displacement (V_cm_z * t) with independent Gaussian measurement noise."
      }
    ],
    "dataCharacteristicsSummary": "The six input signals span symmetric 3D spatial bounds with independent 0.05 m Gaussian measurement noise, creating an expected per-axis noise floor near 0.0707 m while maintaining strict linear additivity between initial center-of-mass position and displacement."
  },
  "mlLearningApproach": {
    "paradigm": "Supervised Learning — Multi-Target Continuous Regression",
    "methodologyDescription": "The pipeline casts 3D constant-velocity motion prediction as a supervised multi-output regression task. Using six noise-perturbed kinematic features, a single LinearRegression estimator learns a 3x6 mapping matrix to jointly predict future X, Y, and Z coordinates.",
    "justification": "Because constant-velocity center-of-mass translation obeys linear superposition across orthogonal axes, Ordinary Least Squares achieves minimal estimation variance without overfitting to Gaussian sensor noise, outperforming high-variance nonlinear regressors in interpretability and speed."
  },
  "mlDataPreparation": [
    {
      "step": "01",
      "title": "Multi-Particle Physics Simulation",
      "description": "Sample 2-6 objects per scenario with uniform masses, 3D coordinates, velocities, and time horizons using NumPy default_rng(42).",
      "badge": "Simulation"
    },
    {
      "step": "02",
      "title": "Analytical Vector Aggregation",
      "description": "Compute exact mass-weighted initial center of mass, center-of-mass velocity, displacement, and noiseless future target coordinates.",
      "badge": "Kinematics"
    },
    {
      "step": "03",
      "title": "Gaussian Measurement Noise Injection",
      "description": "Add independent N(0, 0.05^2) perturbations strictly to the 6 measured input features while keeping target coordinates exact.",
      "badge": "Noise Model"
    },
    {
      "step": "04",
      "title": "Deterministic Holdout Splitting",
      "description": "Partition the 3,000-row CSV dataset into 2,400 training and 600 test scenarios using train_test_split with random_state=42.",
      "badge": "80/20 Split"
    }
  ],
  "mlDataPreparationNote": "Gaussian noise is applied exclusively to input features prior to train_test_split (seed 42), preventing target leakage and ensuring unbiased holdout evaluation.",
  "mlModelArchitectureAndTraining": {
    "algorithmName": "Scikit-Learn Multi-Target Linear Regression",
    "architectureSummary": "Closed-form Ordinary Least Squares linear model learning a 3x6 coefficient matrix W and 3D bias vector b mapping noisy [CM_0, dR] inputs directly to future [X, Y, Z] coordinates.",
    "configurationBadge": "OLS Closed-Form",
    "hyperparameters": [
      {
        "name": "fit_intercept",
        "value": "True",
        "description": "Estimates 3D bias term b to absorb finite-sample residual offsets across axes."
      },
      {
        "name": "test_size",
        "value": "0.2",
        "description": "Reserves 600 unseen simulation scenarios (20%) for held-out regression evaluation."
      },
      {
        "name": "noise_std",
        "value": "0.05 m",
        "description": "Standard deviation of isotropic Gaussian noise added to measured initial CM and displacement."
      },
      {
        "name": "random_state",
        "value": "42",
        "description": "Fixes both synthetic dataset generation and train-test partitioning for reproducibility."
      }
    ],
    "objectiveCriterion": "Residual Sum of Squares (Ordinary Least Squares L2 Loss)",
    "optimizationSolver": "LAPACK Singular Value Decomposition (Closed-Form OLS)",
    "trainingProcess": "Training executes in a single deterministic pass over 2,400 training rows via Scikit-Learn's closed-form OLS solver with fit_intercept enabled. Without iterative epochs, the fitted weight matrix converges directly to near-unit diagonals and zero cross-terms, achieving immediate convergence."
  },
  "mlEvaluationStrategy": {
    "validationMethod": "80/20 Holdout Train-Test Split with Analytical Physics Baseline",
    "description": "The fitted regressor is evaluated on 600 unseen test scenarios using overall and per-axis MAE, RMSE, and R-squared, complemented by residual histogram diagnostics and direct 3D trajectory comparison against exact analytical physics formulas.",
    "metrics": [
      {
        "label": "OVERALL R2",
        "value": "0.999927",
        "benchmark": "Target > 0.9900"
      },
      {
        "label": "OVERALL MAE",
        "value": "0.058072 m",
        "benchmark": "Tolerance < 0.1000"
      },
      {
        "label": "OVERALL RMSE",
        "value": "0.072610 m",
        "benchmark": "Baseline < 0.0750"
      },
      {
        "label": "TRAJECTORY DEV",
        "value": "0.00048 m",
        "benchmark": "Tolerance < 0.0050"
      }
    ]
  },
  "mlPerformanceVisualizations": [
    {
      "title": "Interactive Multi-Object Physics Calculator & System KPIs",
      "badge": "Streamlit UI",
      "imageUrl": "/images/projects/center_of_mass_and/image01.webp",
      "telemetryNote": "Displays editable 4-object preset grid with live total mass (14.70 kg), CM, velocity, and momentum KPIs."
    },
    {
      "title": "2D Center of Mass Spatial Distribution (X-Y Plane)",
      "badge": "2D Scatter",
      "imageUrl": "/images/projects/center_of_mass_and/image02.webp",
      "telemetryNote": "Plots 4 weighted particles and locates the exact center of mass at (-0.07, -0.77) m."
    },
    {
      "title": "2D Velocity Vector Quiver Diagram",
      "badge": "Quiver Plot",
      "imageUrl": "/images/projects/center_of_mass_and/image03.webp",
      "telemetryNote": "Visualizes individual particle velocity vectors alongside the resultant V_cm at (-0.27, 0.83) m/s."
    },
    {
      "title": "ML Training Configuration & Holdout Evaluation Metrics",
      "badge": "Metrics Panel",
      "imageUrl": "/images/projects/center_of_mass_and/image04.webp",
      "telemetryNote": "Reports 2,400/600 split metrics: Overall R2 = 0.999927, MAE = 0.058072 m, RMSE = 0.072610 m."
    },
    {
      "title": "Actual vs. Predicted Regression Parity (Dark Mode)",
      "badge": "Parity Plot",
      "imageUrl": "/images/projects/center_of_mass_and/image05.webp",
      "telemetryNote": "Per-axis test predictions align tightly with the ideal y = x diagonal (R2 >= 0.99992 across X, Y, Z)."
    },
    {
      "title": "Residual Error Density Histograms (Dark Mode)",
      "badge": "Residual Plot",
      "imageUrl": "/images/projects/center_of_mass_and/image06.webp",
      "telemetryNote": "Zero-centered bell-shaped error distributions with near-zero means (-0.0019, 0.0052, -0.0002 m)."
    },
    {
      "title": "Learned Weight Matrix & Bias Vector Inspection",
      "badge": "Weight Matrix",
      "imageUrl": "/images/projects/center_of_mass_and/image07.webp",
      "telemetryNote": "Diagonal weights converge to 0.99933-1.00015 while orthogonal cross-axis coefficients stay near zero."
    },
    {
      "title": "Physics vs. ML Future Position Comparator (t = 2.0 s)",
      "badge": "Comparator",
      "imageUrl": "/images/projects/center_of_mass_and/image08.webp",
      "telemetryNote": "Analytical physics (-0.622, 0.895, 2.826) vs. ML (-0.621, 0.895, 2.826) with 0.00048 m deviation."
    },
    {
      "title": "3D Trajectory Comparison Plot (Physics vs. ML)",
      "badge": "3D Trajectory",
      "imageUrl": "/images/projects/center_of_mass_and/image09.webp",
      "telemetryNote": "Illustrates overlapping constant-velocity and ML-predicted 3D paths from t=0 to t=2.0 s."
    },
    {
      "title": "Unified 4-in-1 Summary Dashboard View",
      "badge": "Composite UI",
      "imageUrl": "/images/projects/center_of_mass_and/image10.webp",
      "telemetryNote": "Integrates 2D CM, 2D velocity quiver, X-axis regression parity, and 3D trajectory with PNG download."
    },
    {
      "title": "Mathematical Formulations & Theory Documentation",
      "badge": "Theory Docs",
      "imageUrl": "/images/projects/center_of_mass_and/image11.webp",
      "telemetryNote": "Presents LaTeX equations for multi-particle CM, velocity, momentum, and noisy linear modeling."
    },
    {
      "title": "300 DPI Export: 2D Center of Mass & Velocity Quiver",
      "badge": "300 DPI PNG",
      "imageUrl": "/images/projects/center_of_mass_and/image12.webp",
      "telemetryNote": "Publication-ready white-background 2D plots exported via extract_visualizations.py."
    },
    {
      "title": "300 DPI Export: 3D Center of Mass & Velocity Vectors",
      "badge": "300 DPI 3D",
      "imageUrl": "/images/projects/center_of_mass_and/image13.webp",
      "telemetryNote": "High-resolution 3D spatial positions with projection drop lines and 3D velocity vector arrows."
    },
    {
      "title": "300 DPI Export: 6-Panel Regression & Residual Suite",
      "badge": "300 DPI Eval",
      "imageUrl": "/images/projects/center_of_mass_and/image14.webp",
      "telemetryNote": "Combines X, Y, Z parity scatter plots and residual error density histograms at 300 DPI."
    },
    {
      "title": "300 DPI Export: 3D Physics vs. ML Prediction Comparison",
      "badge": "300 DPI Path",
      "imageUrl": "/images/projects/center_of_mass_and/image15.webp",
      "telemetryNote": "Documents 0.0005 m Euclidean deviation between analytical physics and ML prediction at t = 2.0 s."
    }
  ],
  "mlVisualizationTelemetryNote": "Test set parity plots (R2 = 0.999927) and zero-centered residual histograms confirm unbiased generalization bounded solely by the injected 0.05 m Gaussian noise.",
  "mlModelInterpretationAndLimitations": {
    "interpretationSummary": "Inspecting the 3x6 coefficient matrix shows the regressor autonomously reconstructs the kinematic addition law: matched initial position and displacement weights converge between 0.9993 and 1.0001, while orthogonal cross-axis coefficients collapse near zero.",
    "keyDrivers": [
      {
        "factor": "Matched Initial CM Coordinate (cm_*_awal_terukur)",
        "impact": "Drives ~1.0000 unit weight contribution along the corresponding X, Y, or Z target axis."
      },
      {
        "factor": "Matched Displacement Vector (perpindahan_*_terukur)",
        "impact": "Contributes ~0.9993-1.0001 linear scaling along each target axis to project future motion."
      },
      {
        "factor": "Isotropic Gaussian Noise (sigma = 0.05 m)",
        "impact": "Establishes the irreducible ~0.0726 m RMSE floor across independent test samples."
      }
    ],
    "limitations": [
      {
        "title": "Constant-Velocity Kinematic Assumption",
        "description": "Training data models force-free uniform translation and cannot predict accelerating, rotational, or gravitational trajectories."
      },
      {
        "title": "Synthetic Noise & Sensor Idealization",
        "description": "Assumes zero-mean Gaussian measurement perturbations rather than real hardware bias, drift, or outlier spikes."
      },
      {
        "title": "In-Memory Model Persistence",
        "description": "Models are fitted and cached at runtime without exporting serialized .pkl or .onnx weight artifacts."
      }
    ],
    "futureImprovements": [
      {
        "title": "Nonlinear & Accelerated Motion Dynamics",
        "target": "Extend dataset simulation and polynomial/neural regressors to capture time-varying forces and acceleration."
      },
      {
        "title": "Real Telemetry & Sensor Validation",
        "target": "Benchmark model robustness on physical IMU, LiDAR, or multi-object video tracking logs."
      },
      {
        "title": "Model Serialization & Artifact Export",
        "target": "Persist trained estimators to .joblib or .onnx for reproducible standalone deployment."
      }
    ],
    "discoveryNote": "Direct matrix inspection replaces black-box explainability tools by verifying exact physical coefficient recovery."
  },
  "mlModelArtifacts": [
    {
      "name": "dataset_center_of_mass_ml.csv",
      "format": "CSV"
    },
    {
      "name": "05_evaluasi_linear_regression.png",
      "format": "PNG"
    },
    {
      "name": "06_distribusi_residual_error.png",
      "format": "PNG"
    },
    {
      "name": "08_dashboard_ringkasan_proyek.png",
      "format": "PNG"
    }
  ],
  "mlMethodologySpecs": {
    "learningParadigm": "Supervised Learning",
    "modelArchitecture": "Multi-Target Linear Regression",
    "coreFramework": "Scikit-Learn",
    "validationMethod": "Holdout Train-Test Split"
  }
}
```
