# MECR-RAG: Multi-Evidence Clinical Reasoning for Emergency Triage

A comprehensive Retrieval-Augmented Generation system for emergency medicine triage decision support, implementing multi-evidence reasoning with clinical guidelines and historical case analysis.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Ethics Approved](https://img.shields.io/badge/Ethics-CIRB%202024--561--4-green.svg)]()

---

## 🎯 Research Overview

MECR-RAG addresses the critical challenge of emergency department triage by combining structured clinical guidelines with historical case patterns. This repository contains the complete implementation and evaluation methodology supporting our journal submission.

The system implements a novel **3-step reasoning framework** that integrates:

1. **Clinical Risk Assessment** - Evidence-based urgency evaluation
2. **Guideline-Based Assessment** - Condition-specific protocol application  
3. **Real-World Factors Analysis** - Historical case pattern recognition

### Key Research Contributions

- **Multi-Evidence Architecture**: First system combining clinical guidelines + historical cases for emergency triage
- **3-Step Clinical Reasoning**: Structured decision process mimicking clinical workflow
- **Cross-LLM Evaluation**: Validated on DeepSeek-V3, GPT-4o, and Claude-3.7 Sonnet
- **Clinical Validation**: Non-inferiority to expert nurses (QWK 0.902 vs 0.887)
- **Comprehensive Evaluation**: Component-wise ablation studies and scaling analysis

---

## 🏗️ System Architecture

### Core Pipeline

```
Input Case → Preprocessing → Guideline Retrieval → Specialty Prediction → Case Retrieval → Final Assessment
```

### Multi-Evidence Sources

| Evidence Type | Source | Purpose |
|---------------|---------|---------|
| **Clinical Guidelines** | Hong Kong Accident & Emergency Triage Guidelines | Condition-specific criteria |
| **Historical Cases** | 3,000 anonymized cases per LLM | Real-world decision patterns |
| **Specialty Mapping** | 13 standardized specialties | Targeted case filtering |

### LLM Support

- **DeepSeek-V3**: Primary reasoning engine
- **Azure OpenAI GPT-4o**: Comparative analysis 
- **Anthropic Claude-3.7**: Alternative reasoning model
- **Azure Text-Embedding-3-Small**: Semantic similarity

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- API access to supported LLM providers
- 4GB+ RAM

### Installation

1. **Clone & Setup**
   ```bash
   git clone <repository>
   cd MECR-RAG-Clinical-Research
   pip install -r requirements.txt
   ```

2. **Configure Environment**
   ```bash
   cp .env.template .env
   # Edit .env with your API keys:
   # DEEPSEEK_API_KEY=your_deepseek_key
   # AZURE_OPENAI_API_KEY=your_azure_key
   # AZURE_OPENAI_ENDPOINT=your_endpoint
   # ANTHROPIC_API_KEY=your_claude_key
   ```

### Test with Demo Cases

**Important Limitation**: Past case related data/databases are **NOT included** due to ethics requirements. Our `src/` and `scripts/run_single_case.py` are designed for **COMPLETE MECR pipeline execution** (both past case database + guideline). Users will not be able to launch the full system without a compatible past case database. However, ablation configurations can still be tested and accessed through individual notebooks: `03_generation_eval_basic_prompt.ipynb`, `04_generation_eval_guideline_RAG.ipynb` (excluding `05_generation_eval_past_case_RAG.ipynb` + `06_generation_eval_complete_RAG.ipynb`, since past case database is needed).

We provide five representative emergency department cases spanning all triage categories:

```bash
# Test Critical case (Category 1) - Will show database warnings
python scripts/run_single_case.py scripts/demo_cases/case_1.json --model deepseek --output case_1_prediction.json

# Test Emergency case (Category 2) - Will show database warnings  
python scripts/run_single_case.py scripts/demo_cases/case_2.json --model deepseek --output case_1_prediction.json

# Test with different models - All will show database warnings
python scripts/run_single_case.py scripts/demo_cases/case_3.json --model gpt4o
python scripts/run_single_case.py scripts/demo_cases/case_4.json --model claude
```

### Expected Output

```
{
  "case_file": "scripts/demo_cases/case_1.json",
  "processing_parameters": {
    "model": "deepseek",
    "k_cases": 5,
    "similarity_threshold": 0.7,
    "metadata_threshold": 1.0
  },
  "results": {
    "clinical_summary": "\"42-year-old male involved in high-speed motor vehicle accident, presenting unconscious at scene with initial GCS 3, improved to GCS 8/15 with respiratory distress and multiple abrasions. Critically abnormal findings include hypotension (85/45 mmHg), tachycardia (HR 125), ...",
    "selected_guidelines": [
      "Conscious Level",
      "Respiratory Rate and Signs"
    ],
    "guidelines_count": 2,
    "past_cases_retrieved": 5,
    "final_category": 1,
    "confidence": "high",
    "full_assessment": "... ### FINAL DECISION  \n**Step 1 Category (Clinical Risk):** 1  \n**Step 2 Category (Guidelines):** 1  \n**Step 3 Adjustment (Real-World Factors):** No change (past cases support Category 1).  \n**FINAL TRIAGE CATEGORY:** 1  \n**CONFIDENCE:** High  \n**KEY RATIONALE:**  ... "
  },
  "extracted_info": {
    "category": 1,
    "confidence": "high",
    "rationale": "- **Clinical risk and guidelines unequivocally indicate Category 1** (GCS ≤8, hypotension, hypoxia).  \n- **Past cases validate this**; even less severe cases (e.g., Case 5) were Category 1 due to trauma severity.  \n- No real-world factors (e.g., age, comorbidities) justify downgrading. Immediate resuscitation is mandatory."
  }
}
```

---

## 📊 Repository Contents

### Core System (`src/`)
Multi-evidence retrieval architecture implementing the complete MECR-RAG methodology:

- **Multi-Evidence Retrieval**: Combines clinical guidelines + historical case patterns
- **3-Step Clinical Reasoning**: Structured decision framework mimicking clinical workflow  
- **Cross-LLM Integration**: Supports DeepSeek-V3, GPT-4o, Claude-3.7 Sonnet
- **Privacy-Preserving Pipeline**: HIPAA-compliant processing for medical data

### Research Methodology (`notebooks/`)
Complete evaluation pipeline from our journal submission:

| Notebook | Purpose | Availability |
|----------|---------|--------------|
| `01_data_preprocessing.ipynb` | Data preprocessing | ✅ Full access |
| `02_indexing.ipynb` | Database indexing methodology | ✅ Full access |
| `03_generation_eval_basic_prompt.ipynb` | Baseline LLM evaluation | ✅ **Works without past case DB** |
| `04_generation_eval_guideline_RAG.ipynb` | Guideline-only ablation study | ✅ **Works without past case DB** |
| `05_generation_eval_past_case_RAG.ipynb` | Case-only ablation study | ⚠️ Requires past case database |
| `06_generation_eval_complete_RAG.ipynb` | Complete MECR evaluation | ⚠️ Requires past case database |

### Demo Interface (`scripts/`)
Simple interface for immediate testing and methodology validation:

- **Single Case Processing**: `run_single_case.py` for individual case assessment
- **Representative Cases**: 5 demo cases across all triage categories (1-5)
- **Clinical Data Format**: Demonstrates expected input structure
- **Multi-Model Support**: Test across different LLM architectures

### Clinical Data Documentation (`data/` & `db/`)

#### Included (✅):
- `data/guidelines/` - Hong Kong Accident & Emergency Triage Guidelines (processed)
- `db/triage_sections_with_summaries_*.json` - Processed guideline databases for all LLMs

#### Excluded for Privacy (❌):
- Past case medical databases (due to ethics requirements)
- Patient case vector embeddings
- Historical triage decision data

---

## 🧪 Research Validation

### Functional Testing (Available)

Test ablation configurations that work without past case databases:

```bash
# Baseline LLM evaluation (Notebook 03)
cd notebooks
jupyter notebook 03_generation_eval_basic_prompt.ipynb

# Guideline-only RAG evaluation (Notebook 04)  
jupyter notebook 04_generation_eval_guideline_RAG.ipynb

# Demo case processing
python scripts/run_single_case.py scripts/demo_cases/case_1.json --model deepseek --verbose
```

### Complete System Testing (Requires Past Case Database)

Full MECR-RAG evaluation requires medical case databases:

```bash
# These require past case databases (not included):
jupyter notebook 05_generation_eval_past_case_RAG.ipynb      # Past case ablation
jupyter notebook 06_generation_eval_complete_RAG.ipynb      # Complete system

# Single case processing with full pipeline
python scripts/run_single_case.py scripts/demo_cases/case_1.json --model deepseek  # Will show database missing warnings
```

---

## 📋 Clinical Data Structure

Our system processes emergency department cases following this structure:

### Case Format

```json
{
  "Demographics": {
    "Sex": "M",
    "Age": "42 years",
    "Ambulatory status": "Stretcher",
    "Informant": "Paramedic",
    "Allergies": "(1)No Known Drug Allergy",
    "Past health": "GPH"
  },
  "Clinical Presentation": {
    "Chief complaint": "MVA high speed impact\nunconsious at scene\nGCS 3 initially",
    "Condition on Arrival": "GCS 8/15\nrespiratory distress\nmultiple abrasions"
  },
  "Vitals and Observations": {
    "Vital Signs": {
      "GCS": "E: 2, V: 2, M: 4, Score: 8/15",
      "BP": "1st: 85/45 mmHg",
      "PR/AR": "125",
      "SpO2": "88% on 15L O2, bag mask ventilation"
    }
  },
  "Case Disposition": {
    "Triage Category": "1",
    "Attending Specialty": "Neurosurgery"
  }
}
```

### Demo Cases Overview

| Case | Category | Clinical Scenario | Key Features |
|------|----------|------------------|--------------|
| `case_1.json` | Critical (1) | MVA with traumatic brain injury | GCS 8, hypotension, respiratory distress |
| `case_2.json` | Emergency (2) | Acute MI presentation | Chest pain, cardiac risk factors |
| `case_3.json` | Urgent (3) | Abdominal pain, ?biliary colic | Surgical Condition, stable vitals |
| `case_4.json` | Semi-urgent (4) | Viral pharyngitis | Stable condition |
| `case_5.json` | Non-urgent (5) | Minor complaint | Low acuity, routine assessment |

---

## ⚠️ System Limitations & Data Privacy

### Past Case Database Exclusion

**Important**: Past case medical databases are **NOT included** in this repository due to ethics requirements and privacy protection:

- **Missing Components**: `db/past_case/` directories containing 3,000 medical cases per LLM
- **Impact on Functionality**: 
  - `run_single_case.py` will show warnings about missing databases
  - Notebooks 05-06 require past case databases to execute
  - Full MECR-RAG pipeline needs user-provided medical case databases
- **Alternative Access**: Notebooks 03-04 demonstrate methodology without requiring past case data

### Functional Capabilities

#### ✅ **Available Without Past Case Database**:
- Clinical guideline retrieval and processing
- Baseline LLM evaluation (Notebook 03)
- Guideline-only RAG evaluation (Notebook 04)  
- Demo case structure validation
- System architecture exploration

#### ⚠️ **Requires Past Case Database**:
- Complete 3-step MECR reasoning
- Past case similarity retrieval
- Full system evaluation (Notebooks 05-06)

### For Researchers

To implement the complete MECR-RAG system:

1. **Prepare Medical Case Database**: Create past case databases following our preprocessing methodology
2. **Structure Requirements**: Cases must follow our clinical data format (see demo cases)
3. **Privacy Compliance**: Ensure appropriate IRB approval for medical data use
4. **Database Integration**: Follow our indexing methodology (Notebook 02) for vector database creation

---
## 🔨 Setting Up Your Own Past Case Database

For researchers wanting to implement the complete MECR-RAG system with past case retrieval, follow these steps to create your own medical case database:

### Step 1: Prepare Medical Case Data

#### Option A: If You Have Hong Kong AED PDF Medical Records
If you have Hong Kong AED PDF medical records, follow the complete preprocessing pipeline:

1. **Create the required folder structure:**
   ```
   data/
   └── past_case/
       ├── pdf/
       │   ├── Cat 1/        # Critical cases
       │   ├── Cat 2/        # Emergency cases  
       │   ├── Cat 3/        # Urgent cases
       │   ├── Cat 4/        # Semi-urgent cases
       │   └── Cat 5/        # Non-urgent cases
       ├── txt/              # Will be generated by preprocessing
       ├── csv/              # Will be generated by preprocessing
       └── json/             # Will be generated by preprocessing
   ```

2. **Place your PDF files** in the appropriate triage category folders
3. **Run preprocessing pipeline** using `01_data_preprocessing.ipynb`

#### Option B: If You Have Structured Medical Data (Recommended)
If you already have structured medical case data, skip directly to JSON format:

1. **Create the simplified folder structure:**
   ```
   data/
   └── past_case/
       └── json/
           ├── Cat 1/        # Critical cases (.json files)
           ├── Cat 2/        # Emergency cases (.json files)
           ├── Cat 3/        # Urgent cases (.json files)
           ├── Cat 4/        # Semi-urgent cases (.json files)
           └── Cat 5/        # Non-urgent cases (.json files)
   ```

2. **Format your cases** following our clinical data structure (see demo cases for exact format)
3. **Save each case** as a separate `.json` file in the appropriate category folder

### Step 2: Create Vector Database

Once you have your JSON case files ready:

1. **Run the indexing notebook:**
   ```bash
   cd notebooks
   jupyter notebook 02_indexing.ipynb
   ```

2. **The notebook will create:**
   ```
   db/
   └── past_case/
       ├── db_{model}_1000case/     # 1K case database
       ├── db_{model}_2000case/     # 2K case database  
       └── db_{model}_3000case/     # 3K case database
           ├── summary_vectordb/    # Vector embeddings with metadata
           ├── json_store/           # Processed cases json files
   ```

### Step 3: Test Complete System

After setting up your past case database:

```bash
# Test complete MECR-RAG pipeline
python scripts/run_single_case.py scripts/demo_cases/case_1.json --model deepseek

# Run complete evaluation notebooks
jupyter notebook 05_generation_eval_past_case_RAG.ipynb     # Past case ablation
jupyter notebook 06_generation_eval_complete_RAG.ipynb     # Complete MECR evaluation
```

### Data Requirements

- **Minimum Cases**: 1,000+ cases recommended for meaningful retrieval
- **Balanced Distribution**: Include cases across all 5 triage categories
- **Clinical Completeness**: Each case should include demographics, clinical presentation, vitals
- **Anonymization**: Ensure all patient identifiers are removed
- **IRB Approval**: Obtain appropriate ethics approval for medical data use

### JSON Case Format

Each case file should follow the structure (matching our demo cases), documented in **Case Format** under **Clinical Data Structure** above.

---
## 🔧 Environment Configuration

### Required API Keys

Configure these in your `.env` file:

```bash
# Primary LLM (Required for any model)
DEEPSEEK_API_KEY=your_deepseek_key

# Azure OpenAI (Required for embeddings + GPT-4o)
AZURE_OPENAI_API_KEY=your_azure_key
AZURE_OPENAI_ENDPOINT=your_azure_endpoint
AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT_NAME=text-embedding-3-small
AZURE_OPENAI_LLM_DEPLOYMENT_NAME=gpt-4o

# Anthropic (Required for Claude)
ANTHROPIC_API_KEY=your_claude_key
```

### System Requirements

```
Python: 3.11+
Memory: 4GB+ RAM
Storage: 2GB for guidelines and demo data
Network: Internet access for LLM APIs
```

---

## 📖 Research Methodology

### Experimental Design

Our study implements a comparative evaluation across three LLM architectures:

- **Dataset**: 236 consensus-labeled emergency department cases
- **Retrieval Database**: 3,000 anonymized cases per LLM model
- **Primary Metric**: Quadratic weighted kappa (QWK) agreement with expert nurses
- **Statistical Analysis**: Bootstrap confidence intervals, McNemar's tests, ablation studies
- **Cross-LLM Validation**: DeepSeek-V3, GPT-4o, Claude-3.7 Sonnet

### Key Findings

- **Expert-Level Performance**: QWK 0.902 vs expert agreement 0.887
- **Significant Improvement**: +0.101 QWK over baseline LLM
- **Reduced Overtriage**: 12.7% vs 28.8% for baseline LLM  
- **Cross-LLM Generalization**: Consistent improvements across all models
- **Component Validation**: Both guideline and case retrieval contribute meaningfully

---

## 🔒 Privacy & Ethics

### Data Protection

- **Ethics Approval**: CIRB-2024-561-4 approved research protocol
- **Privacy Compliance**: All medical data excluded from repository
- **Anonymization**: Demo cases are fully synthetic/anonymized
- **Academic Use**: Repository designed for peer review and research validation

### Code Availability

The code supporting the findings of this study is available from the corresponding author or via this private GitHub repository. For peer review purposes, access can be granted upon request.

---

## 📚 Citation

If you use this code in your research, please cite our journal submission:

```
Wong, HS and Wong, TK. "Comparative Evaluation of Multi-Evidence Clinical 
Reasoning RAG for Emergency Triage Accuracy and Expert Consensus Agreement." 
Submitted for journal review, 2025.
```

---

## 📁 Repository Structure

```
MECR-RAG-Clinical-Research/
├── README.md                    # This documentation
├── LICENSE                      # MIT License for code sharing
├── requirements.txt             # Python dependencies with versions
├── .env.template                # Environment configuration template
├── .gitignore                   # Privacy protection and cleanup
├── src/                         # Core MECR-RAG implementation
│   ├── config/                  # System configuration
│   ├── models/                  # LLM factory and embeddings  
│   ├── retrieval/               # Multi-evidence retrieval systems
│   ├── generation/              # 3-step triage reasoning
│   ├── pipeline/                # LangGraph orchestration
│   ├── preprocessing/           # Clinical data processing
│   ├── data/                    # Data management utilities
│   └── utils/                   # Helper functions
├── notebooks/                   # Research methodology and evaluation
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_indexing.ipynb
│   ├── 03_generation_eval_basic_prompt.ipynb      # ✅ Works without past case DB
│   ├── 04_generation_eval_guideline_RAG.ipynb    # ✅ Works without past case DB  
│   ├── 05_generation_eval_past_case_RAG.ipynb    # ⚠️ Requires past case DB
│   └── 06_generation_eval_complete_RAG.ipynb     # ⚠️ Requires past case DB
├── scripts/                     # Research interface and utilities
│   ├── run_single_case.py       # Single case processing script
│   └── demo_cases/              # Representative clinical cases
│       ├── case_1.json          # Category 1 (Critical)
│       ├── case_2.json          # Category 2 (Emergency)
│       ├── case_3.json          # Category 3 (Urgent)
│       ├── case_4.json          # Category 4 (Semi-urgent)
│       └── case_5.json          # Category 5 (Non-urgent)
├── data/                        # Clinical knowledge base
│   └── guidelines/              # ✅ Hong Kong Accident & Emergency Triage Guidelines
└── db/                          # Processed knowledge databases
    ├── triage_sections_with_summaries_deepseekv3.json    # ✅ Included
    ├── triage_sections_with_summaries_gpt-4o.json       # ✅ Included
    └── triage_sections_with_summaries_claude-3-7.json   # ✅ Included
```

**Note**: Medical case data (`data/past_case`, `data/test_case`) and databases (`db/past_case/`) are excluded due to privacy requirements but can be reconstructed following our preprocessing methodology with appropriate medical data and IRB approval.
