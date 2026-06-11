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

## 📚 Related Publications

This repository supports two research publications on retrieval-augmented LLMs for emergency medicine:

### Publication 1: Multi-Evidence Clinical Reasoning System (Published)

**Citation**: Wong HS, Wong TK. Multi-Evidence Clinical Reasoning With Retrieval-Augmented Generation for Emergency Triage: Retrospective Evaluation Study. *JMIR Med Inform* 2025;13:e82026. DOI: [10.2196/82026](https://doi.org/10.2196/82026)

**Focus**: Development and validation of the MECR-RAG system for emergency triage category assignment across all acuity levels (Categories 1-5).

**Key Findings**:
- MECR-RAG achieved QWK 0.902 vs expert nurse consensus (0.887)
- Multi-evidence retrieval (guidelines + past cases) outperformed single-evidence approaches
- DeepSeek-V3 showed strongest performance among tested LLMs
- System demonstrated non-inferiority to clinical experts

**Implementation**: Original system with string-based LLM outputs (v1.0)

### Publication 2: Category 3 Deterioration Detection 

**Authors**: Li CY, Wong TK, Wong HS

**Title**: Low-burden identification of severe early deterioration among Category 3 emergency department attendances using a retrieval-augmented large language model: a retrospective outcome-defined case-control study

**Focus**: Application of MECR-RAG for identifying severe early deterioration risk specifically in Category 3 (Urgent) ED attendances.

**Key Findings**:
- Retrieval-augmented model achieved **68.1% sensitivity** at **10% alert burden**
- Baseline LLM (no retrieval) achieved only 27.8% sensitivity at same burden
- Multi-evidence retrieval critical for detecting subtle deterioration indicators
- Demonstrates clinical utility for triaging-within-triage in high-volume urgent cases

**Implementation**: Enhanced system with structured Pydantic outputs (v2.0) enabling programmatic extraction of triage categories, confidence scores, and reasoning steps for case-control study analysis.

### Repository Version History

- **v1.0** (Publication 1): String-based LLM outputs with regex parsing for category extraction
- **v2.0** (Publication 2): Structured Pydantic outputs with automatic schema validation, enabling robust programmatic analysis of multi-step reasoning for deterioration detection research

The structured output architecture (v2.0) was developed to support the deterioration detection study's requirement for reliable, machine-readable extraction of clinical reasoning at each decision step.

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

- **DeepSeek V4 Pro**: Primary reasoning engine (migrated from V3; `deepseek-chat` deprecated by provider 2026-07-24)
- **Azure OpenAI GPT-4o**: Comparative analysis 
- **Anthropic Claude-3.7**: Alternative reasoning model
- **Azure Text-Embedding-3-Small**: Semantic similarity

---

## ⚠️ Repository Scope & Data Availability

### What's Included

✅ **Complete source code** - All preprocessing, indexing, retrieval, and generation modules
✅ **Executable templates** - Ready-to-run scripts and notebooks
✅ **Demo cases** - 5 synthetic/anonymized cases across all triage categories
✅ **Documentation** - Comprehensive setup and methodology guides
✅ **Structured output implementation** - v2.0 Pydantic schema architecture

### What Requires Reconstruction

Due to **institutional and regulatory restrictions**, the following are **not included**:

❌ **Hong Kong A&E Triage Guidelines (HKAETG)** - Source text redistribution not permitted
❌ **Processed guideline databases** - Derived from restricted source material
❌ **Medical case records** - Patient data excluded for privacy compliance
❌ **Vector embeddings** - Historical case embeddings excluded

### Reproducibility Statement

This repository provides **executable templates, transformation scripts, redacted examples, and documentation** sufficient to rerun the complete workflow on an appropriately governed local dataset. To reproduce the full MECR-RAG system:

1. Obtain source HKAETG text with institutional permission
2. Collect medical case data with ethics approval (IRB/CIRB)
3. Run preprocessing pipelines (`notebooks/01_Data_Preprocessing/`)
4. Build vector databases (`notebooks/02_Indexing/`)
5. Execute evaluation notebooks (`notebooks/03_Generation/`)

**What works immediately**: Baseline LLM evaluation with included demo cases
**What requires data**: All RAG variants (guideline-only, case-only, complete MECR-RAG)

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
- **Cross-LLM Integration**: Supports DeepSeek V4 Pro, GPT-4o, Claude-3.7 Sonnet
- **Privacy-Preserving Pipeline**: HIPAA-compliant processing for medical data

### Research Methodology (`notebooks/`)
Complete evaluation pipeline with structured output architecture (v2.0):

| Notebook | Purpose | Data Requirements |
|----------|---------|-------------------|
| `01_Data_Preprocessing/Preprocessing Pipeline for Traige Guideline and AE case notes.ipynb` | Data preprocessing pipeline | ⚠️ Requires source HKAETG and medical records |
| `02_Indexing/Indexing Pipeline.ipynb` | Vector database indexing methodology | ⚠️ Requires preprocessed data |
| `03_Generation/Baseline LLM.ipynb` | Baseline LLM evaluation | ✅ **Works with demo cases only** |
| `03_Generation/RAG (Guideline Only).ipynb` | Guideline-only ablation study | ⚠️ Requires guideline database |
| `03_Generation/RAG (Past Case Only).ipynb` | Case-only ablation study | ⚠️ Requires past case database |
| `03_Generation/RAG (Guideline + Past Case).ipynb` | Complete MECR-RAG evaluation (v2.0) | ⚠️ Requires both guideline & past case databases |

### Demo Interface (`scripts/`)
Simple interface for immediate testing and methodology validation:

- **Single Case Processing**: `run_single_case.py` for individual case assessment
- **Representative Cases**: 5 demo cases across all triage categories (1-5)
- **Clinical Data Format**: Demonstrates expected input structure
- **Multi-Model Support**: Test across different LLM architectures

### Clinical Data Documentation (`data/` & `db/`)

#### Excluded Due to Institutional and Regulatory Restrictions (❌):

Because of institutional and regulatory restrictions, the repository does not contain:
- **Raw electronic medical record data** - Past case databases
- **Full historical retrieval-corpus text** - Patient case vector embeddings
- **Full source HKAETG text** - Hong Kong Accident & Emergency Triage Guidelines (where redistribution is not permitted)
- **Processed guideline databases** - `db/triage_sections_with_summaries_*.json`

#### What's Included Instead (✅):
- **Executable templates** - Complete source code and processing scripts
- **Transformation scripts** - Data preprocessing and indexing pipelines
- **Redacted examples** - Demo cases with synthetic/anonymized data
- **Documentation** - Sufficient to rerun the workflow on an appropriately governed local dataset

---

## 🧪 Research Validation

### Available Testing (With Demo Cases Only)

The following functionality works with the included demo cases:

```bash
# Baseline LLM evaluation (no retrieval databases needed)
cd notebooks/03_Generation
jupyter notebook "Baseline LLM.ipynb"

# Demo case processing with structured output (v2.0)
cd ../..
python scripts/run_single_case.py scripts/demo_cases/case_1.json --model deepseek --verbose
```

### Complete System Testing (Requires Local Datasets)

Full MECR-RAG evaluation requires reconstructing the clinical knowledge bases on your local dataset with appropriate institutional approval:

```bash
# Step 1: Preprocess your institutional data (requires source HKAETG + medical records)
cd notebooks/01_Data_Preprocessing
jupyter notebook "Preprocessing Pipeline for Traige Guideline and AE case notes.ipynb"

# Step 2: Build vector databases
cd ../02_Indexing
jupyter notebook "Indexing Pipeline.ipynb"

# Step 3: Run ablation studies
cd ../03_Generation
jupyter notebook "RAG (Guideline Only).ipynb"          # Requires guideline database
jupyter notebook "RAG (Past Case Only).ipynb"          # Requires past case database
jupyter notebook "RAG (Guideline + Past Case).ipynb"  # Complete MECR-RAG (v2.0)

# Single case processing with full pipeline
cd ../../..
python scripts/run_single_case.py <your_case.json> --model deepseek --verbose
```

**Note**: All RAG variants require reconstructing the knowledge bases following our preprocessing methodology with your own appropriately governed medical data and institutional approval.

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

### Excluded Clinical Knowledge Bases

**Important**: Due to institutional and regulatory restrictions, the following are **NOT included** in this repository:

#### 1. **Hong Kong A&E Triage Guidelines (HKAETG)**
- **Missing**: `data/guidelines/` - Source HKAETG text and processed guideline databases
- **Reason**: Redistribution of HKAETG source material is not permitted
- **Impact**:
  - Cannot run guideline-based RAG evaluations without reconstruction
  - `notebooks/03_Generation/RAG (Guideline Only).ipynb` requires guideline database
  - `notebooks/03_Generation/RAG (Guideline + Past Case).ipynb` requires guideline database

#### 2. **Past Case Medical Databases**
- **Missing**: `db/past_case/` - Vector databases containing 3,000 medical cases per LLM
- **Reason**: Patient privacy and ethics requirements
- **Impact**:
  - `run_single_case.py` will show warnings about missing databases
  - `notebooks/03_Generation/RAG (Past Case Only).ipynb` requires past case database
  - `notebooks/03_Generation/RAG (Guideline + Past Case).ipynb` requires past case database
  - Full MECR-RAG pipeline needs user-provided medical case databases

### Functional Capabilities

#### ✅ **Available Without External Data**:
- Baseline LLM evaluation (no retrieval)
- Demo case structure validation
- System architecture exploration
- Source code and preprocessing pipelines

#### ⚠️ **Requires Guideline Database**:
- Guideline section retrieval
- Guideline-only RAG evaluation
- Condition-specific protocol application

#### ⚠️ **Requires Past Case Database**:
- Historical case similarity retrieval
- Past case-only RAG evaluation
- Real-world pattern recognition

#### ⚠️ **Requires Both Databases**:
- Complete 3-step MECR reasoning
- Full MECR-RAG evaluation
- Multi-evidence integration

### For Researchers

To implement the complete MECR-RAG system, you must reconstruct both knowledge bases:

1. **Obtain Source Materials**:
   - Acquire HKAETG text with institutional permission
   - Collect medical case data with appropriate IRB/ethics approval

2. **Follow Preprocessing Methodology**:
   - Process HKAETG using `notebooks/01_Data_Preprocessing/` (adapt to your guideline structure)
   - Process medical cases following our clinical data format (see demo cases)

3. **Build Vector Databases**:
   - Create guideline database using `notebooks/02_Indexing/`
   - Create past case database using `notebooks/02_Indexing/`

4. **Adapt to Your Guidelines**:
   - **CRITICAL**: Our system is designed for HKAETG structure
   - If using different triage guidelines, you must adapt the preprocessing and chunking strategy
   - Follow our methodology but adjust section boundaries and metadata extraction to match your guideline format

---
## 🔨 Reconstructing Clinical Knowledge Bases

For researchers wanting to implement the complete MECR-RAG system, you must reconstruct both the guideline and past case databases following our methodology.

### ⚠️ Important: Guideline Adaptation Required

**Our system is designed specifically for Hong Kong Accident & Emergency Triage Guidelines (HKAETG) structure**. If you are using different triage guidelines (e.g., ESI, CTAS, MTS, ATS), you **must adapt**:

1. **Preprocessing Strategy**: Modify chunking logic in `notebooks/01_Data_Preprocessing/` to match your guideline structure
2. **Section Boundaries**: Adjust how sections are identified and extracted
3. **Metadata Extraction**: Adapt clinical indicator and triage criteria parsing
4. **Retrieval Logic**: May need modification if your guidelines are not organized by clinical presentation

**Key HKAETG Characteristics** (for comparison):
- Organized by clinical presentations (e.g., "Chest Pain", "Shortness of Breath", "Abdominal Pain")
- Each section contains condition-specific triage criteria
- Sections include discriminators for category assignment
- Metadata includes clinical indicators and assessment criteria

If your guidelines have a different structure (e.g., discriminator-based like MTS, algorithm-based like ESI), you will need to modify the preprocessing and retrieval components accordingly.

### Part A: Setting Up Guideline Database

#### Step 1: Obtain Source Guidelines

1. **Acquire triage guidelines** appropriate for your institution:
   - Hong Kong A&E Triage Guidelines (HKAETG) - requires institutional permission
   - Alternative: Your institution's emergency triage protocols
   - Alternative: Publicly available triage guidelines (e.g., ESI, CTAS, MTS)

2. **Understand guideline structure**:
   - Our system is designed for HKAETG's condition-specific sections
   - Each section covers a clinical presentation (e.g., "Chest Pain", "Shortness of Breath")
   - Sections contain triage criteria and category recommendations

#### Step 2: Preprocess Guidelines

1. **Adapt preprocessing to your guideline format**:
   ```bash
   cd notebooks/01_Data_Preprocessing
   jupyter notebook "Preprocessing Pipeline for Traige Guideline and AE case notes.ipynb"
   ```

2. **Critical adaptations needed**:
   - **Section Chunking**: Adjust chunking strategy to match your guideline structure
   - **Metadata Extraction**: Extract section titles, summaries, and clinical indicators
   - **Format Conversion**: Convert to our expected JSON format

3. **Expected output structure**:
   ```
   data/
   └── guidelines/
       └── processed_guidelines.json    # Your processed guideline sections
   ```

4. **JSON format per section**:
   ```json
   {
     "section_title": "Chest Pain",
     "summary": "Assessment criteria for chest pain presentations",
     "content": "Full guideline text...",
     "clinical_indicators": ["chest pain", "cardiac", "MI"],
     "triage_criteria": {
       "Category 1": "...",
       "Category 2": "..."
     }
   }
   ```

#### Step 3: Create Guideline Vector Database

1. **Run indexing for guidelines**:
   ```bash
   cd notebooks/02_Indexing
   jupyter notebook "Indexing Pipeline.ipynb"
   ```

2. **The notebook will create**:
   ```
   db/
   ├── triage_sections_with_summaries_deepseek-v4-pro.json
   ├── triage_sections_with_summaries_gpt-4o.json
   └── triage_sections_with_summaries_claude-3-7.json
   ```

### Part B: Setting Up Past Case Database

For researchers wanting to implement past case retrieval, follow these steps to create your own medical case database:

#### Step 1: Prepare Medical Case Data

**Option A: If You Have Hong Kong AED PDF Medical Records**
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
3. **Run preprocessing pipeline** using `notebooks/01_Data_Preprocessing/Preprocessing Pipeline for Traige Guideline and AE case notes.ipynb`

**Option B: If You Have Structured Medical Data (Recommended)**
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

#### Step 2: Create Past Case Vector Database

Once you have your JSON case files ready:

1. **Run the indexing notebook:**
   ```bash
   cd notebooks/02_Indexing
   jupyter notebook "Indexing Pipeline.ipynb"
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

#### Step 3: Test Complete System

After setting up both guideline and past case databases:

```bash
# Test complete MECR-RAG pipeline
python scripts/run_single_case.py scripts/demo_cases/case_1.json --model deepseek

# Run evaluation notebooks
cd notebooks/03_Generation
jupyter notebook "RAG (Guideline Only).ipynb"          # Guideline-only ablation
jupyter notebook "RAG (Past Case Only).ipynb"          # Past case-only ablation
jupyter notebook "RAG (Guideline + Past Case).ipynb"  # Complete MECR-RAG (v2.0)
```

### Data Requirements

#### For Guideline Database:
- **Source Material**: Triage guidelines structured by clinical presentation/condition
- **Minimum Sections**: 15+ guideline sections recommended for comprehensive coverage
- **Structure Adaptation**: If not using HKAETG, adapt preprocessing to match your guideline format
- **Critical Note**: Our retrieval assumes condition-based sections; different structures require code modification

#### For Past Case Database:
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

## 🔄 System Updates: Structured Output Architecture (v2.0)

### Motivation for Update

Following the publication of our initial work, we have enhanced the MECR-RAG system with **structured output architecture** to improve reliability, reproducibility, and machine-readability of clinical reasoning outputs. This update addresses key challenges identified during deployment and peer review.

### Technical Enhancements

#### 1. **Pydantic Schema-Based Outputs**

**Previous Implementation (v1.0)**:
- LLM outputs were free-form markdown text
- Manual regex-based parsing to extract triage categories
- Prone to parsing failures and format inconsistencies
- Difficult to programmatically access reasoning steps

**Current Implementation (v2.0)**:
- Native Pydantic schema validation using `.with_structured_output()`
- Automatic type checking and field validation
- Direct JSON-serializable outputs
- Machine-readable reasoning at each decision step

**Schema Definitions**:
```python
class GuidelineRetrievalOutput(BaseModel):
    """Structured guideline section selection"""
    selected_sections: List[str]  # Max 2 sections
    reasoning: str                # Explanation for selection

class TriageStep(BaseModel):
    """Individual reasoning step"""
    category: Literal["1", "2", "3", "4", "5", "Not specified"]
    confidence: Literal["High", "Medium", "Low"]
    reason: str

class TriagePredictionOutput(BaseModel):
    """Complete 3-step triage assessment"""
    step1_clinical_risk: TriageStep
    step2_guidelines: TriageStep
    step3_realworld_factors: TriageStep
    final_decision: TriageStep
```

#### 2. **Enhanced LLM Integration**

**DeepSeek V4 Pro Support**:
- Migrated to `langchain-deepseek` native client (`langchain_deepseek.ChatDeepSeek`)
- Replaced deprecated `ChatOpenAI` wrapper (`deepseek-chat` deprecated by provider 2026-07-24)
- Optimized for structured output generation with `deepseek-v4-pro` model

**Multi-Model Compatibility**:
- All three LLMs (DeepSeek V4 Pro, GPT-4o, Claude-3.7) now support structured outputs
- Consistent schema validation across different model architectures
- Unified error handling and retry logic

#### 3. **Improved Reliability Features**

**Automatic Retry Logic**:
```python
def execute_structured_node_with_retry(node_func, state, node_name, max_retries=3):
    """Retry failed structured outputs with validation"""
    for attempt in range(max_retries):
        try:
            return node_func(state)  # Pydantic validates automatically
        except ValidationError:
            # Retry on schema violation
            continue
    return failed_state  # Mark processing failure
```

**Error Tracking**:
- Failed processing marked in state with `processing_failed=True`
- Detailed error messages captured in `failure_reason`
- Node-level failure tracking for debugging

#### 4. **Backwards Compatibility**

The system maintains full backwards compatibility with v1.0 outputs:

```python
def extract_final_category_from_assessment(assessment: Union[str, Dict[str, Any]]):
    """Handles both structured dict (v2.0) and string (v1.0) formats"""
    if isinstance(assessment, dict):
        # v2.0: Direct access to structured output
        return assessment["final_decision"]["category"]
    else:
        # v1.0: Fallback to regex parsing
        return parse_category_from_text(assessment)
```

### Implementation Changes

#### Updated Components

| Component | Change | Impact |
|-----------|--------|--------|
| **Guideline Retrieval** | `GuidelineRetrievalOutput` schema | Now captures reasoning for section selection |
| **Specialty Prediction** | Dynamic schema generation | Validates specialty names at runtime |
| **Final Triage Assessment** | `TriagePredictionOutput` schema | Structured 4-step reasoning with confidence |
| **State Management** | Added fields: `guideline_reasoning`, `final_category`, `final_confidence` | Enhanced output analysis |

#### New Files

- `src/models/schemas.py`: Pydantic schema definitions
- `src/utils/structured_output.py`: Retry logic and token tracking
- `IMPLEMENTATION_GUIDE.md`: Technical migration documentation
- `CHANGES_SUMMARY.md`: Detailed changelog for v2.0

#### Modified Files

- `src/models/llm_factory.py`: DeepSeek native client support
- `src/data/state.py`: Extended state with structured output fields
- `src/retrieval/guideline_retriever.py`: Structured guideline selection
- `src/retrieval/specialty_predictor.py`: Dynamic specialty schema
- `src/generation/triage_assessor.py`: Structured 3-step reasoning
- `scripts/run_single_case.py`: Enhanced output display

### Benefits for Research

#### 1. **Reproducibility**
- Deterministic output structure across runs
- No parsing ambiguity or format variations
- Consistent schema validation

#### 2. **Interpretability**
- Direct access to confidence scores for each reasoning step
- Explicit reasoning captured for each decision
- Machine-readable audit trail

#### 3. **Extensibility**
- Easy to add new reasoning steps or fields
- Schema evolution supported via Pydantic versioning
- Compatible with automated evaluation pipelines

#### 4. **Clinical Deployment**
- Structured outputs integrate with electronic health records (EHR)
- Real-time monitoring of confidence levels
- Automated quality assurance via schema validation

### Migration Guide

For researchers using v1.0, the system automatically handles both formats:

```python
# v1.0 output (string)
assessment = "### STEP 1: CLINICAL RISK ASSESSMENT\n..."
category = extract_category_from_text(assessment)  # Regex parsing

# v2.0 output (structured dict)
assessment = {
    "step1_clinical_risk": {"category": "2", "confidence": "High", ...},
    "final_decision": {"category": "2", "confidence": "High", ...}
}
category = assessment["final_decision"]["category"]  # Direct access
```

**No changes required** to existing evaluation scripts - the `extract_final_category_from_assessment()` function handles both formats transparently.

### Validation

The structured output update has been validated on:
- ✅ All 236 test cases from the original study
- ✅ Cross-LLM consistency (DeepSeek V4 Pro, GPT-4o, Claude-3.7)
- ✅ Backwards compatibility with v1.0 outputs
- ✅ Retry logic under simulated failures

### Documentation

Complete technical documentation available in:
- `IMPLEMENTATION_GUIDE.md` - Detailed migration guide
- `CHANGES_SUMMARY.md` - Comprehensive changelog
- `notebooks/03_Generation/` - Evaluation notebooks with structured outputs

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
│   ├── 01_Data_Preprocessing/
│   │   └── Preprocessing Pipeline for Traige Guideline and AE case notes.ipynb
│   ├── 02_Indexing/
│   │   └── Indexing Pipeline.ipynb
│   └── 03_Generation/
│       ├── Baseline LLM.ipynb                      # ✅ Works with demo cases
│       ├── RAG (Guideline Only).ipynb             # ⚠️ Requires guideline DB
│       ├── RAG (Past Case Only).ipynb             # ⚠️ Requires past case DB
│       └── RAG (Guideline + Past Case).ipynb      # ⚠️ Requires both DBs (v2.0)
├── scripts/                     # Research interface and utilities
│   ├── run_single_case.py       # Single case processing script
│   └── demo_cases/              # ✅ Synthetic/anonymized representative cases
│       ├── case_1.json          # Category 1 (Critical)
│       ├── case_2.json          # Category 2 (Emergency)
│       ├── case_3.json          # Category 3 (Urgent)
│       ├── case_4.json          # Category 4 (Semi-urgent)
│       └── case_5.json          # Category 5 (Non-urgent)
├── data/                        # Clinical knowledge base (empty - see note below)
│   ├── guidelines/              # ❌ HKAETG text not included (redistribution restricted)
│   ├── past_case/               # ❌ Medical records not included (privacy)
│   └── test_case/               # ❌ Test cases not included (privacy)
└── db/                          # Processed knowledge databases (empty - see note below)
    ├── triage_sections_with_summaries_*.json  # ❌ Guideline DBs not included
    └── past_case/                              # ❌ Case embeddings not included
```

**Important Note**: Due to institutional and regulatory restrictions, clinical knowledge bases (`data/guidelines/`, `data/past_case/`, `data/test_case/`) and processed databases (`db/`) are **not included** in this repository. To reproduce the full system, you must:

1. **Obtain source data** with appropriate institutional approval and ethics clearance
2. **Run preprocessing pipelines** (`01_Data_Preprocessing/`) on your local dataset
3. **Build vector databases** (`02_Indexing/`) following our methodology
4. **Execute evaluation notebooks** (`03_Generation/`) with your reconstructed knowledge bases

The repository provides executable templates, transformation scripts, and documentation sufficient to rerun the complete workflow on an appropriately governed local dataset.
