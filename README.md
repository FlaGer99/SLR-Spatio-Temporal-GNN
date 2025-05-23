<div align="center">

  <h2><b>
  A Systematic Literature Review of Spatio-Temporal Graph Neural Network Models for Time Series Forecasting and Classification
  </b></h2>

  <p><i> by Flavio Corradini, Flavio Gerosa, Marco Gori, Carlo Lucheroni, Marco Piangerelli, Martina Zannotti <br>
  Corresponding author: Martina Zannotti </i> <a href="mailto:martina.zannotti@example.com">✉️</a></p>

</div>

<div align="center">

**<a href="https://doi.org/10.48550/arXiv.2410.22377">[Pre-print (second version)]</a>**

</div>

---

Our review examines the available literature on the use of spatio-temporal GNNs for time series classification and forecasting. It synthesizes insights from the fragmented literature to support researchers, presenting comprehensive tables of model outcomes and benchmarks. To the best of our knowledge, this is the first systematic literature review to provide such a detailed compilation.

<!--- (COMMENTATO)
If you find this project interesting, please refer to our paper and cite it in your work:
```
Bibtex reference to the paper
```
-->

## 🗓️ History of the project
<!--- (COMMENTATO)
- [2025-05-28] Last version with all conference papers has been submitted to _Neural Networks_
-->
- [2025-03-10] Second version with all papers from 2024 has been completed <a href="https://doi.org/10.48550/arXiv.2410.22377">[link]</a>
- [2024-10-29] The first version of the pre-print has been published on ArXiv

## 🗂️ Collection of papers
A total of 2663 records were identified using the search query:
```(("graph neural network" OR gnn) AND "time series") AND (classification OR forecasting)```
- 766 from Scopus;
- 258 from IEEE Xplore;
- 292 from Web of Science;
- 1347 from ACM.

After removing 1617 duplicates and papers from journals outside the Q1–Q2 rankings or conferences below the A++ rating, 1046 records were manually evaluated. Among these
- 666 were discarded because they were considered outside the scope
- 12 were discarded because they were not in English.

As a result, **368 records** are included in the review, comprising **263 papers from Q1–Q2 journals** and **105 papers from A\* and A conferences**.

## 📑 Groups
The selected journal papers are divided in groups according to the application domain. The groups of papers are presented in alphabetical order, with the exception of the "Generic" applications group (which cannot be directly attributed to a specific case study) and the "Other topics" group (focusing on specific problems of other disciplines):  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;• [⚡ Energy](#energy)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;• [🌍 Environment](#environment)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;• [📈 Finance](#finance)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;• [🧑‍⚕️ Health](#health)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;• [🚗 Mobility](#mobility)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;• [🔍 Predictive monitoring](#predictive-monitoring)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;• [🌐 Generic](#generic)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;• [📚 Other topics](#other-topics)  

Here is a list of all the papers categorized into each group.

<div id='energy'/>

### ⚡ Energy
- A multivariate time series graph neural network for district heat load forecasting (Energy, 2023) <a href="https://doi.org/10.1016/j.energy.2023.127911">[link]</a>
- TransformGraph: A novel short-term electricity net load forecasting model (Energy Reports, 2023) <a href="https://doi.org/10.1016/j.egyr.2023.01.050">[link]</a>
- A short-term residential load forecasting scheme based on the multiple correlation-temporal graph neural networks[Formula presented] (Applied Soft Computing, 2023) <a href="https://doi.org/10.1016/j.asoc.2023.110629">[link]</a>
- Spatial-temporal learning structure for short-term load forecasting (IET Generation, Transmission and Distribution, 2023) <a href="http://dx.doi.org/10.1049/gtd2.12684">[link]</a>
- Multiplex parallel GAT-ALSTM: A novel spatial-temporal learning model for multi-sites wind power collaborative forecasting (Frontiers in Energy Research, 2022) <a href="https://doi.org/10.3389/fenrg.2022.974682">[link]</a>
- Spatio-Temporal Graph Neural Networks for Multi-Site PV Power Forecasting (IEEE Transactions on Sustainable Energy, 2022) <a href="https://doi.org/10.1109/TSTE.2021.3125200">[link]</a>
- Interpretable temporal-spatial graph attention network for multi-site PV power forecasting (Applied Energy, 2022) <a href="https://doi.org/10.1016/j.apenergy.2022.120127">[link]</a>
- Structure-Informed Graph Learning of Networked Dependencies for Online Prediction of Power System Transient Dynamics (IEEE Transactions on Power Systems, 2022) <a href="https://doi.org/10.1109/TPWRS.2022.3153328">[link]</a>
- Superposition Graph Neural Network for offshore wind power prediction (Future Generation Computer Systems, 2020) <a href="https://doi.org/10.1016/j.future.2020.06.024">[link]</a>
- Times series forecasting for urban building energy consumption based on graph convolutional network (Applied Energy, 2022) <a href="https://doi.org/10.1016/j.apenergy.2021.118231">[link]</a>
- Geometric Deep-Learning-Based Spatiotemporal Forecasting for Inverter-Based Solar Power (IEEE Systems Journal, 2023) <a href="https://doi.org/10.1109/JSYST.2023.3250403">[link]</a>
- Short-Term Wind Power Prediction via Spatial Temporal Analysis and Deep Residual Networks (Frontiers in Energy Research, 2022) <a href="https://doi.org/10.3389/fenrg.2022.920407">[link]</a>
- Wind Power Forecasting Based on a Spatial–Temporal Graph Convolution Network With Limited Engineering Knowledge (IEEE TRANSACTIONS ON INSTRUMENTATION AND MEASUREMENT, 2024) <a href="https://doi.org/10.1109/TIM.2024.3374321">[link]</a>
- Towards Effective Long-Term Wind Power Forecasting: A Deep Conditional Generative Spatio-Temporal Approach (IEEE TRANSACTIONS ON KNOWLEDGE AND DATA ENGINEERING, 2024) <a href="https://doi.org/10.1109/TKDE.2024.3435859">[link]</a>
- Deep spatio-temporal feature fusion learning for multi-step building load (ENERGY AND BUILDINGS, 2024) <a href="https://doi.org/10.1016/j.enbuild.2024.114735">[link]</a>
- Self-learning dynamic graph neural network with self-attention based on historical data and future data for multi-task multivariate residential air conditioning forecasting (Applied Energy, 2024) <a href="https://doi.org/10.1016/j.apenergy.2024.123156">[link]</a>
- Mining latent patterns with multi-scale decomposition for electricity demand and price forecasting using modified deep graph convolutional neural networks (Sustainable Energy, Grids and Networks, 2024) <a href="https://doi.org/10.1016/j.segan.2024.101436">[link]</a>
- Boosting short term electric load forecasting of high & medium voltage substations with visibility graphs and graph neural networks (Sustainable Energy, Grids and Networks, 2024) <a href="https://doi.org/10.1016/j.segan.2024.101304">[link]</a>
- Short-term Residential Load Forecasting Based on Dynamic Association Graph Attention Networks for Virtual Power Plant (Dianli Xitong Zidonghua/Automation of Electric Power Systems, 2024) <a href="https://doi.org/10.1109/PSET62496.2024.10808734">[link]</a>
- Modeling and predicting energy consumption of chiller based on dynamic spatial-temporal graph neural network (Journal of Building Engineering, 2024) <a href="https://doi.org/10.1016/j.jobe.2024.109657">[link]</a>
- Electricity demand forecasting at distribution and household levels using explainable causal graph neural network (Energy and AI, 2024) <a href="https://doi.org/10.1016/j.egyai.2024.100368">[link]</a>
- Anomaly detection in smart grid using a trace-based graph deep learning model (Electrical Engineering, 2024) <a href="http://dx.doi.org/10.1007/s00202-024-02327-6">[link]</a>
- Towards efficient model recommendation: An innovative hybrid graph neural network approach integrating multisignature analysis of electrical time series (e-Prime - Advances in Electrical Engineering, Electronics and Energy, 2024) <a href="https://doi.org/10.1016/j.prime.2024.100544">[link]</a>
- Multi-Scale Graph Attention Network Based on Encoding Decomposition for Electricity Consumption Prediction (Energies, 2024) <a href="https://doi.org/10.3390/en17235813">[link]</a>
- Enabling fast prediction of district heating networks transients via a physics-guided graph neural network (Applied Energy, 2024) <a href="https://doi.org/10.1016/j.apenergy.2024.123634">[link]</a>
- A Temporal Ensembling Based Semi-Supervised Graph Convolutional Network for Power Quality Disturbances Classification (IEEE Access, 2024) <a href="https://doi.org/10.1109/ACCESS.2024.3406164">[link]</a>
- MLFGCN: short-term residential load forecasting via graph attention temporal convolution network (Frontiers in Neurorobotics, 2024) <a href="https://doi.org/10.3389/fnbot.2024.1461403">[link]</a>
- Explainable Spatio-Temporal Graph Neural Networks for multi-site photovoltaic energy production (Applied Energy, 2024) <a href="https://doi.org/10.1016/j.apenergy.2023.122151">[link]</a>
- Spatial-Temporal load forecasting of electric vehicle charging stations based on graph neural network (Journal of Intelligent and Fuzzy Systems, 2024) <a href="https://doi.org/10.3233/JIFS-231775">[link]</a>
- A Novel Multiscale Transformer Network Framework for Natural Gas Consumption Forecasting (IEEE Transactions on Industrial Informatics, 2024) <a href="https://doi.org/10.1109/TII.2024.3388089">[link]</a>
- PIDGeuN: Graph Neural Network-Enabled Transient Dynamics Prediction of Networked Microgrids Through Full-Field Measurement (IEEE Access, 2024) <a href="https://doi.org/10.1109/ACCESS.2024.3384457">[link]</a>
- An attentive Copula-based spatio-temporal graph model for multivariate time-series forecasting (Applied Soft Computing, 2024) <a href="https://doi.org/10.1016/j.asoc.2024.111324">[link]</a>
- Residential Electric Load Forecasting via Attentive Transfer of Graph Neural Networks (IJCAI, 2021) <a href="https://doi.org/10.24963/ijcai.2021/374">[link]</a>

<div id='environment'/>

### 🌍 Environment (43 papers)
- Group-Aware Graph Neural Network for Nationwide City Air Quality Forecasting (ACM Transactions on Knowledge Discovery from Data, 2023) <a href="https://doi.org/10.1145/3631713">[link]</a>
- MGC-LSTM: a deep learning model based on graph convolution of multiple graphs for PM2.5 prediction (International Journal of Environmental Science and Technology, 2023) <a href="https://doi.org/10.1007/s13762-022-04553-6">[link]</a>
- Effective PM2.5 concentration forecasting based on multiple spatial–temporal GNN for areas without monitoring stations (Expert Systems with Applications, 2023) <a href="https://doi.org/10.1016/j.eswa.2023.121074">[link]</a>
- A nested machine learning approach to short-term PM2.5 prediction in metropolitan areas using PM2.5 data from different sensor networks (Science of the Total Environment, 2023) <a href="http://dx.doi.org/10.1016/j.scitotenv.2023.162336">[link]</a>
- A Hybrid Model for Spatiotemporal Air Quality Prediction Based on Interpretable Neural Networks and a Graph Neural Network (Atmosphere, 2023) <a href="https://doi.org/10.3390/atmos14121807">[link]</a>
- Adaptive scalable spatio-temporal graph convolutional network for PM2.5 prediction (Engineering Applications of Artificial Intelligence, 2023) <a href="https://doi.org/10.1016/j.engappai.2023.107080">[link]</a>
- Spatiotemporal graph neural networks for predicting mid-to-long-term PM2.5 concentrations (Journal of Cleaner Production, 2023) <a href="https://doi.org/10.1016/j.jclepro.2023.138880">[link]</a>
- An Ensemble Model with Adaptive Variational Mode Decomposition and Multivariate Temporal Graph Neural Network for PM2.5 Concentration Forecasting (Sustainability, 2022) <a href="https://doi.org/10.3390/su142013191">[link]</a>
- HiGRN: A Hierarchical Graph Recurrent Network for Global Sea Surface Temperature Prediction (ACM Trans. Intell. Syst. Technol., 2023) <a href="https://doi.org/10.1145/3597937">[link]</a>
- Spatio-temporal wind speed forecasting using graph networks and novel Transformer architectures (Applied Energy, 2023) <a href="https://doi.org/10.1016/j.apenergy.2022.120565">[link]</a>
- A long-term water quality prediction model for marine ranch based on time-graph convolutional neural network (Ecological Indicators, 2023) <a href="https://doi.org/10.1016/j.ecolind.2023.110782">[link]</a>
- The data-based adaptive graph learning network for analysis and prediction of offshore wind speed (Energy, 2023) <a href="https://doi.org/10.1016/j.energy.2022.126590">[link]</a>
- Significant wave height prediction based on dynamic graph neural network with fusion of ocean characteristics (Dynamics of Atmospheres and Oceans, 2023) <a href="https://doi.org/10.1016/j.dynatmoce.2023.101388">[link]</a>
- Spatiotemporal graph neural network for multivariate multi-step ahead time-series forecasting of sea temperature (Engineering Applications of Artificial Intelligence, 2023) <a href="https://doi.org/10.1016/j.engappai.2023.106854">[link]</a>
- A Unified Graph Formulation for Spatio-Temporal Wind Forecasting (Energies, 2023) <a href="https://doi.org/10.3390/en16207179">[link]</a>
- Adaptive graph neural network based South China Sea seawater temperature prediction and multivariate uncertainty correlation analysis (Stochastic Environmental Research and Risk Assessment, 2023) <a href="https://doi.org/10.1007/s00477-022-02371-3">[link]</a>
- Global Spatiotemporal Graph Attention Network for Sea Surface Temperature Prediction (IEEE Geoscience and Remote Sensing Letters, 2023) <a href="https://doi.org/10.1109/LGRS.2023.3250237">[link]</a>
- A Graph Neural Network with Spatio-Temporal Attention for Multi-Sources Time Series Data: An Application to Frost Forecast (Sensors, 2022) <a href="https://doi.org/10.3390/s22041486">[link]</a>
- Time Series Prediction of Sea Surface Temperature Based on an Adaptive Graph Learning Neural Model (Future Internet, 2022) <a href="https://doi.org/10.3390/fi14060171">[link]</a>
- Graph optimization neural network with spatio-temporal correlation learning for multi-node offshore wind speed forecasting (Renewable Energy, 2021) <a href="https://doi.org/10.1016/j.renene.2021.08.066">[link]</a>
- Time-Series Graph Network for Sea Surface Temperature Prediction (Big Data Research, 2021) <a href="https://doi.org/10.1016/j.bdr.2021.100237">[link]</a>
- Graph neural network for groundwater level forecasting (JOURNAL OF HYDROLOGY, 2023) <a href="https://doi.org/10.1016/j.jhydrol.2022.128792">[link]</a>
- Dynamic adaptive spatio-temporal graph neural network for multi-node offshore wind speed forecasting (APPLIED SOFT COMPUTING, 2023) <a href="https://doi.org/10.1016/j.asoc.2023.110294">[link]</a>
- A Structured Graph Neural Network for Improving the Numerical Weather Prediction of Rainfall (JOURNAL OF GEOPHYSICAL RESEARCH-ATMOSPHERES, 2023) <a href="https://doi.org/10.1029/2023JD039011">[link]</a>
- Ocean Wind Speed Prediction Based on the Fusion of Spatial Clustering and an Improved Residual Graph Attention Network (JOURNAL OF MARINE SCIENCE AND ENGINEERING, 2023) <a href="https://doi.org/10.3390/jmse11122350">[link]</a>
- A graph multi-head self-attention neural network for the multi-point long-term prediction of sea surface temperature (REMOTE SENSING LETTERS, 2023) <a href="https://doi.org/10.1080/2150704X.2023.2240506">[link]</a>
- PM2.5 prediction based on dynamic spatiotemporal graph neural network (APPLIED INTELLIGENCE, 2024) <a href="https://doi.org/10.1007/s10489-024-05801-7">[link]</a>
- Sea clutter prediction based on fusion of Fourier transform and graph neural network (INTERNATIONAL JOURNAL OF REMOTE SENSING, 2024) <a href="http://dx.doi.org/10.1080/01431161.2024.2391104">[link]</a>
- Modeling freshwater plankton community dynamics with static and dynamic interactions using graph convolution embedded long short-term memory (Water Research, 2024) <a href="https://doi.org/10.1016/j.watres.2024.122401">[link]</a>
- Spatial-temporal graph neural networks for groundwater data (Scientific Reports, 2024) <a href="https://doi.org/10.1038/s41598-024-75385-2">[link]</a>
- Dynamic spatial–temporal model for carbon emission forecasting (Journal of Cleaner Production, 2024) <a href="https://doi.org/10.1016/j.jclepro.2024.142581">[link]</a>
- TEMDI: A Temporal Enhanced Multisource Data Integration model for accurate PM2.5 concentration forecasting (Atmospheric Pollution Research, 2024) <a href="https://doi.org/10.1016/j.apr.2024.102269">[link]</a>
- Ensemble Empirical Mode Decomposition Granger Causality Test Dynamic Graph Attention Transformer Network: Integrating Transformer and Graph Neural Network Models for Multi-Sensor Cross-Temporal Granularity Water Demand Forecasting (Applied Sciences (Switzerland), 2024) <a href="https://doi.org/10.3390/app14083428">[link]</a>
- Pre-training enhanced spatio-temporal graph neural network for predicting influent water quality and flow rate of wastewater treatment plant: Improvement of forecast accuracy and analysis of related factors (Science of the Total Environment, 2024) <a href="https://doi.org/10.1016/j.scitotenv.2024.175411">[link]</a>
- A novel physics-aware graph network using high-order numerical methods in weather forecasting model (Knowledge-Based Systems, 2024) <a href="https://doi.org/10.1016/j.knosys.2024.112158">[link]</a>
- A GCN-based adaptive generative adversarial network model for short-term wind speed scenario prediction (Energy, 2024) <a href="https://doi.org/10.1016/j.energy.2024.130931">[link]</a>
- Spatiotemporal hierarchical transmit neural network for regional-level air-quality prediction (Knowledge-Based Systems, 2024) <a href="https://doi.org/10.1016/j.knosys.2024.111555">[link]</a>
- Predicting Global Average Temperature Time Series Using an Entire Graph Node Training Approach (IEEE Transactions on Geoscience and Remote Sensing, 2024) <a href="http://dx.doi.org/10.1109/TGRS.2024.3480888">[link]</a>
- A Multi-Modal Deep-Learning Air Quality Prediction Method Based on Multi-Station Time-Series Data and Remote-Sensing Images: Case Study of Beijing and Tianjin (Entropy, 2024) <a href="https://doi.org/10.3390/e26010091">[link]</a>
- Application of graph-structured data for forecasting the dynamics of time series of natural origin (European Physical Journal: Special Topics, 2024) <a href="https://doi.org/10.1140/epjs/s11734-024-01368-z">[link]</a>
- Physics-Guided Graph Meta Learning for Predicting Water Temperature and Streamflow in Stream Networks (KDD, 2022) <a href="https://doi.org/10.1145/3534678.3539115">[link]</a>
- Forecasting Soil Moisture Using Domain Inspired Temporal Graph Convolution Neural Networks To Guide Sustainable Crop Management (IJCAI, 2023) <a href="https://doi.org/10.24963/ijcai.2023/654">[link]</a>
- Spatio-Temporal Transformer Network with Physical Knowledge Distillation for Weather Forecasting (CIKM, 2024) <a href="https://doi.org/10.1145/3627673.3679841">[link]</a>

<div id='finance'/>

### 📈 Finance (14 papers)
- ML-GAT:A Multilevel Graph Attention Model for Stock Prediction (IEEE Access, 2022) <a href="https://doi.org/10.1109/ACCESS.2022.3199008">[link]</a>
- A representation-learning-based approach to predict stock price trend via dynamic spatiotemporal feature embedding (Engineering Applications of Artificial Intelligence, 2023) <a href="https://doi.org/10.1016/j.engappai.2023.106849">[link]</a>
- Learning Dynamic Dependencies With Graph Evolution Recurrent Unit for Stock Predictions (IEEE Transactions on Systems, Man, and Cybernetics: Systems, 2023) <a href="https://doi.org/10.1109/TSMC.2023.3284840">[link]</a>
- A knowledge graph–GCN–community detection integrated model for large-scale stock price prediction (Applied Soft Computing, 2023) <a href="https://doi.org/10.1016/j.asoc.2023.110595">[link]</a>
- Corporate investment prediction using a weighted temporal graph neural network (Wiley Interdisciplinary Reviews: Data Mining and Knowledge Discovery, 2022) <a href="https://doi.org/10.1002/widm.1472">[link]</a>
- MONEY: Ensemble learning for stock price movement prediction via a convolutional network with adversarial hypergraph model (AI Open, 2023) <a href="https://doi.org/10.1016/j.aiopen.2023.10.002">[link]</a>
- Financial time series forecasting with multi-modality graph neural network (Pattern Recognition, 2022) <a href="https://doi.org/10.1016/j.patcog.2021.108218">[link]</a>
- Attentive gated graph sequence neural network-based time-series information fusion for financial trading (Information Fusion, 2023) <a href="https://doi.org/10.1016/j.inffus.2022.10.006">[link]</a>
- Research on Commodities Constraint Optimization Based on Graph Neural Network Prediction (IEEE Access, 2023) <a href="https://doi.org/10.1109/ACCESS.2023.3302923">[link]</a>
- Intricate Supply Chain Demand Forecasting Based on Graph Convolution Network (Sustainability (Switzerland), 2024) <a href="https://doi.org/10.3390/su16219608">[link]</a>
- Volatility forecasting for stock market incorporating media reports, investors' sentiment, and attention based on MTGNN model (Journal of Forecasting, 2024) <a href="https://doi.org/10.1002/for.3101">[link]</a>
- Multi-Granularity Spatio-Temporal Correlation Networks for Stock Trend Prediction (IEEE Access, 2024) <a href="https://doi.org/10.1109/ACCESS.2024.3393774">[link]</a>
- A Graph-based Spatiotemporal Model for Energy Markets (CIKM, 2022) <a href="https://doi.org/10.1145/3511808.3557530">[link]</a>
- Temporal and Heterogeneous Graph Neural Network for Financial Time Series Prediction (CIKM, 2022) <a href="https://doi.org/10.1145/3511808.3557089">[link]</a>

<div id='health'/>

### 🧑‍⚕️ Health (23 papers)
- SSGCNet: A Sparse Spectra Graph Convolutional Network for Epileptic EEG Signal Classification (IEEE TRANSACTIONS ON NEURAL NETWORKS AND LEARNING SYSTEMS, 2023) <a href="http://dx.doi.org/10.1109/TNNLS.2023.3252569">[link]</a>
- Predicting influenza with pandemic-awareness via Dynamic Virtual Graph Significance Networks (Computers in Biology and Medicine, 2023) <a href="https://doi.org/10.1016/j.compbiomed.2023.106807">[link]</a>
- Spatio-temporal Graph Learning for Epidemic Prediction (ACM Trans. Intell. Syst. Technol., 2023) <a href="https://doi.org/10.1145/3579815">[link]</a>
- Integrating Transformer and GCN for COVID-19 Forecasting (Sustainability (Switzerland), 2022) <a href="https://doi.org/10.3390/su141610393">[link]</a>
- Deep learning of contagion dynamics on complex networks (Nature Communications, 2021) <a href="https://doi.org/10.1038/s41467-021-24732-2">[link]</a>
- Forecasting the COVID-19 Space-Time Dynamics in Brazil With Convolutional Graph Neural Networks and Transport Modals (IEEE Access, 2022) <a href="https://doi.org/10.1109/access.2022.3195535">[link]</a>
- Exploring unsupervised multivariate time series representation learning for chronic disease diagnosis (International Journal of Data Science and Analytics, 2023) <a href="https://doi.org/10.1007/s41060-021-00290-0">[link]</a>
- Time-Aware Context-Gated Graph Attention Network for Clinical Risk Prediction (IEEE Transactions on Knowledge and Data Engineering, 2023) <a href="https://doi.org/10.1109/TKDE.2022.3181780">[link]</a>
- Integrating gated recurrent unit in graph neural network to improve infectious disease prediction: an attempt (FRONTIERS IN PUBLIC HEALTH, 2024) <a href="https://doi.org/10.3389/fpubh.2024.1397260">[link]</a>
- EEG Emotion Recognition Network Based on Attention and Spatiotemporal Convolution (SENSORS, 2024) <a href="https://doi.org/10.3390/s24113464">[link]</a>
- Convolutional gated recurrent unit-driven multidimensional dynamic graph neural network for subject-independent emotion recognition (Expert Systems with Applications, 2024) <a href="https://doi.org/10.1016/j.eswa.2023.121889">[link]</a>
- Graph neural ordinary differential equations for epidemic forecasting (CCF Transactions on Pervasive Computing and Interaction, 2024) <a href="https://doi.org/10.1007/s42486-024-00161-0">[link]</a>
- Modeling epidemic dynamics using Graph Attention based Spatial Temporal networks (PLoS ONE, 2024) <a href="https://doi.org/10.1371/journal.pone.0307159">[link]</a>
- Backbone-based Dynamic Spatio-Temporal Graph Neural Network for epidemic forecasting (Knowledge-Based Systems, 2024) <a href="https://doi.org/10.1016/j.knosys.2024.111952">[link]</a>
- Forecasting Epidemic Spread with Recurrent Graph Gate Fusion Transformers (IEEE Journal of Biomedical and Health Informatics, 2024) <a href="http://dx.doi.org/10.1109/JBHI.2024.3488274">[link]</a>
- BrainNet: Epileptic Wave Detection from SEEG with Hierarchical Graph Diffusion Learning (KDD, 2022) <a href="https://doi.org/10.1145/3534678.3539178">[link]</a>
- Does Air Quality Really Impact COVID-19 Clinical Severity: Coupling NASA Satellite Datasets with Geometric Deep Learning (KDD, 2021) <a href="https://doi.org/10.1145/3447548.3467207">[link]</a>
- MBrain: A Multi-channel Self-Supervised Learning Framework for Brain Signals (KDD, 2023) <a href="https://doi.org/10.1145/3580305.3599426">[link]</a>
- A Bayesian Graph Neural Network for EEG Classification - A Win-Win on Performance and Interpretability (ICDE, 2023) <a href="https://doi.org/10.1109/ICDE55515.2023.00165">[link]</a>
- Temporal Multiresolution Graph Neural Networks For Epidemic Prediction (ICML, 2022) <a href="https://proceedings.mlr.press/v184/hy22a/hy22a.pdf">[link]</a>
- Cola-GNN: Cross-location Attention based Graph Neural Networks for Long-term ILI Prediction (CIKM, 2020) <a href="https://doi.org/10.1145/3340531.3411975">[link]</a>
- Hierarchical Spatio-Temporal Graph Neural Networks for Pandemic Forecasting (CIKM, 2022) <a href="https://doi.org/10.1145/3511808.3557350">[link]</a>
- HierST: A Unified Hierarchical Spatial-temporal Framework for COVID-19 Trend Forecasting (CIKM, 2021) <a href="https://doi.org/10.1145/3459637.3481927">[link]</a>

<div id='mobility'/>

### 🚗 Mobility (103 papers)
- Make More Connections: Urban Traffic Flow Forecasting with Spatiotemporal Adaptive Gated Graph Convolution Network (ACM Trans. Intell. Syst. Technol., 2022) <a href="https://doi.org/10.1145/3488902">[link]</a>
- 3DGCN: 3-Dimensional Dynamic Graph Convolutional Network for Citywide Crowd Flow Prediction (ACM Trans. Knowl. Discov. Data, 2021) <a href="https://doi.org/10.1145/3451394">[link]</a>
- Crowd Flow Prediction for Irregular Regions with Semantic Graph Attention Network (ACM Trans. Intell. Syst. Technol., 2022) <a href="https://doi.org/10.1145/3501805">[link]</a>
- Dynamic Multi-View Graph Neural Networks for Citywide Traffic Inference (ACM Trans. Knowl. Discov. Data, 2023) <a href="https://doi.org/10.1145/3564754">[link]</a>
- Deep Spatio-temporal Adaptive 3D Convolutional Neural Networks for Traffic Flow Prediction (ACM Trans. Intell. Syst. Technol., 2022) <a href="https://doi.org/10.1145/3510829">[link]</a>
- DMGF-Net: An Efficient Dynamic Multi-Graph Fusion Network for Traffic Prediction (ACM Trans. Knowl. Discov. Data, 2023) <a href="https://doi.org/10.1145/3586164">[link]</a>
- Dynamic Graph Convolutional Recurrent Network for Traffic Prediction: Benchmark and Solution (ACM Trans. Knowl. Discov. Data, 2023) <a href="https://doi.org/10.1145/3532611">[link]</a>
- Passenger Mobility Prediction via Representation Learning for Dynamic Directed and Weighted Graphs (ACM Trans. Intell. Syst. Technol., 2021) <a href="https://doi.org/10.1145/3446344">[link]</a>
- GSAA: A Novel Graph Spatiotemporal Attention Algorithm&nbsp;for Smart City Traffic Prediction (ACM Trans. Sen. Netw., 2023) <a href="https://doi.org/10.1145/3631608">[link]</a>
- Graph Neural Network-Driven Traffic Forecasting for the Connected Internet of Vehicles (IEEE Transactions on Network Science and Engineering, 2022) <a href="https://doi.org/10.1109/TNSE.2021.3126830">[link]</a>
- Sequential Graph Neural Network for Urban Road Traffic Speed Prediction (IEEE Access, 2020) <a href="https://doi.org/10.1109/ACCESS.2019.2915364">[link]</a>
- Bayesian Combination Approach to Traffic Forecasting With Graph Attention Network and ARIMA Model (IEEE Access, 2023) <a href="https://doi.org/10.1109/ACCESS.2023.3310821">[link]</a>
- Automated Dilated Spatio-Temporal Synchronous Graph Modeling for Traffic Prediction (IEEE Transactions on Intelligent Transportation Systems, 2023) <a href="https://doi.org/10.1109/TITS.2022.3195232">[link]</a>
- Multi-View Spatial-Temporal Graph Neural Network for Traffic Prediction (Computer Journal, 2023) <a href="https://doi.org/10.1093/comjnl/bxac086">[link]</a>
- Spatio-temporal graph mixformer for traffic forecasting (Expert Systems with Applications, 2023) <a href="https://doi.org/10.1016/j.eswa.2023.120281">[link]</a>
- DyGCN-LSTM: A dynamic GCN-LSTM based encoder-decoder framework for multistep traffic prediction (Applied Intelligence, 2023) <a href="https://doi.org/10.1007/s10489-023-04871-3">[link]</a>
- Deep trip generation with graph neural networks for bike sharing system expansion (Transportation Research Part C: Emerging Technologies, 2023) <a href="https://doi.org/10.1016/j.trc.2023.104241">[link]</a>
- Spatial-temporal correlated graph neural networks based on neighborhood feature selection for traffic data prediction (Applied Intelligence, 2023) <a href="https://doi.org/10.1007/s10489-022-03753-4">[link]</a>
- Adaptive graph generation based on generalized pagerank graph neural network for traffic flow forecasting (Applied Intelligence, 2023) <a href="https://doi.org/10.1007/s10489-023-05137-8">[link]</a>
- Dynamic Correlation Adjacency-Matrix-Based Graph Neural Networks for Traffic Flow Prediction (Sensors, 2023) <a href="https://doi.org/10.3390/s23062897">[link]</a>
- Taxi demand forecasting based on the temporal multimodal information fusion graph neural network (Applied Intelligence, 2022) <a href="https://doi.org/10.1007/s10489-021-03128-1">[link]</a>
- Graph Attention LSTM: A Spatiotemporal Approach for Traffic Flow Forecasting (IEEE Intelligent Transportation Systems Magazine, 2022) <a href="https://doi.org/10.1109/MITS.2020.2990165">[link]</a>
- Multi-featured spatial-temporal and dynamic multi-graph convolutional network for metro passenger flow prediction (Connection Science, 2022) <a href="https://doi.org/10.1080/09540091.2022.2061915">[link]</a>
- TFGAN: Traffic forecasting using generative adversarial network with multi-graph convolutional network (Knowledge-Based Systems, 2022) <a href="https://doi.org/10.1016/j.knosys.2022.108990">[link]</a>
- Dc-stgcn: Dual-channel based graph convolutional networks for network traffic forecasting (Electronics (Switzerland), 2021) <a href="https://doi.org/10.3390/electronics10091014">[link]</a>
- Graph Sequence Neural Network with an Attention Mechanism for Traffic Speed Prediction (ACM Transactions on Intelligent Systems and Technology, 2022) <a href="https://doi.org/10.1145/3470889">[link]</a>
- Spatio-temporal causal graph attention network for traffic flow prediction in intelligent transportation systems (PeerJ Computer Science, 2023) <a href="http://dx.doi.org/10.7717/peerj-cs.1484">[link]</a>
- Dynamic spatio-temporal graph-based CNNs for traffic flow prediction (IEEE Access, 2020) <a href="https://doi.org/10.1109/ACCESS.2020.3027375">[link]</a>
- Spatial-temporal graph neural network for traffic forecasting: An overview and open research issues (Applied Intelligence, 2022) <a href="https://doi.org/10.1007/s10489-021-02587-w">[link]</a>
- A Deep Learning Approach for Flight Delay Prediction Through Time-Evolving Graphs (IEEE Transactions on Intelligent Transportation Systems, 2022) <a href="https://doi.org/10.1109/TITS.2021.3103502">[link]</a>
- STAGCN: Spatial–Temporal Attention Graph Convolution Network for Traffic Forecasting (Mathematics, 2022) <a href="https://doi.org/10.3390/math10091599">[link]</a>
- A Novel Spatial-Temporal Multi-Scale Alignment Graph Neural Network Security Model for Vehicles Prediction (IEEE Transactions on Intelligent Transportation Systems, 2023) <a href="https://doi.org/10.1109/TITS.2022.3140229">[link]</a>
- MSASGCN: Multi-Head Self-Attention Spatiotemporal Graph Convolutional Network for Traffic Flow Forecasting (Journal of Advanced Transportation, 2022) <a href="https://doi.org/10.1155/2022/2811961">[link]</a>
- Effect of dockless bike-sharing scheme on the demand for London Cycle Hire at the disaggregate level using a deep learning approach (Transportation Research Part A: Policy and Practice, 2022) <a href="https://doi.org/10.1016/j.tra.2022.10.013">[link]</a>
- Bi-GRCN: A Spatio-Temporal Traffic Flow Prediction Model Based on Graph Neural Network (Journal of Advanced Transportation, 2022) <a href="https://doi.org/10.1155/2022/5221362">[link]</a>
- A Recurrent Spatio-Temporal Graph Neural Network Based on Latent Time Graph for Multi-Channel Time Series Forecasting (IEEE Signal Processing Letters, 2024) <a href="http://dx.doi.org/10.1109/LSP.2024.3479917">[link]</a>
- Scale-Aware Neural Architecture Search for Multivariate Time Series Forecasting (ACM Trans. Knowl. Discov. Data, 2024) <a href="https://doi.org/10.1145/3701038">[link]</a>
- Spatiotemporal Forecasting of Traffic Flow Using Wavelet-Based Temporal Attention (IEEE ACCESS, 2024) <a href="https://doi.org/10.1109/ACCESS.2024.3516195">[link]</a>
- Urban Traffic Flow Forecasting Based on Graph Structure Learning (JOURNAL OF ADVANCED TRANSPORTATION, 2024) <a href="https://doi.org/10.1155/atr/7878081">[link]</a>
- Metro Flow Prediction with Hierarchical Hypergraph Attention Networks (IEEE Transactions on Artificial Intelligence, 2024) <a href="https://doi.org/10.1109/TAI.2023.3337052">[link]</a>
- Predictive resilience assessment of road networks based on dynamic multi-granularity graph neural network (Neurocomputing, 2024) <a href="https://doi.org/10.1016/j.neucom.2024.128207">[link]</a>
- Transformer-Based Spatiotemporal Graph Diffusion Convolution Network for Traffic Flow Forecasting (Electronics (Switzerland), 2024) <a href="https://doi.org/10.3390/electronics13163151">[link]</a>
- Network traffic prediction with Attention-based Spatial–Temporal Graph Network (Computer Networks, 2024) <a href="https://doi.org/10.1016/j.comnet.2024.110296">[link]</a>
- Dynamic attention aggregated missing spatial–temporal data imputation for traffic speed prediction (Neurocomputing, 2024) <a href="https://doi.org/10.1016/j.neucom.2024.128441">[link]</a>
- A two-level resolution neural network with enhanced interpretability for freeway traffic forecasting (Scientific Reports, 2024) <a href="https://doi.org/10.1038/s41598-024-78148-1">[link]</a>
- Predicting the Aggregate Mobility of a Vehicle Fleet within a City Graph (Algorithms, 2024) <a href="https://doi.org/10.3390/a17040166">[link]</a>
- Integrated Spatio-Temporal Graph Neural Network for Traffic Forecasting (Applied Sciences (Switzerland), 2024) <a href="https://doi.org/10.3390/app142411477">[link]</a>
- Dynamic spatial aware graph transformer for spatiotemporal traffic flow forecasting (Knowledge-Based Systems, 2024) <a href="https://doi.org/10.1016/j.knosys.2024.111946">[link]</a>
- Long-Term Airport Network Performance Forecasting With Linear Diffusion Graph Networks (IEEE Transactions on Intelligent Transportation Systems, 2024) <a href="https://doi.org/10.1109/TITS.2024.3420423">[link]</a>
- Periodicity aware spatial-temporal adaptive hypergraph neural network for traffic forecasting (GeoInformatica, 2024) <a href="https://doi.org/10.1007/s10707-024-00527-7">[link]</a>
- Dual Graph for Traffic Forecasting (IEEE Access, 2024) <a href="https://doi.org/10.1109/ACCESS.2019.2958380">[link]</a>
- PGCN: Progressive Graph Convolutional Networks for Spatial-Temporal Traffic Forecasting (IEEE Transactions on Intelligent Transportation Systems, 2024) <a href="https://doi.org/10.1109/TITS.2024.3349565">[link]</a>
- Channel attention-based spatial-temporal graph neural networks for traffic prediction (Data Technologies and Applications, 2024) <a href="http://dx.doi.org/10.1108/DTA-09-2022-0378">[link]</a>
- Spatiotemporal Propagation Learning for Network-Wide Flight Delay Prediction (IEEE Transactions on Knowledge and Data Engineering, 2024) <a href="https://doi.org/10.1109/TKDE.2023.3286690">[link]</a>
- AMGCN: adaptive multigraph convolutional networks for traffic speed forecasting (Applied Intelligence, 2024) <a href="https://doi.org/10.1007/s10489-024-05301-8">[link]</a>
- A self-attention dynamic graph convolution network model for traffic flow prediction (International Journal of Machine Learning and Cybernetics, 2024) <a href="https://doi.org/10.1007/s13042-024-02210-7">[link]</a>
- A Spatial-Temporal Aggregated Graph Neural Network for Docked Bike-sharing Demand Forecasting (ACM Trans. Knowl. Discov. Data, 2024) <a href="https://doi.org/10.1145/3690388">[link]</a>
- BigST: Linear Complexity Spatio-Temporal Graph Neural Network for Traffic Forecasting on Large-Scale Road Networks (Proc. VLDB Endow., 2024) <a href="https://doi.org/10.14778/3641204.3641217">[link]</a>
- Score-based Graph Learning for Urban Flow Prediction (ACM Trans. Intell. Syst. Technol., 2024) <a href="https://doi.org/10.1145/3655629">[link]</a>
- STWave+: A Multi-Scale Efficient Spectral Graph Attention Network With Long-Term Trends for Disentangled Traffic Flow Forecasting (IEEE Transactions on Knowledge and Data Engineering, 2024) <a href="https://doi.org/10.1109/TKDE.2023.3324501">[link]</a>
- Exploring Bus Stop Mobility Pattern: A Multi-Pattern Deep Learning Prediction Framework (IEEE Transactions on Intelligent Transportation Systems, 2024) <a href="http://dx.doi.org/10.1109/TITS.2023.3345872">[link]</a>
- Pre-training Enhanced Spatial-temporal Graph Neural Network for Multivariate Time Series Forecasting (KDD, 2022) <a href="https://doi.org/10.1145/3534678.3539396">[link]</a>
- SAGDFN: A Scalable Adaptive Graph Diffusion Forecasting Network for Multivariate Time Series Forecasting (ICDE, 2024) <a href="https://doi.org/10.1109/ICDE60146.2024.00101">[link]</a>
- Dynamic and Multi-faceted Spatio-temporal Deep Learning for Traffic Speed Forecasting (KDD, 2021) <a href="https://doi.org/10.1145/3447548.3467275">[link]</a>
- Dynamic Flow Distribution Prediction for Urban Dockless E-Scooter Sharing Reconfiguration (WWW, 2020) <a href="https://doi.org/10.1145/3366423.3380101">[link]</a>
- Irregular Traffic Time Series Forecasting Based on Asynchronous Spatio-Temporal Graph Convolutional Networks (KDD, 2024) <a href="https://doi.org/10.1145/3637528.3671665">[link]</a>
- Spatio-Temporal Graph Few-Shot Learning with Cross-City Knowledge Transfer (KDD, 2022) <a href="https://doi.org/10.1145/3534678.3539281">[link]</a>
- Towards Fine-grained Flow Forecasting: A Graph Attention Approach for Bike Sharing Systems (WWW, 2020) <a href="https://doi.org/10.1145/3366423.3380097">[link]</a>
- Traffic Flow Prediction via Spatial Temporal Graph Neural Network (WWW, 2020) <a href="https://doi.org/10.1145/3366423.3380186">[link]</a>
- Transferable Graph Structure Learning for Graph-based Traffic Forecasting Across Cities (KDD, 2023) <a href="https://doi.org/10.1145/3580305.3599529">[link]</a>
- When Spatio-Temporal Meet Wavelets: Disentangled Traffic Forecasting via Efficient Spectral Graph Attention Networks (ICDE, 2023) <a href="https://doi.org/10.1109/ICDE55515.2023.00046">[link]</a>
- Decoupled Dynamic Spatial-Temporal Graph Neural Network for Traffic Forecasting (VLDB, 2022) <a href="https://doi.org/10.14778/3551793.3551827">[link]</a>
- STSD: Modeling Spatial Temporal Staticity and Dynamicity in Traffic Forecasting (ICDM, 2023) <a href="https://doi.org/10.1109/ICDM58522.2023.00209">[link]</a>
- DSTAGNN: Dynamic Spatial-Temporal Aware Graph Neural Network for Traffic Flow Forecasting (ICML, 2022) <a href="https://proceedings.mlr.press/v162/lan22a/lan22a.pdf">[link]</a>
- Spatial-Temporal Graph ODE Networks for Traffic Flow Forecasting (KDD, 2021) <a href="https://doi.org/10.1145/3447548.3467430">[link]</a>
- Frigate: Frugal Spatio-temporal Forecasting on Road Networks (KDD, 2023) <a href="https://doi.org/10.1145/3580305.3599357">[link]</a>
- MSDR: Multi-Step Dependency Relation Networks for Spatial Temporal Forecasting (KDD, 2022) <a href="https://doi.org/10.1145/3534678.3539397">[link]</a>
- Adaptive graph convolutional recurrent network for traffic forecasting (NeurIPS, 2020) <a href="https://dl.acm.org/doi/abs/10.5555/3495724.3497218">[link]</a>
- Prompt-Based Spatio-Temporal Graph Transfer Learning (CIKM, 2024) <a href="https://doi.org/10.1145/3627673.3679554">[link]</a>
- Spatial-Temporal Graph Boosting Networks: Enhancing Spatial-Temporal Graph Neural Networks via Gradient Boosting (CIKM, 2023) <a href="https://doi.org/10.1145/3583780.3615066">[link]</a>
- Enhancing Dependency Dynamics in Traffic Flow Forecasting via Graph Risk Bootstrap (SIGSPATIAL, 2024) <a href="https://doi.org/10.1145/3678717.3691237">[link]</a>
- MultiSPANS: A Multi-range Spatial-Temporal Transformer Network for Traffic Forecast via Structural Entropy Optimization (WSDM, 2024) <a href="https://doi.org/10.1145/3616855.3635820">[link]</a>
- Adaptive Graph Neural Diffusion for Traffic Demand Forecasting (CIKM, 2023) <a href="https://doi.org/10.1145/3583780.3615153">[link]</a>
- Adaptive Graph Spatial-Temporal Transformer Network for Traffic Forecasting (CIKM, 2022) <a href="https://doi.org/10.1145/3511808.3557540">[link]</a>
- Automated Spatio-Temporal Synchronous Modeling with Multiple Graphs for Traffic Prediction (CIKM, 2022) <a href="https://doi.org/10.1145/3511808.3557243">[link]</a>
- ByGCN: Spatial Temporal Byroad-Aware Graph Convolution Network for Traffic Flow Prediction in Road Networks (CIKM, 2024) <a href="https://doi.org/10.1145/3627673.3679690">[link]</a>
- Contagion Process Guided Cross-scale Spatio-Temporal Graph Neural Network for Traffic Congestion Prediction (SIGSPATIAL, 2023) <a href="https://doi.org/10.1145/3589132.3625639">[link]</a>
- CreST: A Credible Spatiotemporal Learning Framework for Uncertainty-aware Traffic Forecasting (WSDM, 2024) <a href="https://doi.org/10.1145/3616855.3635759">[link]</a>
- DetectorNet: Transformer-enhanced Spatial Temporal Graph Neural Network for Traffic Prediction (SIGSPATIAL, 2021) <a href="https://doi.org/10.1145/3474717.3483920">[link]</a>
- Domain Adversarial Spatial-Temporal Network: A Transferable Framework for Short-term Traffic Forecasting across Cities (CIKM, 2022) <a href="https://doi.org/10.1145/3511808.3557294">[link]</a>
- Graph Convolutional Networks with Kalman Filtering for Traffic Prediction (SIGSPATIAL, 2020) <a href="https://doi.org/10.1145/3397536.3422257">[link]</a>
- Learning Dynamic Graphs from All Contextual Information for Accurate Point-of-Interest Visit Forecasting (SIGSPATIAL, 2023) <a href="https://doi.org/10.1145/3589132.3625567">[link]</a>
- Multi-task graph neural network for truck speed prediction under extreme weather conditions (SIGSPATIAL, 2022) <a href="https://doi.org/10.1145/3557915.3561029">[link]</a>
- Multi-Task Synchronous Graph Neural Networks for Traffic Spatial-Temporal Prediction (SIGSPATIAL, 2021) <a href="https://doi.org/10.1145/3474717.3483921">[link]</a>
- Periodic Shift and Event-aware Spatio-Temporal Graph Convolutional Network for Traffic Congestion Prediction (SIGSPATIAL, 2023) <a href="https://doi.org/10.1145/3589132.3625612">[link]</a>
- Rethinking Attention Mechanism for Spatio-Temporal Modeling: A Decoupling Perspective in Traffic Flow Prediction (CIKM, 2024) <a href="https://doi.org/10.1145/3627673.3679571">[link]</a>
- Rethinking Sensors Modeling: Hierarchical Information Enhanced Traffic Forecasting (CIKM, 2023) <a href="https://doi.org/10.1145/3583780.3614910">[link]</a>
- Spatial-Temporal Convolutional Graph Attention Networks for Citywide Traffic Flow Forecasting (CIKM, 2020) <a href="https://doi.org/10.1145/3340531.3411941">[link]</a>
- Spatio-Temporal Pyramid Networks for Traffic Forecasting (ECML PKDD, 2023) <a href="https://doi.org/10.1007/978-3-031-43412-9_20">[link]</a>
- Spatiotemporal Adaptive Gated Graph Convolution Network for Urban Traffic Flow Forecasting (CIKM, 2020) <a href="https://doi.org/10.1145/3340531.3411894">[link]</a>
- ST-GRAT: A Novel Spatio-temporal Graph Attention Networks for Accurately Forecasting Dynamically Changing Road Speed (CIKM, 2020) <a href="https://doi.org/10.1145/3340531.3411940">[link]</a>
- STP-TrellisNets: Spatial-Temporal Parallel TrellisNets for Metro Station Passenger Flow Prediction (CIKM, 2020) <a href="https://doi.org/10.1145/3340531.3411874">[link]</a>
- SUSTeR: Sparse Unstructured Spatio Temporal Reconstruction on Traffic Prediction (SIGSPATIAL, 2023) <a href="https://doi.org/10.1145/3589132.3625631">[link]</a>

<div id='predictive-monitoring'/>

### 🔍 Predictive monitoring (42 papers)
- Correlation-Based Anomaly Detection Method for Multi-sensor System (Computational Intelligence and Neuroscience, 2022) <a href="https://doi.org/10.1155/2022/4756480">[link]</a>
- Abnormality Detection of Blast Furnace Ironmaking Process Based on an Improved Diffusion Convolutional Gated Recurrent Unit Network (IEEE Transactions on Instrumentation and Measurement, 2023) <a href="https://doi.org/10.1109/TIM.2023.3320734">[link]</a>
- Cost-effective fault diagnosis of nearby photovoltaic systems using graph neural networks (Energy, 2023) <a href="https://doi.org/10.1016/j.energy.2022.126444">[link]</a>
- A recursive multi-head graph attention residual network for high-speed train wheelset bearing fault diagnosis (Measurement Science and Technology, 2023) <a href="https://doi.org/10.1088/1361-6501/acb609">[link]</a>
- Fault Diagnosis of Rolling Bearing Based on WHVG and GCN (IEEE Transactions on Instrumentation and Measurement, 2021) <a href="https://doi.org/10.1109/TIM.2021.3087834">[link]</a>
- Fault Diagnosis of Energy Networks Based on Improved Spatial–Temporal Graph Neural Network With Massive Missing Data (IEEE Transactions on Automation Science and Engineering, 2023) <a href="https://doi.org/10.1109/TASE.2023.3281394">[link]</a>
- Intelligent fault diagnosis of rolling bearings based on the visibility algorithm and graph neural networks (Journal of the Brazilian Society of Mechanical Sciences and Engineering, 2023) <a href="https://doi.org/10.1007/s40430-022-03913-0">[link]</a>
- Fault Prediction for Electromechanical Equipment Based on Spatial-Temporal Graph Information (IEEE Transactions on Industrial Informatics, 2023) <a href="https://doi.org/10.1109/TII.2022.3176891">[link]</a>
- Local-Global Correlation Fusion-Based Graph Neural Network for Remaining Useful Life Prediction (IEEE Transactions on Neural Networks and Learning Systems, 2023) <a href="https://doi.org/10.1109/TNNLS.2023.3330487">[link]</a>
- Convolution-Graph Attention Network with Sensor Embeddings for Remaining Useful Life Prediction of Turbofan Engines (IEEE Sensors Journal, 2023) <a href="https://doi.org/10.1109/JSEN.2023.3279365">[link]</a>
- Comprehensive Dynamic Structure Graph Neural Network for Aero-Engine Remaining Useful Life Prediction (IEEE Transactions on Instrumentation and Measurement, 2023) <a href="https://doi.org/10.1109/TIM.2023.3322481">[link]</a>
- Remaining useful life estimation for engineered systems operating under uncertainty with causal graphnets (Sensors, 2021) <a href="https://doi.org/10.3390/s21196325">[link]</a>
- Bearing Remaining Useful Life Prediction Based on Regression Shapalet and Graph Neural Network (IEEE Transactions on Instrumentation and Measurement, 2022) <a href="https://doi.org/10.1109/TIM.2022.3151169">[link]</a>
- Spatio-Temporal Fusion Attention: A Novel Approach for Remaining Useful Life Prediction Based on Graph Neural Network (IEEE Transactions on Instrumentation and Measurement, 2022) <a href="https://doi.org/10.1109/TIM.2022.3184352">[link]</a>
- Prediction of state of health and remaining useful life of lithium-ion battery using graph convolutional network with dual attention mechanisms (Reliability Engineering and System Safety, 2023) <a href="https://doi.org/10.1016/j.ress.2022.108947">[link]</a>
- Signal Feature Extract Based on Dual-Channel Wavelet Convolutional Network Mixed with Hypergraph Convolutional Network for Fault Diagnosis (IEEE Sensors Journal, 2023) <a href="https://doi.org/10.1109/JSEN.2023.3319537">[link]</a>
- Multivariate multi-step time series prediction of induction motor situation based on fused temporal-spatial features (International Journal of Hydrogen Energy, 2024) <a href="https://doi.org/10.1016/j.ijhydene.2023.11.047">[link]</a>
- Graph Neural Networks With Trainable Adjacency Matrices for Fault Diagnosis on Multivariate Sensor Data (IEEE ACCESS, 2024) <a href="https://doi.org/10.1109/ACCESS.2024.3481331">[link]</a>
- Root cause localization for wind turbines using physics guided multivariate graphical modeling and fault propagation analysis (Knowledge-Based Systems, 2024) <a href="https://doi.org/10.1016/j.knosys.2024.111838">[link]</a>
- Coupling Fault Diagnosis Based on Dynamic Vertex Interpretable Graph Neural Network (Sensors, 2024) <a href="https://doi.org/10.3390/s24134356">[link]</a>
- Outlier detection in temporal and spatial sequences via correlation analysis based on graph neural networks (Displays, 2024) <a href="https://doi.org/10.1016/j.displa.2024.102775">[link]</a>
- Graph spatiotemporal process for multivariate time series anomaly detection with missing values (Information Fusion, 2024) <a href="https://doi.org/10.1016/j.inffus.2024.102255">[link]</a>
- A dual-stream spatio-temporal fusion network with multi-sensor signals for remaining useful life prediction (Journal of Manufacturing Systems, 2024) <a href="https://doi.org/10.1016/j.jmsy.2024.07.004">[link]</a>
- DPDGAD: A Dual-Process Dynamic Graph-based Anomaly Detection for multivariate time series analysis in cyber-physical systems (Advanced Engineering Informatics, 2024) <a href="https://doi.org/10.1016/j.aei.2024.102547">[link]</a>
- Community inspired edge specific message graph convolution network for predictive monitoring of large-scale polymerization processes (Control Engineering Practice, 2024) <a href="https://doi.org/10.1016/j.conengprac.2024.106020">[link]</a>
- EGNN: Energy-efficient anomaly detection for IoT multivariate time series data using graph neural network (Future Generation Computer Systems, 2024) <a href="https://doi.org/10.1016/j.future.2023.09.028">[link]</a>
- Causality Enhanced Global-Local Graph Neural Network for Bioprocess Factor Forecasting (IEEE Transactions on Industrial Informatics, 2024) <a href="https://doi.org/10.1109/TII.2024.3424266">[link]</a>
- Graph neural network-based anomaly detection for river network systems (F1000Research, 2024) <a href="https://doi.org/10.12688/f1000research.136097.2">[link]</a>
- Adversarial Graph Neural Network for Multivariate Time Series Anomaly Detection (IEEE Transactions on Knowledge and Data Engineering, 2024) <a href="https://doi.org/10.1109/TKDE.2024.3419891">[link]</a>
- Gear Fault Diagnosis Method Based on the Optimized Graph Neural Networks (IEEE Transactions on Instrumentation and Measurement, 2024) <a href="https://doi.org/10.1109/TIM.2023.3346512">[link]</a>
- Fault Diagnosis of Energy Networks Based on Improved Spatial-Temporal Graph Neural Network with Massive Missing Data (IEEE Transactions on Automation Science and Engineering, 2024) <a href="https://doi.org/10.1109/TASE.2023.3281394">[link]</a>
- Spatio-Temporal Propagation: An Extended Message Passing Graph Neural Network for Remaining Useful Life Prediction (IEEE Sensors Journal, 2024) <a href="https://doi.org/10.1109/JSEN.2024.3404072">[link]</a>
- Variate Associated Domain Adaptation for Unsupervised Multivariate Time Series Anomaly Detection (ACM Trans. Knowl. Discov. Data, 2024) <a href="https://doi.org/10.1145/3663573">[link]</a>
- Data-Augmentation Based CBAM-ResNet-GCN Method for Unbalance Fault Diagnosis of Rotating Machinery (IEEE Access, 2024) <a href="https://doi.org/10.1109/ACCESS.2024.3368755">[link]</a>
- Learning Sparse Latent Graph Representations for Anomaly Detection in Multivariate Time Series (KDD, 2022) <a href="https://doi.org/10.1145/3534678.3539117">[link]</a>
- Parallel-friendly Spatio-Temporal Graph Learning for Photovoltaic Degradation Analysis at Scale (CIKM, 2024) <a href="https://doi.org/10.1145/3627673.3680026">[link]</a>
- Multivariate Time-Series Anomaly Detection based on Enhancing Graph Attention Networks with Topological Analysis (CIKM, 2024) <a href="https://doi.org/10.1145/3627673.3679614">[link]</a>
- Multi-view Causal Graph Fusion Based Anomaly Detection in Cyber-Physical Infrastructures (CIKM, 2024) <a href="https://doi.org/10.1145/3627673.3680096">[link]</a>
- Accurate Anomaly Detection Leveraging Knowledge-enhanced GAT (ICWS, 2024) <a href="https://doi.org/10.1109/ICWS62655.2024.00077">[link]</a>
- DuoGAT: Dual Time-oriented Graph Attention Networks for Accurate, Efficient and Explainable Anomaly Detection on Time-series (CIKM, 2023) <a href="https://doi.org/10.1145/3583780.3614857">[link]</a>
- Efficient Graph Learning for Anomaly Detection Systems (WSDM, 2023) <a href="https://doi.org/10.1145/3539597.3572990">[link]</a>
- Multivariate Time-Series Anomaly Detection with Temporal Self-supervision and Graphs: Application to Vehicle Failure Prediction (ECML PKDD, 2023) <a href="https://doi.org/10.1007/978-3-031-43430-3_15">[link]</a>

<div id='generic'/>

### 🌐 Generic (77 papers)
- Graph Deep Factors for Probabilistic Time-series Forecasting (ACM Transactions on Knowledge Discovery from Data, 2023) <a href="https://doi.org/10.1145/3543511">[link]</a>
- A Hybrid Continuous-Time Dynamic Graph Representation Learning Model by Exploring Both Temporal and Repetitive Information (ACM Trans. Knowl. Discov. Data, 2023) <a href="https://doi.org/10.1145/3596447">[link]</a>
- Multi-Task Time Series Forecasting Based on Graph Neural Networks (Entropy, 2023) <a href="https://doi.org/10.3390/e25081136">[link]</a>
- Sparse Graph Learning from Spatiotemporal Time Series (Journal of Machine Learning Research, 2023) <a href="https://dl.acm.org/doi/10.5555/3648699.3648941">[link]</a>
- Dynamic spatio-temporal graph network with adaptive propagation mechanism for multivariate time series forecasting (Expert Systems with Applications, 2023) <a href="https://doi.org/10.1016/j.eswa.2022.119374">[link]</a>
- Dynamic spatiotemporal interactive graph neural network for multivariate time series forecasting (Knowledge-Based Systems, 2023) <a href="https://doi.org/10.1016/j.knosys.2023.110995">[link]</a>
- Multivariate Time Series Forecasting With Dynamic Graph Neural ODEs (IEEE Transactions on Knowledge and Data Engineering, 2023) <a href="https://doi.org/10.1109/TKDE.2022.3221989">[link]</a>
- Temporal Chain Network With Intuitive Attention Mechanism for Long-Term Series Forecasting (IEEE Transactions on Instrumentation and Measurement, 2023) <a href="https://doi.org/10.1109/TIM.2023.3322508">[link]</a>
- Multivariate Time Series Forecasting with Transfer Entropy Graph (Tsinghua Science and Technology, 2023) <a href="https://doi.org/10.26599/TST.2021.9010081">[link]</a>
- Dynamic graph structure learning for multivariate time series forecasting (Pattern Recognition, 2023) <a href="https://doi.org/10.1016/j.patcog.2023.109423">[link]</a>
- Learning and integration of adaptive hybrid graph structures for multivariate time series forecasting (Information Sciences, 2023) <a href="https://doi.org/10.1016/j.ins.2023.119560">[link]</a>
- Multi-Scale Adaptive Graph Neural Network for Multivariate Time Series Forecasting (IEEE Transactions on Knowledge and Data Engineering, 2023) <a href="https://doi.org/10.1109/TKDE.2023.3268199">[link]</a>
- TDG4MSF: A temporal decomposition enhanced graph neural network for multivariate time series forecasting (Applied Intelligence, 2023) <a href="https://doi.org/10.1007/s10489-023-04987-6">[link]</a>
- Multivariate long sequence time-series forecasting using dynamic graph learning (Journal of Ambient Intelligence and Humanized Computing, 2023) <a href="https://doi.org/10.1007/s12652-023-04579-9">[link]</a>
- Adaptive dependency learning graph neural networks (Information Sciences, 2023) <a href="https://doi.org/10.1016/j.ins.2022.12.086">[link]</a>
- Multi-scale temporal features extraction based graph convolutional network with attention for multivariate time series prediction (Expert Systems with Applications, 2022) <a href="https://doi.org/10.1016/j.eswa.2022.117011">[link]</a>
- Hierarchical Joint Graph Learning and Multivariate Time Series Forecasting (IEEE Access, 2023) <a href="https://doi.org/10.1109/ACCESS.2023.3325041">[link]</a>
- Multi-channel fusion graph neural network for multivariate time series forecasting (Journal of Computational Science, 2022) <a href="https://doi.org/10.1016/j.jocs.2022.101862">[link]</a>
- MTHetGNN: A heterogeneous graph embedding framework for multivariate time series forecasting (Pattern Recognition Letters, 2022) <a href="https://doi.org/10.1016/j.patrec.2021.12.008">[link]</a>
- Learning Latent ODEs With Graph RNN for Multi-Channel Time Series Forecasting (IEEE Signal Processing Letters, 2023) <a href="https://doi.org/10.1109/LSP.2023.3320439">[link]</a>
- Graph Construction Method for GNN-Based Multivariate Time-Series Forecasting (Computers, Materials and Continua, 2023) <a href="http://dx.doi.org/10.32604/cmc.2023.036830">[link]</a>
- Multivariate Time Series Deep Spatiotemporal Forecasting with Graph Neural Network (Applied Sciences (Switzerland), 2022) <a href="https://doi.org/10.3390/app12115731">[link]</a>
- A New Framework for Smartphone Sensor-Based Human Activity Recognition Using Graph Neural Network (IEEE Sensors Journal, 2021) <a href="http://dx.doi.org/10.1109/JSEN.2020.3015726">[link]</a>
- Graph construction on complex spatiotemporal data for enhancing graph neural network-based approaches (International Journal of Data Science and Analytics, 2023) <a href="https://doi.org/10.1007/s41060-023-00452-2">[link]</a>
- T-GAN: A deep learning framework for prediction of temporal complex networks with adaptive graph convolution and attention mechanism (Displays, 2021) <a href="https://doi.org/10.1016/j.displa.2021.102023">[link]</a>
- Dynamic Causal Explanation Based Diffusion-Variational Graph Neural Network for Spatiotemporal Forecasting (IEEE TRANSACTIONS ON NEURAL NETWORKS AND LEARNING SYSTEMS, 2024) <a href="https://doi.org/10.1109/TNNLS.2024.3415149">[link]</a>
- Multivariate time series classification based on fusion features (Expert Systems with Applications, 2024) <a href="https://doi.org/10.1016/j.eswa.2024.123452">[link]</a>
- CATodyNet: Cross-attention temporal dynamic graph neural network for multivariate time series classification (Knowledge-Based Systems, 2024) <a href="https://doi.org/10.1016/j.knosys.2024.112210">[link]</a>
- Contrastive learning enhanced by graph neural networks for Universal Multivariate Time Series Representation (Information Systems, 2024) <a href="https://doi.org/10.1016/j.is.2024.102429">[link]</a>
- Advanced series decomposition with a gated recurrent unit and graph convolutional neural network for non-stationary data patterns (Journal of Cloud Computing, 2024) <a href="https://doi.org/10.1186/s13677-023-00560-1">[link]</a>
- TodyNet: Temporal dynamic graph neural network for multivariate time series classification (Information Sciences, 2024) <a href="https://doi.org/10.1016/j.ins.2024.120914">[link]</a>
- Multivariate sequence prediction for graph convolutional networks based on ESMD and transfer entropy (Multimedia Tools and Applications, 2024) <a href="https://doi.org/10.1007/s11042-024-18787-8">[link]</a>
- GraphSensor: A Graph Attention Network for Time-Series Sensor (Electronics (Switzerland), 2024) <a href="https://doi.org/10.3390/electronics13122290">[link]</a>
- Dynamic multi-fusion spatio-temporal graph neural network for multivariate time series forecasting (Expert Systems with Applications, 2024) <a href="https://doi.org/10.1016/j.eswa.2023.122729">[link]</a>
- MDG: A Multi-Task Dynamic Graph Generation Framework for Multivariate Time Series Forecasting (IEEE Transactions on Emerging Topics in Computational Intelligence, 2024) <a href="https://doi.org/10.1109/TETCI.2024.3352407">[link]</a>
- Messages are Never Propagated Alone: Collaborative Hypergraph Neural Network for Time-Series Forecasting (IEEE Transactions on Pattern Analysis and Machine Intelligence, 2024) <a href="https://doi.org/10.1109/TPAMI.2023.3331389">[link]</a>
- Dynamic Hypergraph Structure Learning for Multivariate Time Series Forecasting (IEEE Transactions on Big Data, 2024) <a href="https://doi.org/10.1109/TBDATA.2024.3362188">[link]</a>
- On the generalization discrepancy of spatiotemporal dynamics-informed graph convolutional networks (Frontiers in Mechanical Engineering, 2024) <a href="https://doi.org/10.3389/fmech.2024.1397131">[link]</a>
- Multivariate Time-Series Representation Learning via Hierarchical Correlation Pooling Boosted Graph Neural Network (IEEE Transactions on Artificial Intelligence, 2024) <a href="https://doi.org/10.1109/TAI.2023.3241896">[link]</a>
- Graph-Enabled Reinforcement Learning for Time Series Forecasting With Adaptive Intelligence (IEEE Transactions on Emerging Topics in Computational Intelligence, 2024) <a href="https://doi.org/10.1109/TETCI.2024.3398024">[link]</a>
- Dynamic personalized graph neural network with linear complexity for multivariate time series forecasting (Engineering Applications of Artificial Intelligence, 2024) <a href="https://doi.org/10.1016/j.engappai.2023.107291">[link]</a>
- MagNet: Multilevel Dynamic Wavelet Graph Neural Network for Multivariate Time Series Classification (ACM Trans. Knowl. Discov. Data, 2024) <a href="https://doi.org/10.1145/3703915">[link]</a>
- GAGNN: Generative Adversarial Network and Graph Neural Network for Prognostic and Health Management (IEEE Internet of Things Journal, 2024) <a href="https://doi.org/10.1109/JIOT.2024.3435917">[link]</a>
- Hypergraph Convolutional Recurrent Neural Network (KDD, 2020) <a href="https://doi.org/10.1145/3394486.3403389">[link]</a>
- Decoupled Invariant Attention Network for Multivariate Time-series Forecasting (IJCAI, 2024) <a href="https://doi.org/10.24963/ijcai.2024/275">[link]</a>
- DISCRETE GRAPH STRUCTURE LEARNING FOR FORECASTING MULTIPLE TIME SERIES (ICLR, 2021) <a href="https://openreview.net/pdf?id=WEHSlH5mOk">[link]</a>
- GinAR: An End-To-End Multivariate Time Series Forecasting Model Suitable for Variable Missing (KDD, 2024) <a href="https://doi.org/10.1145/3637528.3672055">[link]</a>
- GraFITi: Graphs for Forecasting Irregularly Sampled Time Series (AAAI, 2024) <a href="https://doi.org/10.1609/aaai.v38i15.29560">[link]</a>
- Graph-based Forecasting with Missing Data through Spatiotemporal Downsampling (ICML, 2024) <a href="https://dl.acm.org/doi/10.5555/3692070.3693487">[link]</a>
- GRAPH-GUIDED NETWORK FOR IRREGULARLY SAMPLED MULTIVARIATE TIME SERIES (ICLR, 2022) <a href="https://zitniklab.hms.harvard.edu/publications/papers/raindrop-iclr22.pdf">[link]</a>
- Irregular Multivariate Time Series Forecasting: A Transformable Patching Graph Neural Networks Approach (ICML, 2024) <a href="https://openreview.net/pdf?id=UZlMXUGI6e">[link]</a>
- Learning Decomposed Spatial Relations for Multi-Variate Time-Series Modeling (AAAI, 2023) <a href="https://doi.org/10.1609/aaai.v37i6.25915">[link]</a>
- Learning the Evolutionary and Multi-scale Graph Structure for Multivariate Time Series Forecasting (KDD, 2022) <a href="https://doi.org/10.1145/3534678.3539274">[link]</a>
- Learning Time-Aware Graph Structures for Spatially Correlated Time Series Forecasting (ICDE, 2024) <a href="https://doi.org/10.1109/ICDE60146.2024.00338">[link]</a>
- AGSTN: Learning attention-adjusted graph spatio-temporal networks for short-term urban sensor value forecasting (ICDM, 2020) <a href="https://doi.org/10.1109/ICDM50108.2020.00140">[link]</a>
- Connecting the Dots: Multivariate Time Series Forecasting with Graph Neural Networks (KDD, 2020) <a href="https://doi.org/10.1145/3394486.3403118">[link]</a>
- METRO: A Generic Graph Neural Network Framework for Multivariate Time Series Forecasting (VLDB, 2021) <a href="https://doi.org/10.14778/3489496.3489503">[link]</a>
- Z-GCNETs: Time Zigzags at Graph Convolutional Networks for Time Series Forecasting (ICML, 2021) <a href="https://par.nsf.gov/servlets/purl/10258413">[link]</a>
- TAMP-S2GCNETS: COUPLING TIME-AWARE MULTIPERSISTENCE KNOWLEDGE REPRESENTATION WITH SPATIO-SUPRA GRAPH CONVOLUTIONAL NETWORKS FOR TIME-SERIES FORECASTING (ICLR, 2022) <a href="https://openreview.net/pdf?id=wv6g8fWLX2q">[link]</a>
- HyperTime: A Dynamic Hypergraph Approach for Time Series Classification (ICDM, 2024) <a href="https://doi.org/10.1109/ICDM59182.2024.00064">[link]</a>
- Spectral temporal graph neural network for multivariate time-series forecasting (NeurIPS, 2020) <a href="https://dl.acm.org/doi/10.5555/3495724.3497215">[link]</a>
- Multivariate Time-Series Forecasting with Temporal Polynomial Graph Neural Networks (NeurIPS, 2022) <a href="https://dl.acm.org/doi/10.5555/3600270.3601681">[link]</a>
- Time-Conditioned Dances with Simplicial Complexes: Zigzag Filtration Curve based Supra-Hodge Convolution Networks for Time-series Forecasting (NeurIPS, 2022) <a href="https://dl.acm.org/doi/10.5555/3600270.3600920">[link]</a>
- CrossGNN: Confronting Noisy Multivariate Time Series Via Cross Interaction Refinement (NeurIPS, 2023) <a href="https://dl.acm.org/doi/10.5555/3666122.3668153">[link]</a>
- FourierGNN: Rethinking Multivariate Time Series Forecasting from a Pure Graph Perspective (NeurIPS, 2023) <a href="https://dl.acm.org/doi/10.5555/3666122.3669172">[link]</a>
- Taming Local Effects in Graph-based Spatiotemporal Forecasting (NeurIPS, 2023) <a href="https://dl.acm.org/doi/10.5555/3666122.3668539">[link]</a>
- Towards Unifying Diffusion Models for Probabilistic Spatio-Temporal Graph Learning (SIGSPATIAL, 2024) <a href="https://doi.org/10.1145/3678717.3691235">[link]</a>
- AGCNT: Adaptive Graph Convolutional Network for Transformer-based Long Sequence Time-Series Forecasting (CIKM, 2021) <a href="https://doi.org/10.1145/3459637.3482054">[link]</a>
- DiffSTG: Probabilistic Spatio-Temporal Graph Forecasting with Denoising Diffusion Models (SIGSPATIAL, 2023) <a href="https://doi.org/10.1145/3589132.3625614">[link]</a>
- Explainable Spatio-Temporal Graph Neural Networks (CIKM, 2023) <a href="https://doi.org/10.1145/3583780.3614871">[link]</a>
- H2-Nets: Hyper-hodge Convolutional Neural Networks for Time-Series Forecasting (ECML PKDD, 2023) <a href="https://doi.org/10.1007/978-3-031-43424-2_17">[link]</a>
- Memory Augmented Graph Learning Networks for Multivariate Time Series Forecasting (CIKM, 2022) <a href="https://doi.org/10.1145/3511808.3557638">[link]</a>
- Multivariate Time Series Forecasting By Graph Attention Networks With Theoretical Guarantees (AISTATS, 2024) <a href="https://proceedings.mlr.press/v238/zhang24g/zhang24g.pdf">[link]</a>
- PyTorch Geometric Temporal: Spatiotemporal Signal Processing with Neural Machine Learning Models (CIKM, 2021) <a href="https://doi.org/10.1145/3459637.3482014">[link]</a>
- Spatio-Temporal Meta Contrastive Learning (CIKM, 2023) <a href="https://doi.org/10.1145/3583780.3615065">[link]</a>
- Temporal Graph Neural Networks for Irregular Data (AISTATS, 2023) <a href="https://doi.org/10.48550/arXiv.2302.08415">[link]</a>
- Towards Similarity-Aware Time-Series Classification (SDM, 2022) <a href="https://doi.org/10.1137/1.9781611977172.23">[link]</a>


<div id='other-topics'/>

### 📚 Other topics (33 papers)
- Automatic Modulation Classification Based on CNN-Transformer Graph Neural Network (Sensors, 2023) <a href="https://doi.org/10.3390/s23167281">[link]</a>
- AvgNet: Adaptive Visibility Graph Neural Network and Its Application in Modulation Classification (IEEE Transactions on Network Science and Engineering, 2022) <a href="https://doi.org/10.1109/TNSE.2022.3146836">[link]</a>
- EC-GCN: A encrypted traffic classification framework based on multi-scale graph convolution networks (Computer Networks, 2023) <a href="https://doi.org/10.1016/j.comnet.2023.109614">[link]</a>
- Adaptive Dual-View WaveNet for urban spatial–temporal event prediction (Information Sciences, 2022) <a href="https://doi.org/10.1016/j.ins.2021.12.085">[link]</a>
- A method for the spatiotemporal correlation prediction of the quality of multiple operational processes based on S-GGRU (Advanced Engineering Informatics, 2023) <a href="https://doi.org/10.1016/j.aei.2023.102219">[link]</a>
- Spatial-Temporal Cellular Traffic Prediction for 5G and Beyond: A Graph Neural Networks-Based Approach (IEEE Transactions on Industrial Informatics, 2023) <a href="https://doi.org/10.1109/TII.2022.3182768">[link]</a>
- Graph neural networks for multivariate time series regression with application to seismic data (International Journal of Data Science and Analytics, 2023) <a href="https://doi.org/10.1007/s41060-022-00349-6">[link]</a>
- Data-driven spatiotemporal modeling for structural dynamics on irregular domains by stochastic dependency neural estimation (Computer Methods in Applied Mechanics and Engineering, 2023) <a href="https://doi.org/10.1016/j.cma.2022.115831">[link]</a>
- Proactive control model for safety prediction in tailing dam management: Applying graph depth learning optimization (Process Safety and Environmental Protection, 2023) <a href="https://doi.org/10.1016/j.psep.2023.02.019">[link]</a>
- Physics-informed graph neural network for spatial-temporal production forecasting (Geoenergy Science and Engineering, 2023) <a href="https://doi.org/10.1016/j.geoen.2023.211486">[link]</a>
- A Novel Cellular Network Traffic Prediction Algorithm Based on Graph Convolution Neural Networks and Long Short-Term Memory through Extraction of Spatial-Temporal Characteristics (Processes, 2023) <a href="https://doi.org/10.3390/pr11082257">[link]</a>
- Long-term multivariate time series forecasting in data centers based on multi-factor separation evolutionary spatial–temporal graph neural networks (Knowledge-Based Systems, 2023) <a href="https://doi.org/10.1016/j.knosys.2023.110997">[link]</a>
- Runoff Prediction Based on Dynamic Spatiotemporal Graph Neural Network (Water (Switzerland), 2023) <a href="https://doi.org/10.3390/w15132463">[link]</a>
- App Popularity Prediction by Incorporating Time-Varying Hierarchical Interactions (IEEE Transactions on Mobile Computing, 2022) <a href="https://doi.org/10.1109/TMC.2020.3029718">[link]</a>
- STEP: A Spatio-Temporal Fine-Granular User Traffic Prediction System for Cellular Networks (IEEE Transactions on Mobile Computing, 2021) <a href="https://doi.org/10.1109/TMC.2020.3001225">[link]</a>
- Graph Convolutional Recurrent Neural Networks for Water Demand Forecasting (Water Resources Research, 2022) <a href="https://doi.org/10.1029/2022WR032299">[link]</a>
- A symmetric adaptive visibility graph classification method of orthogonal signals for automatic modulation classification (IET Communications, 2023) <a href="https://doi.org/10.1049/cmu2.12608">[link]</a>
- Improved Pearson Correlation Coefficient-Based Graph Neural Network for Dynamic Soft Sensor of Polypropylene Industries (INDUSTRIAL & ENGINEERING CHEMISTRY RESEARCH, 2024) <a href="https://doi.org/10.1021/acs.iecr.4c02832?urlappend=%3Fref%3DPDF&jav=VoR&rel=cite-as">[link]</a>
- GNN-Based Network Traffic Analysis for the Detection of Sequential Attacks in IoT (Electronics (Switzerland), 2024) <a href="https://doi.org/10.3390/electronics13122274">[link]</a>
- Multi-Scenario Cellular KPI Prediction Based on Spatiotemporal Graph Neural Network (IEEE Transactions on Automation Science and Engineering, 2024) <a href="https://doi.org/10.1109/TASE.2024.3416952">[link]</a>
- Long Sequence Multivariate Time-Series Forecasting for Industrial Processes Using SASGNN (IEEE Transactions on Industrial Informatics, 2024) <a href="https://doi.org/10.1109/TII.2024.3424214">[link]</a>
- Automatic Modulation Recognition of Unknown Interference Signals Based on Graph Model (IEEE Wireless Communications Letters, 2024) <a href="https://doi.org/10.1109/LWC.2024.3401720">[link]</a>
- Spatial-Temporal Graph Model Based on Attention Mechanism for Anomalous IoT Intrusion Detection (IEEE Transactions on Industrial Informatics, 2024) <a href="https://doi.org/10.1109/TII.2023.3308784">[link]</a>
- DTSG-Net: Dynamic Time Series Graph Neural Network and It's Application in Modulation Recognition (IEEE Internet of Things Journal, 2024) <a href="http://dx.doi.org/10.1109/JIOT.2024.3514875">[link]</a>
- HDM-GNN: A Heterogeneous Dynamic Multi-view Graph Neural Network for Crime Prediction (ACM Trans. Sen. Netw., 2024) <a href="https://doi.org/10.1145/3665141">[link]</a>
- Mobile Traffic Prediction in Consumer Applications: A Multimodal Deep Learning Approach (IEEE Transactions on Consumer Electronics, 2024) <a href="https://doi.org/10.1109/TCE.2024.3361037">[link]</a>
- Talent Demand-Supply Joint Prediction with Dynamic Heterogeneous Graph Enhanced Meta-Learning (KDD, 2022) <a href="https://doi.org/10.1145/3534678.3539139">[link]</a>
- TFE-GNN: A Temporal Fusion Encoder Using Graph Neural Networks for Fine-grained Encrypted Traffic Classification (WWW, 2023) <a href="https://doi.org/10.1145/3543507.3583227">[link]</a>
- Telecommunication Traffic Forecasting via Multi-task Learning (WSDM, 2023) <a href="https://doi.org/10.1145/3539597.3570440">[link]</a>
- Multivariate and Propagation Graph Attention Network for Spatial-Temporal Prediction with Outdoor Cellular Traffic (CIKM, 2021) <a href="https://doi.org/10.1145/3459637.3482152">[link]</a>
- Spatio-Temporal Multi-graph Networks for Demand Forecasting in Online Marketplaces (ECML PKDD, 2021) <a href="https://doi.org/10.1007/978-3-030-86514-6_12">[link]</a>
- Twin Graph-Based Anomaly Detection via Attentive Multi-Modal Learning for Microservice System (ASE, 2024) <a href="https://doi.org/10.1109/ASE56229.2023.00138">[link]</a>
- Hierarchical Spatio-Temporal Graph Learning Based on Metapath Aggregation for Emergency Supply Forecasting (CIKM, 2024) <a href="https://doi.org/10.1145/3627673.3679854">[link]</a>
