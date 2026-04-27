# EasyLocalization

easy localization lib





# DA3 VO


visual odometry using depthanything 3

* optical flow to get keyframes.
* run DA3 for all key frames if new keyframes (half DA3 input images size) added.


**Use optical-flow VO as the real-time frontend, and use DA3 as a delayed local-geometry backend.**
DA3 should generate high-quality depth/pose constraints per chunk; pose graph optimization will integrate those constraints globally.

DA3 is suitable for this because it predicts spatially consistent geometry from arbitrary visual inputs, with or without known camera poses, and the official DA3-Streaming code is explicitly designed for long videos through chunk streaming under limited GPU memory.


```mermaid
flowchart LR
    A[Video frames] --> B[Optical flow tracking]
    B --> C{Enough pixel motion?}
    C -- no --> B
    C -- yes --> D[Add keyframe]

    D --> E{Enough new keyframes?}
    E -- no --> B
    E -- yes --> F[Run DA3 on <br/>10-keyframe chunk]
    F --> H[Align new chunk to previous chunk<br/>using overlap keyframes]
    H --> I[Add relative pose / Sim3 edges<br>Pose graph optimization]
    I --> L[Update global keyframe poses<br>Fuse / update map]
```
