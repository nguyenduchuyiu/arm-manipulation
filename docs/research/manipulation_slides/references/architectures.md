# Editable architecture sketches

These Mermaid diagrams supplement the rendered vector figures. Model blocks are named **FM**; its weight initialization is explained below each relevant diagram.

## Coordinate-conditioned reaching

```mermaid
flowchart LR
    G["3D goal"] --> E["Goal encoder"]
    E --> F["FM"]
    Q["Joint and gripper state"] --> S["State encoder"]
    S --> F
    N["Noise and flow time"] --> F
    F --> A["EE delta trajectory"]
    A --> K["IK and servo"]
    K --> J["Joint commands"]
```

FM is trained from random weights on validated reaching demonstrations.

## Visual behavior cloning

```mermaid
flowchart LR
    I["Top RGB"] --> V["ViT, frozen"]
    V --> D["Target-query decoder"]
    T["Task ID"] --> D
    D --> H["Target latent"]
    H --> F["FM"]
    Q["Joint and gripper state"] --> F
    F --> A["EE deltas, then IK"]
```

The FM uses weights trained on reaching; the visual conditioning modules are new. The ViT remains frozen while the conditioning modules and FM are fine-tuned.

## Visual behavior cloning with auxiliary XYZ

```mermaid
flowchart LR
    I["Top RGB"] --> V["ViT, frozen"]
    V --> D["Target query and shared latent"]
    T["Task ID"] --> D
    D --> F["FM"]
    Q["Joint and gripper state"] --> F
    F --> A["EE deltas, then IK"]
    D -.-> X["XYZ head"]
    X -.-> L["Auxiliary XYZ loss"]
    Y["Target-center training label"] -.-> L
```

XYZ is a training-only loss branch. It does not supply numerical coordinates to FM. The random-weight ablation uses this same topology but initializes FM randomly; action/state normalization is kept fixed.

## Point-prompted TCOW adaptation

```mermaid
flowchart LR
    I["Top RGB history"] --> T["Causal TCOW"]
    P["Point prompt in first frame"] --> T
    T --> H["Target representation"]
    H --> X["XYZ head"]
    X --> O["Predicted target center"]
```

TCOW starts from its original official RGB/query-channel weights. The original segmentation outputs are replaced by an object-center head. The deck includes a separately labeled qualitative video from the earlier e100 pilot at step 1,250.

## Joint TCOW and FM

```mermaid
flowchart LR
    I["Top RGB history"] --> T["Causal TCOW"]
    P["Initial point prompt"] --> T
    T --> H["Current target latent"]
    H --> G["Goal projection"]
    G --> F["FM"]
    Q["Joint and gripper state"] --> F
    F --> A["EE deltas, then IK"]
    H -.-> X["XYZ head"]
    X -.-> L["Auxiliary XYZ loss"]
    Y["Target-center training label"] -.-> L
```

FM uses weights trained on reaching, motivated by the 10-epoch visual BC initialization ablation. During expert segments, TCOW and FM are fine-tuned together. Before expert actions exist, only the tracker/XYZ branch is updated. Final manipulation results remain blank.
