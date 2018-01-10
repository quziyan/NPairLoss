# NPairLoss
Follow the paper &lt;Improved Deep Metric Learning with Multi-class N-pair Loss Objective>


PROTO DEFINITION
```
optional NPairLossParameter npair_loss_param = 8866720;
message NPairLossParameter {
    # The margin-offset used for positive sample mining
    optional float margin_ident = 1 [default = 0];
    # The margin-offset used for negative sample mining
    optional float margin_diff = 8 [default = 0];
    # for RELATIVE-MINING-METHOD
    # the ident/diff for positive/negative sample selection
    # These modes first sort the positive/negative sample distances,
    # if the number SN is greater than 0, select the bigest INT(SN) sample pairs in MINING-REGION for LOSS-calculation;
    # if the number SN is smaller than 0, select the bigest INT(SIZE(MINING-REGION)*(-SN)) sample pairs in MINING-REGION for LOSS-calculation
    optional float identsn = 2 [default = -1];
    optional float diffsn = 3 [default = -1];
    # indicate the mining region whether GLOBAL-WISE OR QUERY-WISE
    enum MiningRegion {
        GLOBAL = 0;
        LOCAL = 1;
    }
    enum MiningMethod {
        HARD = 0;
        EASY = 1;
        RAND = 2;
        RELATIVE_HARD = 3;
        RELATIVE_EASY= 4;
    }
    optional MiningRegion ap_mining_region = 4 [default = LOCAL];
    optional MiningMethod ap_mining_method = 5 [default = RAND];
    optional MiningRegion an_mining_region = 6 [default = LOCAL];
    optional MiningMethod an_mining_method = 7 [default = RAND];
}
```

USAGE
```

layer {
    bottom: "pool5/7x7_s1"
    name: "loss3/pool5/7x7_s1/norm"
    type: "L2Normalize"
    top: "loss3/pool5/7x7_s1/norm"
}
layer {
    bottom: "loss3/pool5/7x7_s1/norm"
    #bottom: "pool5/7x7_s1"
    bottom: "label_type_mb"
    name: "loss3/type_mb"
    type: "NPairMultiClassLoss"
    top: "loss3/type_npair_mc"
    top: "loss3/type_npair_mc_retrieve_top1"
    top: "loss3/type_npair_mc_retrieve_top5"
    top: "loss3/type_npair_mc_retrieve_top10"
    top: "loss3/feature_asum"
    loss_weight: 1
    loss_weight: 1
    loss_weight: 1
    loss_weight: 1
    loss_weight: 1
    npair_loss_param {
        margin_ident: 0.0
        margin_diff: -0.05
        identsn: -0.0
        diffsn: -0.3 #对于绝对选择来说该项无效
        ap_mining_region: GLOBAL
        ap_mining_method: RELATIVE_HARD
        an_mining_region: LOCAL
        an_mining_method: HARD # RELATIVE_HARD
    }
    #loss_weight: 1
    #include {
    #    phase: TRAIN
    #}
} 
```
