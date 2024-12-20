static void add_qemu_cpu_model_features(S390FeatBitmap fbm)

{

    static const int feats[] = {

        S390_FEAT_DAT_ENH,

        S390_FEAT_IDTE_SEGMENT,

        S390_FEAT_STFLE,

        S390_FEAT_SENSE_RUNNING_STATUS,

        S390_FEAT_EXTENDED_IMMEDIATE,

        S390_FEAT_EXTENDED_TRANSLATION_2,

        S390_FEAT_MSA,

        S390_FEAT_EXTENDED_TRANSLATION_3,

        S390_FEAT_LONG_DISPLACEMENT,

        S390_FEAT_LONG_DISPLACEMENT_FAST,

        S390_FEAT_ETF2_ENH,

        S390_FEAT_STORE_CLOCK_FAST,

        S390_FEAT_MOVE_WITH_OPTIONAL_SPEC,

        S390_FEAT_ETF3_ENH,


        S390_FEAT_COMPARE_AND_SWAP_AND_STORE,

        S390_FEAT_COMPARE_AND_SWAP_AND_STORE_2,

        S390_FEAT_GENERAL_INSTRUCTIONS_EXT,

        S390_FEAT_EXECUTE_EXT,

        S390_FEAT_FLOATING_POINT_SUPPPORT_ENH,

        S390_FEAT_STFLE_45,

        S390_FEAT_STFLE_49,

        S390_FEAT_LOCAL_TLB_CLEARING,

        S390_FEAT_INTERLOCKED_ACCESS_2,

        S390_FEAT_STFLE_53,

        S390_FEAT_MSA_EXT_5,

        S390_FEAT_MSA_EXT_3,

        S390_FEAT_MSA_EXT_4,

    };

    int i;



    for (i = 0; i < ARRAY_SIZE(feats); i++) {

        set_bit(feats[i], fbm);

    }

}