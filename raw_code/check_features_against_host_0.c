static int check_features_against_host(x86_def_t *guest_def)

{

    x86_def_t host_def;

    uint32_t mask;

    int rv, i;

    struct model_features_t ft[] = {

        {&guest_def->features, &host_def.features,

            ~0, feature_name, 0x00000000},

        {&guest_def->ext_features, &host_def.ext_features,

            ~CPUID_EXT_HYPERVISOR, ext_feature_name, 0x00000001},

        {&guest_def->ext2_features, &host_def.ext2_features,

            ~PPRO_FEATURES, ext2_feature_name, 0x80000000},

        {&guest_def->ext3_features, &host_def.ext3_features,

            ~CPUID_EXT3_SVM, ext3_feature_name, 0x80000001}};



    cpu_x86_fill_host(&host_def);

    for (rv = 0, i = 0; i < ARRAY_SIZE(ft); ++i)

        for (mask = 1; mask; mask <<= 1)

            if (ft[i].check_feat & mask && *ft[i].guest_feat & mask &&

                !(*ft[i].host_feat & mask)) {

                    unavailable_host_feature(&ft[i], mask);

                    rv = 1;

                }

    return rv;

}
