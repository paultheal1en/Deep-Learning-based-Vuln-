static void aarch64_any_initfn(Object *obj)

{

    ARMCPU *cpu = ARM_CPU(obj);



    set_feature(&cpu->env, ARM_FEATURE_V8);

    set_feature(&cpu->env, ARM_FEATURE_VFP4);

    set_feature(&cpu->env, ARM_FEATURE_NEON);

    set_feature(&cpu->env, ARM_FEATURE_AARCH64);

    set_feature(&cpu->env, ARM_FEATURE_V8_AES);

    set_feature(&cpu->env, ARM_FEATURE_V8_SHA1);

    set_feature(&cpu->env, ARM_FEATURE_V8_SHA256);

    set_feature(&cpu->env, ARM_FEATURE_V8_PMULL);

    set_feature(&cpu->env, ARM_FEATURE_CRC);

    cpu->ctr = 0x80030003; /* 32 byte I and D cacheline size, VIPT icache */

    cpu->dcz_blocksize = 7; /*  512 bytes */

}
