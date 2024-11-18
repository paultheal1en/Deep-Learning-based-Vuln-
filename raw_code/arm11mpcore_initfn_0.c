static void arm11mpcore_initfn(Object *obj)

{

    ARMCPU *cpu = ARM_CPU(obj);

    set_feature(&cpu->env, ARM_FEATURE_V6K);

    set_feature(&cpu->env, ARM_FEATURE_VFP);

    set_feature(&cpu->env, ARM_FEATURE_VAPA);

    cpu->midr = ARM_CPUID_ARM11MPCORE;

    cpu->reset_fpsid = 0x410120b4;

    cpu->mvfr0 = 0x11111111;

    cpu->mvfr1 = 0x00000000;

    cpu->ctr = 0x1dd20d2;

    cpu->id_pfr0 = 0x111;

    cpu->id_pfr1 = 0x1;

    cpu->id_dfr0 = 0;

    cpu->id_afr0 = 0x2;

    cpu->id_mmfr0 = 0x01100103;

    cpu->id_mmfr1 = 0x10020302;

    cpu->id_mmfr2 = 0x01222000;

    cpu->id_isar0 = 0x00100011;

    cpu->id_isar1 = 0x12002111;

    cpu->id_isar2 = 0x11221011;

    cpu->id_isar3 = 0x01102131;

    cpu->id_isar4 = 0x141;

}
