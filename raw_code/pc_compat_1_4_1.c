static void pc_compat_1_4(QEMUMachineInitArgs *args)

{

    pc_compat_1_5(args);

    has_pvpanic = false;

    x86_cpu_compat_set_features("n270", FEAT_1_ECX, 0, CPUID_EXT_MOVBE);

    x86_cpu_compat_set_features("Westmere", FEAT_1_ECX, 0, CPUID_EXT_PCLMULQDQ);

}
