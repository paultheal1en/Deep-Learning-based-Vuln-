int av_get_cpu_flags(void)

{

    if (checked)

        return flags;



    if (ARCH_AARCH64)

        flags = ff_get_cpu_flags_aarch64();

    if (ARCH_ARM)

        flags = ff_get_cpu_flags_arm();

    if (ARCH_PPC)

        flags = ff_get_cpu_flags_ppc();

    if (ARCH_X86)

        flags = ff_get_cpu_flags_x86();



    checked = 1;

    return flags;

}
