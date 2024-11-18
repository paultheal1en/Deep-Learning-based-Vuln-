static void imx6_defer_clear_reset_bit(int cpuid,

                                       IMX6SRCState *s,

                                       unsigned long reset_shift)

{

    struct SRCSCRResetInfo *ri;



    ri = g_malloc(sizeof(struct SRCSCRResetInfo));

    ri->s = s;

    ri->reset_bit = reset_shift;



    async_run_on_cpu(arm_get_cpu_by_id(cpuid), imx6_clear_reset_bit,

                     RUN_ON_CPU_HOST_PTR(ri));

}
