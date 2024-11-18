uint32_t cpu_mips_get_count (CPUState *env)

{

    if (env->CP0_Cause & (1 << CP0Ca_DC)) {

        return env->CP0_Count;

    } else {

        return env->CP0_Count +

            (uint32_t)muldiv64(qemu_get_clock(vm_clock),

                               TIMER_FREQ, get_ticks_per_sec());

    }

}
