int cpu_is_bsp(CPUX86State *env)

{

    /* We hard-wire the BSP to the first CPU. */

    return env->cpu_index == 0;

}
