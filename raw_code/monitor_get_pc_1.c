static target_long monitor_get_pc (const struct MonitorDef *md, int val)

{

    CPUState *env = mon_get_cpu();

    if (!env)

        return 0;

    return env->eip + env->segs[R_CS].base;

}
