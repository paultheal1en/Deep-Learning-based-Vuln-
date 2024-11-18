void helper_movl_drN_T0(CPUX86State *env, int reg, target_ulong t0)

{

#ifndef CONFIG_USER_ONLY

    if (reg < 4) {

        if (hw_breakpoint_enabled(env->dr[7], reg)

            && hw_breakpoint_type(env->dr[7], reg) != DR7_TYPE_IO_RW) {

            hw_breakpoint_remove(env, reg);

            env->dr[reg] = t0;

            hw_breakpoint_insert(env, reg);

        } else {

            env->dr[reg] = t0;

        }

    } else if (reg == 7) {

        cpu_x86_update_dr7(env, t0);

    } else {

        env->dr[reg] = t0;

    }

#endif

}
