void cpu_x86_update_dr7(CPUX86State *env, uint32_t new_dr7)

{

    int i;



    for (i = 0; i < DR7_MAX_BP; i++) {

        hw_breakpoint_remove(env, i);

    }

    env->dr[7] = new_dr7;

    for (i = 0; i < DR7_MAX_BP; i++) {

        hw_breakpoint_insert(env, i);

    }

}
