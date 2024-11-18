void helper_movl_crN_T0(int reg)

{

    env->cr[reg] = T0;

    switch(reg) {

    case 0:

        cpu_x86_update_cr0(env);

        break;

    case 3:

        cpu_x86_update_cr3(env);

        break;

    }

}
