void cpu_loop_exit(CPUState *env1)

{

    env1->current_tb = NULL;

    longjmp(env1->jmp_env, 1);

}
