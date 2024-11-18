void cpu_dump_state(CPUState *env, FILE *f, fprintf_function cpu_fprintf,

        int flags)

{

    int i;



    cpu_fprintf(f, "PC=%08x\n", env->pc);



    for (i = 0; i < 16; ++i) {

        cpu_fprintf(f, "A%02d=%08x%c", i, env->regs[i],

                (i % 4) == 3 ? '\n' : ' ');

    }

}
