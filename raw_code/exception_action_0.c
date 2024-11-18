static void exception_action(CPUState *cpu)

{

#if defined(TARGET_I386)

    X86CPU *x86_cpu = X86_CPU(cpu);

    CPUX86State *env1 = &x86_cpu->env;



    raise_exception_err(env1, cpu->exception_index, env1->error_code);

#else

    cpu_loop_exit(cpu);

#endif

}
