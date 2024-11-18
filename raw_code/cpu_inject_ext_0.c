void cpu_inject_ext(S390CPU *cpu, uint32_t code, uint32_t param,

                    uint64_t param64)

{

    CPUS390XState *env = &cpu->env;



    if (env->ext_index == MAX_EXT_QUEUE - 1) {

        /* ugh - can't queue anymore. Let's drop. */

        return;

    }



    env->ext_index++;

    assert(env->ext_index < MAX_EXT_QUEUE);



    env->ext_queue[env->ext_index].code = code;

    env->ext_queue[env->ext_index].param = param;

    env->ext_queue[env->ext_index].param64 = param64;



    env->pending_int |= INTERRUPT_EXT;

    cpu_interrupt(CPU(cpu), CPU_INTERRUPT_HARD);

}
