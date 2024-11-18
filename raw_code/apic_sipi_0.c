void apic_sipi(CPUState *env)

{

    APICState *s = env->apic_state;



    cpu_reset_interrupt(env, CPU_INTERRUPT_SIPI);



    if (!s->wait_for_sipi)

        return;



    env->eip = 0;

    cpu_x86_load_seg_cache(env, R_CS, s->sipi_vector << 8, s->sipi_vector << 12,

                           0xffff, 0);

    env->halted = 0;

    s->wait_for_sipi = 0;

}
