static int vapic_enable(VAPICROMState *s, CPUX86State *env)

{

    int cpu_number = get_kpcr_number(env);

    target_phys_addr_t vapic_paddr;

    static const uint8_t enabled = 1;



    if (cpu_number < 0) {

        return -1;

    }

    vapic_paddr = s->vapic_paddr +

        (((target_phys_addr_t)cpu_number) << VAPIC_CPU_SHIFT);

    cpu_physical_memory_rw(vapic_paddr + offsetof(VAPICState, enabled),

                           (void *)&enabled, sizeof(enabled), 1);

    apic_enable_vapic(env->apic_state, vapic_paddr);



    s->state = VAPIC_ACTIVE;



    return 0;

}
