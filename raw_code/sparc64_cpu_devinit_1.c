SPARCCPU *sparc64_cpu_devinit(const char *cpu_model,

                              const char *default_cpu_model, uint64_t prom_addr)

{

    SPARCCPU *cpu;

    CPUSPARCState *env;

    ResetData *reset_info;



    uint32_t   tick_frequency = 100 * 1000000;

    uint32_t  stick_frequency = 100 * 1000000;

    uint32_t hstick_frequency = 100 * 1000000;



    if (cpu_model == NULL) {

        cpu_model = default_cpu_model;

    }

    cpu = SPARC_CPU(cpu_generic_init(TYPE_SPARC_CPU, cpu_model));

    if (cpu == NULL) {

        fprintf(stderr, "Unable to find Sparc CPU definition\n");

        exit(1);

    }

    env = &cpu->env;



    env->tick = cpu_timer_create("tick", cpu, tick_irq,

                                  tick_frequency, TICK_INT_DIS,

                                  TICK_NPT_MASK);



    env->stick = cpu_timer_create("stick", cpu, stick_irq,

                                   stick_frequency, TICK_INT_DIS,

                                   TICK_NPT_MASK);



    env->hstick = cpu_timer_create("hstick", cpu, hstick_irq,

                                    hstick_frequency, TICK_INT_DIS,

                                    TICK_NPT_MASK);



    reset_info = g_malloc0(sizeof(ResetData));

    reset_info->cpu = cpu;

    reset_info->prom_addr = prom_addr;

    qemu_register_reset(main_cpu_reset, reset_info);



    return cpu;

}
