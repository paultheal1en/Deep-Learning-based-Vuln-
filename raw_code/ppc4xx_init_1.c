PowerPCCPU *ppc4xx_init(const char *cpu_model,

                        clk_setup_t *cpu_clk, clk_setup_t *tb_clk,

                        uint32_t sysclk)

{

    PowerPCCPU *cpu;

    CPUPPCState *env;



    /* init CPUs */

    cpu = POWERPC_CPU(cpu_generic_init(TYPE_POWERPC_CPU, cpu_model));

    if (cpu == NULL) {

        fprintf(stderr, "Unable to find PowerPC %s CPU definition\n",

                cpu_model);

        exit(1);

    }

    env = &cpu->env;



    cpu_clk->cb = NULL; /* We don't care about CPU clock frequency changes */

    cpu_clk->opaque = env;

    /* Set time-base frequency to sysclk */

    tb_clk->cb = ppc_40x_timers_init(env, sysclk, PPC_INTERRUPT_PIT);

    tb_clk->opaque = env;

    ppc_dcr_init(env, NULL, NULL);

    /* Register qemu callbacks */

    qemu_register_reset(ppc4xx_reset, cpu);



    return cpu;

}
