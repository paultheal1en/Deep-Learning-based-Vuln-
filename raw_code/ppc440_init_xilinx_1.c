static PowerPCCPU *ppc440_init_xilinx(ram_addr_t *ram_size,

                                      int do_init,

                                      const char *cpu_model,

                                      uint32_t sysclk)

{

    PowerPCCPU *cpu;

    CPUPPCState *env;

    qemu_irq *irqs;



    cpu = POWERPC_CPU(cpu_generic_init(TYPE_POWERPC_CPU, cpu_model));

    if (cpu == NULL) {

        fprintf(stderr, "Unable to initialize CPU!\n");

        exit(1);

    }

    env = &cpu->env;



    ppc_booke_timers_init(cpu, sysclk, 0/* no flags */);



    ppc_dcr_init(env, NULL, NULL);



    /* interrupt controller */

    irqs = g_malloc0(sizeof(qemu_irq) * PPCUIC_OUTPUT_NB);

    irqs[PPCUIC_OUTPUT_INT] = ((qemu_irq *)env->irq_inputs)[PPC40x_INPUT_INT];

    irqs[PPCUIC_OUTPUT_CINT] = ((qemu_irq *)env->irq_inputs)[PPC40x_INPUT_CINT];

    ppcuic_init(env, irqs, 0x0C0, 0, 1);

    return cpu;

}
