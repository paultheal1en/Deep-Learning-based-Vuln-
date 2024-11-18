static void cpu_devinit(const char *cpu_model, unsigned int id,

                        uint64_t prom_addr, qemu_irq **cpu_irqs)

{

    CPUState *cs;

    SPARCCPU *cpu;

    CPUSPARCState *env;



    cpu = SPARC_CPU(cpu_generic_init(TYPE_SPARC_CPU, cpu_model));

    if (cpu == NULL) {

        fprintf(stderr, "qemu: Unable to find Sparc CPU definition\n");

        exit(1);

    }

    env = &cpu->env;



    cpu_sparc_set_id(env, id);

    if (id == 0) {

        qemu_register_reset(main_cpu_reset, cpu);

    } else {

        qemu_register_reset(secondary_cpu_reset, cpu);

        cs = CPU(cpu);

        cs->halted = 1;

    }

    *cpu_irqs = qemu_allocate_irqs(cpu_set_irq, cpu, MAX_PILS);

    env->prom_addr = prom_addr;

}
