static void xtensa_sim_init(MachineState *machine)

{

    XtensaCPU *cpu = NULL;

    CPUXtensaState *env = NULL;

    ram_addr_t ram_size = machine->ram_size;

    const char *cpu_model = machine->cpu_model;

    const char *kernel_filename = machine->kernel_filename;

    int n;



    if (!cpu_model) {

        cpu_model = XTENSA_DEFAULT_CPU_MODEL;

    }



    for (n = 0; n < smp_cpus; n++) {

        cpu = XTENSA_CPU(cpu_generic_init(TYPE_XTENSA_CPU, cpu_model));

        if (cpu == NULL) {

            error_report("unable to find CPU definition '%s'",

                         cpu_model);

            exit(EXIT_FAILURE);

        }

        env = &cpu->env;



        env->sregs[PRID] = n;

        qemu_register_reset(sim_reset, cpu);

        /* Need MMU initialized prior to ELF loading,

         * so that ELF gets loaded into virtual addresses

         */

        sim_reset(cpu);

    }



    if (env) {

        XtensaMemory sysram = env->config->sysram;



        sysram.location[0].size = ram_size;

        xtensa_create_memory_regions(&env->config->instrom, "xtensa.instrom");

        xtensa_create_memory_regions(&env->config->instram, "xtensa.instram");

        xtensa_create_memory_regions(&env->config->datarom, "xtensa.datarom");

        xtensa_create_memory_regions(&env->config->dataram, "xtensa.dataram");

        xtensa_create_memory_regions(&env->config->sysrom, "xtensa.sysrom");

        xtensa_create_memory_regions(&sysram, "xtensa.sysram");

    }



    if (serial_hds[0]) {

        xtensa_sim_open_console(serial_hds[0]);

    }

    if (kernel_filename) {

        uint64_t elf_entry;

        uint64_t elf_lowaddr;

#ifdef TARGET_WORDS_BIGENDIAN

        int success = load_elf(kernel_filename, translate_phys_addr, cpu,

                &elf_entry, &elf_lowaddr, NULL, 1, EM_XTENSA, 0, 0);

#else

        int success = load_elf(kernel_filename, translate_phys_addr, cpu,

                &elf_entry, &elf_lowaddr, NULL, 0, EM_XTENSA, 0, 0);

#endif

        if (success > 0) {

            env->pc = elf_entry;

        }

    }

}
