static void ppc_cpu_unrealizefn(DeviceState *dev, Error **errp)

{

    PowerPCCPU *cpu = POWERPC_CPU(dev);

    CPUPPCState *env = &cpu->env;

    opc_handler_t **table;

    int i, j;



    cpu_exec_exit(CPU(dev));



    for (i = 0; i < PPC_CPU_OPCODES_LEN; i++) {

        if (env->opcodes[i] == &invalid_handler) {

            continue;

        }

        if (is_indirect_opcode(env->opcodes[i])) {

            table = ind_table(env->opcodes[i]);

            for (j = 0; j < PPC_CPU_INDIRECT_OPCODES_LEN; j++) {

                if (table[j] != &invalid_handler &&

                        is_indirect_opcode(table[j])) {

                    g_free((opc_handler_t *)((uintptr_t)table[j] &

                        ~PPC_INDIRECT));

                }

            }

            g_free((opc_handler_t *)((uintptr_t)env->opcodes[i] &

                ~PPC_INDIRECT));

        }

    }

}
