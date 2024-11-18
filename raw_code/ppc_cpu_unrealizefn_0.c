static void ppc_cpu_unrealizefn(DeviceState *dev, Error **errp)

{

    PowerPCCPU *cpu = POWERPC_CPU(dev);

    CPUPPCState *env = &cpu->env;

    int i;



    for (i = 0; i < PPC_CPU_OPCODES_LEN; i++) {

        if (env->opcodes[i] != &invalid_handler) {

            g_free(env->opcodes[i]);

        }

    }

}
