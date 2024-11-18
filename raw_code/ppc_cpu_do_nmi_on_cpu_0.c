static void ppc_cpu_do_nmi_on_cpu(CPUState *cs, run_on_cpu_data arg)

{

    cpu_synchronize_state(cs);

    ppc_cpu_do_system_reset(cs);

}
