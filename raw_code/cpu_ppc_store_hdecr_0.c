void cpu_ppc_store_hdecr (CPUPPCState *env, uint32_t value)

{

    PowerPCCPU *cpu = ppc_env_get_cpu(env);



    _cpu_ppc_store_hdecr(cpu, cpu_ppc_load_hdecr(env), value, 0);

}
