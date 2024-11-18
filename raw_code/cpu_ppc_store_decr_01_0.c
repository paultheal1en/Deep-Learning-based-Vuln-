void cpu_ppc_store_decr (CPUPPCState *env, uint32_t value)

{

    PowerPCCPU *cpu = ppc_env_get_cpu(env);



    _cpu_ppc_store_decr(cpu, cpu_ppc_load_decr(env), value, 0);

}
