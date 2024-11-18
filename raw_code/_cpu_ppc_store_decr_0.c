static inline void _cpu_ppc_store_decr(PowerPCCPU *cpu, uint32_t decr,

                                       uint32_t value, int is_excp)

{

    ppc_tb_t *tb_env = cpu->env.tb_env;



    __cpu_ppc_store_decr(cpu, &tb_env->decr_next, tb_env->decr_timer,

                         &cpu_ppc_decr_excp, decr, value, is_excp);

}
