static always_inline void _cpu_ppc_store_hdecr (CPUState *env, uint32_t hdecr,

                                                uint32_t value, int is_excp)

{

    ppc_tb_t *tb_env = env->tb_env;



    __cpu_ppc_store_decr(env, &tb_env->hdecr_next, tb_env->hdecr_timer,

                         &cpu_ppc_hdecr_excp, hdecr, value, is_excp);

}
