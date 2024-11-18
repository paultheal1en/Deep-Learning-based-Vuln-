void helper_rfi(CPUPPCState *env)

{

    do_rfi(env, env->spr[SPR_SRR0], env->spr[SPR_SRR1],

           ~((target_ulong)0x783F0000), 1);

}
