void helper_rfci(CPUPPCState *env)

{

    do_rfi(env, env->spr[SPR_BOOKE_CSRR0], SPR_BOOKE_CSRR1,

           ~((target_ulong)0x3FFF0000), 0);

}
