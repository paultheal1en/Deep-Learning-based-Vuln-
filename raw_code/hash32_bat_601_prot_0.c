static int hash32_bat_601_prot(CPUPPCState *env,

                               target_ulong batu, target_ulong batl)

{

    int key, pp;



    pp = batu & BATU32_601_PP;

    if (msr_pr == 0) {

        key = !!(batu & BATU32_601_KS);

    } else {

        key = !!(batu & BATU32_601_KP);

    }

    return ppc_hash32_pp_check(key, pp, 0);

}
