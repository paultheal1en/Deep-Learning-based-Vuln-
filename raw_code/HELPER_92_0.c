target_ulong HELPER(mfspr)(CPUOpenRISCState *env,

                           target_ulong rd, target_ulong ra, uint32_t offset)

{

#ifndef CONFIG_USER_ONLY

    int spr = (ra | offset);

    int idx;



    OpenRISCCPU *cpu = openrisc_env_get_cpu(env);



    switch (spr) {

    case TO_SPR(0, 0): /* VR */

        return env->vr & SPR_VR;



    case TO_SPR(0, 1): /* UPR */

        return env->upr;    /* TT, DM, IM, UP present */



    case TO_SPR(0, 2): /* CPUCFGR */

        return env->cpucfgr;



    case TO_SPR(0, 3): /* DMMUCFGR */

        return env->dmmucfgr;    /* 1Way, 64 entries */



    case TO_SPR(0, 4): /* IMMUCFGR */

        return env->immucfgr;



    case TO_SPR(0, 16): /* NPC */

        return env->npc;



    case TO_SPR(0, 17): /* SR */

        return env->sr;



    case TO_SPR(0, 18): /* PPC */

        return env->ppc;



    case TO_SPR(0, 32): /* EPCR */

        return env->epcr;



    case TO_SPR(0, 48): /* EEAR */

        return env->eear;



    case TO_SPR(0, 64): /* ESR */

        return env->esr;



    case TO_SPR(1, 512) ... TO_SPR(1, 639): /* DTLBW0MR 0-127 */

        idx = spr - TO_SPR(1, 512);

        return env->tlb->dtlb[0][idx].mr;



    case TO_SPR(1, 640) ... TO_SPR(1, 767): /* DTLBW0TR 0-127 */

        idx = spr - TO_SPR(1, 640);

        return env->tlb->dtlb[0][idx].tr;



    case TO_SPR(1, 768) ... TO_SPR(1, 895):   /* DTLBW1MR 0-127 */

    case TO_SPR(1, 896) ... TO_SPR(1, 1023):  /* DTLBW1TR 0-127 */

    case TO_SPR(1, 1024) ... TO_SPR(1, 1151): /* DTLBW2MR 0-127 */

    case TO_SPR(1, 1152) ... TO_SPR(1, 1279): /* DTLBW2TR 0-127 */

    case TO_SPR(1, 1280) ... TO_SPR(1, 1407): /* DTLBW3MR 0-127 */

    case TO_SPR(1, 1408) ... TO_SPR(1, 1535): /* DTLBW3TR 0-127 */

        break;



    case TO_SPR(2, 512) ... TO_SPR(2, 639): /* ITLBW0MR 0-127 */

        idx = spr - TO_SPR(2, 512);

        return env->tlb->itlb[0][idx].mr;



    case TO_SPR(2, 640) ... TO_SPR(2, 767): /* ITLBW0TR 0-127 */

        idx = spr - TO_SPR(2, 640);

        return env->tlb->itlb[0][idx].tr;



    case TO_SPR(2, 768) ... TO_SPR(2, 895):   /* ITLBW1MR 0-127 */

    case TO_SPR(2, 896) ... TO_SPR(2, 1023):  /* ITLBW1TR 0-127 */

    case TO_SPR(2, 1024) ... TO_SPR(2, 1151): /* ITLBW2MR 0-127 */

    case TO_SPR(2, 1152) ... TO_SPR(2, 1279): /* ITLBW2TR 0-127 */

    case TO_SPR(2, 1280) ... TO_SPR(2, 1407): /* ITLBW3MR 0-127 */

    case TO_SPR(2, 1408) ... TO_SPR(2, 1535): /* ITLBW3TR 0-127 */

        break;



    case TO_SPR(9, 0):  /* PICMR */

        return env->picmr;



    case TO_SPR(9, 2):  /* PICSR */

        return env->picsr;



    case TO_SPR(10, 0): /* TTMR */

        return env->ttmr;



    case TO_SPR(10, 1): /* TTCR */

        cpu_openrisc_count_update(cpu);

        return env->ttcr;



    default:

        break;

    }

#endif



/*If we later need to add tracepoints (or debug printfs) for the return

value, it may be useful to structure the code like this:



target_ulong ret = 0;



switch() {

case x:

 ret = y;

 break;

case z:

 ret = 42;

 break;

...

}



later something like trace_spr_read(ret);



return ret;*/



    /* for rd is passed in, if rd unchanged, just keep it back.  */

    return rd;

}
