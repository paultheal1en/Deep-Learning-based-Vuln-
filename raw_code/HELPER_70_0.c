void HELPER(mtspr)(CPUOpenRISCState *env,

                   target_ulong ra, target_ulong rb, target_ulong offset)

{

#ifndef CONFIG_USER_ONLY

    int spr = (ra | offset);

    int idx;



    OpenRISCCPU *cpu = openrisc_env_get_cpu(env);

    CPUState *cs = CPU(cpu);



    switch (spr) {

    case TO_SPR(0, 0): /* VR */

        env->vr = rb;

        break;



    case TO_SPR(0, 16): /* NPC */

        env->npc = rb;

        break;



    case TO_SPR(0, 17): /* SR */

        if ((env->sr & (SR_IME | SR_DME | SR_SM)) ^

            (rb & (SR_IME | SR_DME | SR_SM))) {

            tlb_flush(env, 1);

        }

        env->sr = rb;

        env->sr |= SR_FO;      /* FO is const equal to 1 */

        if (env->sr & SR_DME) {

            env->tlb->cpu_openrisc_map_address_data =

                &cpu_openrisc_get_phys_data;

        } else {

            env->tlb->cpu_openrisc_map_address_data =

                &cpu_openrisc_get_phys_nommu;

        }



        if (env->sr & SR_IME) {

            env->tlb->cpu_openrisc_map_address_code =

                &cpu_openrisc_get_phys_code;

        } else {

            env->tlb->cpu_openrisc_map_address_code =

                &cpu_openrisc_get_phys_nommu;

        }

        break;



    case TO_SPR(0, 18): /* PPC */

        env->ppc = rb;

        break;



    case TO_SPR(0, 32): /* EPCR */

        env->epcr = rb;

        break;



    case TO_SPR(0, 48): /* EEAR */

        env->eear = rb;

        break;



    case TO_SPR(0, 64): /* ESR */

        env->esr = rb;

        break;

    case TO_SPR(1, 512) ... TO_SPR(1, 639): /* DTLBW0MR 0-127 */

        idx = spr - TO_SPR(1, 512);

        if (!(rb & 1)) {

            tlb_flush_page(env, env->tlb->dtlb[0][idx].mr & TARGET_PAGE_MASK);

        }

        env->tlb->dtlb[0][idx].mr = rb;

        break;



    case TO_SPR(1, 640) ... TO_SPR(1, 767): /* DTLBW0TR 0-127 */

        idx = spr - TO_SPR(1, 640);

        env->tlb->dtlb[0][idx].tr = rb;

        break;

    case TO_SPR(1, 768) ... TO_SPR(1, 895):   /* DTLBW1MR 0-127 */

    case TO_SPR(1, 896) ... TO_SPR(1, 1023):  /* DTLBW1TR 0-127 */

    case TO_SPR(1, 1024) ... TO_SPR(1, 1151): /* DTLBW2MR 0-127 */

    case TO_SPR(1, 1152) ... TO_SPR(1, 1279): /* DTLBW2TR 0-127 */

    case TO_SPR(1, 1280) ... TO_SPR(1, 1407): /* DTLBW3MR 0-127 */

    case TO_SPR(1, 1408) ... TO_SPR(1, 1535): /* DTLBW3TR 0-127 */

        break;

    case TO_SPR(2, 512) ... TO_SPR(2, 639):   /* ITLBW0MR 0-127 */

        idx = spr - TO_SPR(2, 512);

        if (!(rb & 1)) {

            tlb_flush_page(env, env->tlb->itlb[0][idx].mr & TARGET_PAGE_MASK);

        }

        env->tlb->itlb[0][idx].mr = rb;

        break;



    case TO_SPR(2, 640) ... TO_SPR(2, 767): /* ITLBW0TR 0-127 */

        idx = spr - TO_SPR(2, 640);

        env->tlb->itlb[0][idx].tr = rb;

        break;

    case TO_SPR(2, 768) ... TO_SPR(2, 895):   /* ITLBW1MR 0-127 */

    case TO_SPR(2, 896) ... TO_SPR(2, 1023):  /* ITLBW1TR 0-127 */

    case TO_SPR(2, 1024) ... TO_SPR(2, 1151): /* ITLBW2MR 0-127 */

    case TO_SPR(2, 1152) ... TO_SPR(2, 1279): /* ITLBW2TR 0-127 */

    case TO_SPR(2, 1280) ... TO_SPR(2, 1407): /* ITLBW3MR 0-127 */

    case TO_SPR(2, 1408) ... TO_SPR(2, 1535): /* ITLBW3TR 0-127 */

        break;

    case TO_SPR(9, 0):  /* PICMR */

        env->picmr |= rb;

        break;

    case TO_SPR(9, 2):  /* PICSR */

        env->picsr &= ~rb;

        break;

    case TO_SPR(10, 0): /* TTMR */

        {

            if ((env->ttmr & TTMR_M) ^ (rb & TTMR_M)) {

                switch (rb & TTMR_M) {

                case TIMER_NONE:

                    cpu_openrisc_count_stop(cpu);

                    break;

                case TIMER_INTR:

                case TIMER_SHOT:

                case TIMER_CONT:

                    cpu_openrisc_count_start(cpu);

                    break;

                default:

                    break;

                }

            }



            int ip = env->ttmr & TTMR_IP;



            if (rb & TTMR_IP) {    /* Keep IP bit.  */

                env->ttmr = (rb & ~TTMR_IP) | ip;

            } else {    /* Clear IP bit.  */

                env->ttmr = rb & ~TTMR_IP;

                cs->interrupt_request &= ~CPU_INTERRUPT_TIMER;

            }



            cpu_openrisc_timer_update(cpu);

        }

        break;



    case TO_SPR(10, 1): /* TTCR */

        env->ttcr = rb;

        if (env->ttmr & TIMER_NONE) {

            return;

        }

        cpu_openrisc_timer_update(cpu);

        break;

    default:



        break;

    }

#endif

}
