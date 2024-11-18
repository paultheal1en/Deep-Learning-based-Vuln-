static void mmu6xx_dump_mmu(FILE *f, fprintf_function cpu_fprintf,

                            CPUPPCState *env)

{

    ppc6xx_tlb_t *tlb;

    target_ulong sr;

    int type, way, entry, i;



    cpu_fprintf(f, "HTAB base = 0x%"HWADDR_PRIx"\n", env->htab_base);

    cpu_fprintf(f, "HTAB mask = 0x%"HWADDR_PRIx"\n", env->htab_mask);



    cpu_fprintf(f, "\nSegment registers:\n");

    for (i = 0; i < 32; i++) {

        sr = env->sr[i];

        if (sr & 0x80000000) {

            cpu_fprintf(f, "%02d T=%d Ks=%d Kp=%d BUID=0x%03x "

                        "CNTLR_SPEC=0x%05x\n", i,

                        sr & 0x80000000 ? 1 : 0, sr & 0x40000000 ? 1 : 0,

                        sr & 0x20000000 ? 1 : 0, (uint32_t)((sr >> 20) & 0x1FF),

                        (uint32_t)(sr & 0xFFFFF));

        } else {

            cpu_fprintf(f, "%02d T=%d Ks=%d Kp=%d N=%d VSID=0x%06x\n", i,

                        sr & 0x80000000 ? 1 : 0, sr & 0x40000000 ? 1 : 0,

                        sr & 0x20000000 ? 1 : 0, sr & 0x10000000 ? 1 : 0,

                        (uint32_t)(sr & 0x00FFFFFF));

        }

    }



    cpu_fprintf(f, "\nBATs:\n");

    mmu6xx_dump_BATs(f, cpu_fprintf, env, ACCESS_INT);

    mmu6xx_dump_BATs(f, cpu_fprintf, env, ACCESS_CODE);



    if (env->id_tlbs != 1) {

        cpu_fprintf(f, "ERROR: 6xx MMU should have separated TLB"

                    " for code and data\n");

    }



    cpu_fprintf(f, "\nTLBs                       [EPN    EPN + SIZE]\n");



    for (type = 0; type < 2; type++) {

        for (way = 0; way < env->nb_ways; way++) {

            for (entry = env->nb_tlb * type + env->tlb_per_way * way;

                 entry < (env->nb_tlb * type + env->tlb_per_way * (way + 1));

                 entry++) {



                tlb = &env->tlb.tlb6[entry];

                cpu_fprintf(f, "%s TLB %02d/%02d way:%d %s ["

                            TARGET_FMT_lx " " TARGET_FMT_lx "]\n",

                            type ? "code" : "data", entry % env->nb_tlb,

                            env->nb_tlb, way,

                            pte_is_valid(tlb->pte0) ? "valid" : "inval",

                            tlb->EPN, tlb->EPN + TARGET_PAGE_SIZE);

            }

        }

    }

}
