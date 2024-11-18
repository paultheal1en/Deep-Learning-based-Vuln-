void ppc_tlb_invalidate_one(CPUPPCState *env, target_ulong addr)

{

#if !defined(FLUSH_ALL_TLBS)

    PowerPCCPU *cpu = ppc_env_get_cpu(env);

    CPUState *cs;



    addr &= TARGET_PAGE_MASK;

    switch (env->mmu_model) {

    case POWERPC_MMU_SOFT_6xx:

    case POWERPC_MMU_SOFT_74xx:

        ppc6xx_tlb_invalidate_virt(env, addr, 0);

        if (env->id_tlbs == 1) {

            ppc6xx_tlb_invalidate_virt(env, addr, 1);

        }

        break;

    case POWERPC_MMU_32B:

    case POWERPC_MMU_601:

        /* tlbie invalidate TLBs for all segments */

        addr &= ~((target_ulong)-1ULL << 28);

        cs = CPU(cpu);

        /* XXX: this case should be optimized,

         * giving a mask to tlb_flush_page

         */

        /* This is broken, some CPUs invalidate a whole congruence

         * class on an even smaller subset of bits and some OSes take

         * advantage of this. Just blow the whole thing away.

         */

#if 0

        tlb_flush_page(cs, addr | (0x0 << 28));

        tlb_flush_page(cs, addr | (0x1 << 28));

        tlb_flush_page(cs, addr | (0x2 << 28));

        tlb_flush_page(cs, addr | (0x3 << 28));

        tlb_flush_page(cs, addr | (0x4 << 28));

        tlb_flush_page(cs, addr | (0x5 << 28));

        tlb_flush_page(cs, addr | (0x6 << 28));

        tlb_flush_page(cs, addr | (0x7 << 28));

        tlb_flush_page(cs, addr | (0x8 << 28));

        tlb_flush_page(cs, addr | (0x9 << 28));

        tlb_flush_page(cs, addr | (0xA << 28));

        tlb_flush_page(cs, addr | (0xB << 28));

        tlb_flush_page(cs, addr | (0xC << 28));

        tlb_flush_page(cs, addr | (0xD << 28));

        tlb_flush_page(cs, addr | (0xE << 28));

        tlb_flush_page(cs, addr | (0xF << 28));




        break;

#if defined(TARGET_PPC64)

    case POWERPC_MMU_64B:

    case POWERPC_MMU_2_03:

    case POWERPC_MMU_2_06:

    case POWERPC_MMU_2_06a:

    case POWERPC_MMU_2_07:

    case POWERPC_MMU_2_07a:

        /* tlbie invalidate TLBs for all segments */

        /* XXX: given the fact that there are too many segments to invalidate,

         *      and we still don't have a tlb_flush_mask(env, n, mask) in QEMU,

         *      we just invalidate all TLBs

         */

        env->tlb_need_flush = 1;

        break;

#endif /* defined(TARGET_PPC64) */

    default:

        /* Should never reach here with other MMU models */

        assert(0);

    }


    ppc_tlb_invalidate_all(env);


}