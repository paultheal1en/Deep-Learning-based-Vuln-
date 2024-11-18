static int get_physical_address_data(CPUState *env,

                                     target_phys_addr_t *physical, int *prot,

                                     target_ulong address, int rw, int is_user)

{

    unsigned int i;

    uint64_t context;

    int is_nucleus;



    if ((env->lsu & DMMU_E) == 0) { /* DMMU disabled */

        *physical = ultrasparc_truncate_physical(address);

        *prot = PAGE_READ | PAGE_WRITE;

        return 0;

    }



    context = env->dmmu.mmu_primary_context & 0x1fff;

    is_nucleus = env->tl > 0;



    for (i = 0; i < 64; i++) {

        // ctx match, vaddr match, valid?

        if (ultrasparc_tag_match(&env->dtlb[i],

                                 address, context, physical,

                                 is_nucleus)) {

            // access ok?

            if (((env->dtlb[i].tte & 0x4) && is_user) ||

                (!(env->dtlb[i].tte & 0x2) && (rw == 1))) {

                uint8_t fault_type = 0;



                if ((env->dtlb[i].tte & 0x4) && is_user) {

                    fault_type |= 1; /* privilege violation */

                }



                if (env->dmmu.sfsr & 1) /* Fault status register */

                    env->dmmu.sfsr = 2; /* overflow (not read before

                                             another fault) */



                env->dmmu.sfsr |= (is_user << 3) | ((rw == 1) << 2) | 1;



                env->dmmu.sfsr |= (fault_type << 7);



                env->dmmu.sfar = address; /* Fault address register */

                env->exception_index = TT_DFAULT;

#ifdef DEBUG_MMU

                printf("DFAULT at 0x%" PRIx64 "\n", address);

#endif

                return 1;

            }

            *prot = PAGE_READ;

            if (env->dtlb[i].tte & 0x2)

                *prot |= PAGE_WRITE;

            TTE_SET_USED(env->dtlb[i].tte);

            return 0;

        }

    }

#ifdef DEBUG_MMU

    printf("DMISS at 0x%" PRIx64 "\n", address);

#endif

    env->dmmu.tag_access = (address & ~0x1fffULL) | context;

    env->exception_index = TT_DMISS;

    return 1;

}
