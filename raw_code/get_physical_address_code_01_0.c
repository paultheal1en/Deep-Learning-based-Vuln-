static int get_physical_address_code(CPUState *env,

                                     target_phys_addr_t *physical, int *prot,

                                     target_ulong address, int is_user)

{

    target_ulong mask;

    unsigned int i;



    if ((env->lsu & IMMU_E) == 0) { /* IMMU disabled */

        *physical = address;

        *prot = PAGE_EXEC;

        return 0;

    }



    for (i = 0; i < 64; i++) {

        switch ((env->itlb_tte[i] >> 61) & 3) {

        default:

        case 0x0: // 8k

            mask = 0xffffffffffffe000ULL;

            break;

        case 0x1: // 64k

            mask = 0xffffffffffff0000ULL;

            break;

        case 0x2: // 512k

            mask = 0xfffffffffff80000ULL;

            break;

        case 0x3: // 4M

            mask = 0xffffffffffc00000ULL;

                break;

        }

        // ctx match, vaddr match, valid?

        if (env->dmmuregs[1] == (env->itlb_tag[i] & 0x1fff) &&

            (address & mask) == (env->itlb_tag[i] & mask) &&

            (env->itlb_tte[i] & 0x8000000000000000ULL)) {

            // access ok?

            if ((env->itlb_tte[i] & 0x4) && is_user) {

                if (env->immuregs[3]) /* Fault status register */

                    env->immuregs[3] = 2; /* overflow (not read before

                                             another fault) */

                env->immuregs[3] |= (is_user << 3) | 1;

                env->exception_index = TT_TFAULT;

#ifdef DEBUG_MMU

                printf("TFAULT at 0x%" PRIx64 "\n", address);

#endif

                return 1;

            }

            *physical = ((env->itlb_tte[i] & mask) | (address & ~mask)) &

                        0x1ffffffe000ULL;

            *prot = PAGE_EXEC;

            return 0;

        }

    }

#ifdef DEBUG_MMU

    printf("TMISS at 0x%" PRIx64 "\n", address);

#endif

    /* Context is stored in DMMU (dmmuregs[1]) also for IMMU */

    env->immuregs[6] = (address & ~0x1fffULL) | (env->dmmuregs[1] & 0x1fff);

    env->exception_index = TT_TMISS;

    return 1;

}
