static int mmu_translate_asc(CPUS390XState *env, target_ulong vaddr,

                             uint64_t asc, target_ulong *raddr, int *flags,

                             int rw)

{

    uint64_t asce = 0;

    int level, new_level;

    int r;



    switch (asc) {

    case PSW_ASC_PRIMARY:

        PTE_DPRINTF("%s: asc=primary\n", __func__);

        asce = env->cregs[1];

        break;

    case PSW_ASC_SECONDARY:

        PTE_DPRINTF("%s: asc=secondary\n", __func__);

        asce = env->cregs[7];

        break;

    case PSW_ASC_HOME:

        PTE_DPRINTF("%s: asc=home\n", __func__);

        asce = env->cregs[13];

        break;

    }



    if (asce & _ASCE_REAL_SPACE) {

        /* direct mapping */

        *raddr = vaddr;

        return 0;

    }



    switch (asce & _ASCE_TYPE_MASK) {

    case _ASCE_TYPE_REGION1:

        break;

    case _ASCE_TYPE_REGION2:

        if (vaddr & 0xffe0000000000000ULL) {

            DPRINTF("%s: vaddr doesn't fit 0x%16" PRIx64

                    " 0xffe0000000000000ULL\n", __func__, vaddr);

            trigger_page_fault(env, vaddr, PGM_TRANS_SPEC, asc, rw);

            return -1;

        }

        break;

    case _ASCE_TYPE_REGION3:

        if (vaddr & 0xfffffc0000000000ULL) {

            DPRINTF("%s: vaddr doesn't fit 0x%16" PRIx64

                    " 0xfffffc0000000000ULL\n", __func__, vaddr);

            trigger_page_fault(env, vaddr, PGM_TRANS_SPEC, asc, rw);

            return -1;

        }

        break;

    case _ASCE_TYPE_SEGMENT:

        if (vaddr & 0xffffffff80000000ULL) {

            DPRINTF("%s: vaddr doesn't fit 0x%16" PRIx64

                    " 0xffffffff80000000ULL\n", __func__, vaddr);

            trigger_page_fault(env, vaddr, PGM_TRANS_SPEC, asc, rw);

            return -1;

        }

        break;

    }



    /* fake level above current */

    level = asce & _ASCE_TYPE_MASK;

    new_level = level + 4;

    asce = (asce & ~_ASCE_TYPE_MASK) | (new_level & _ASCE_TYPE_MASK);



    r = mmu_translate_asce(env, vaddr, asc, asce, new_level, raddr, flags, rw);



    if ((rw == 1) && !(*flags & PAGE_WRITE)) {

        trigger_prot_fault(env, vaddr, asc);

        return -1;

    }



    return r;

}
