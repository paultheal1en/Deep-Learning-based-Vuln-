void helper_4xx_tlbwe_hi (target_ulong entry, target_ulong val)

{

    ppcemb_tlb_t *tlb;

    target_ulong page, end;



    LOG_SWTLB("%s entry %d val " TARGET_FMT_lx "\n", __func__, (int)entry,

              val);

    entry &= 0x3F;

    tlb = &env->tlb[entry].tlbe;

    /* Invalidate previous TLB (if it's valid) */

    if (tlb->prot & PAGE_VALID) {

        end = tlb->EPN + tlb->size;

        LOG_SWTLB("%s: invalidate old TLB %d start " TARGET_FMT_lx " end "

                  TARGET_FMT_lx "\n", __func__, (int)entry, tlb->EPN, end);

        for (page = tlb->EPN; page < end; page += TARGET_PAGE_SIZE)

            tlb_flush_page(env, page);

    }

    tlb->size = booke_tlb_to_page_size((val >> 7) & 0x7);

    /* We cannot handle TLB size < TARGET_PAGE_SIZE.

     * If this ever occurs, one should use the ppcemb target instead

     * of the ppc or ppc64 one

     */

    if ((val & 0x40) && tlb->size < TARGET_PAGE_SIZE) {

        cpu_abort(env, "TLB size " TARGET_FMT_lu " < %u "

                  "are not supported (%d)\n",

                  tlb->size, TARGET_PAGE_SIZE, (int)((val >> 7) & 0x7));

    }

    tlb->EPN = val & ~(tlb->size - 1);

    if (val & 0x40)

        tlb->prot |= PAGE_VALID;

    else

        tlb->prot &= ~PAGE_VALID;

    if (val & 0x20) {

        /* XXX: TO BE FIXED */

        cpu_abort(env, "Little-endian TLB entries are not supported by now\n");

    }

    tlb->PID = env->spr[SPR_40x_PID]; /* PID */

    LOG_SWTLB("%s: set up TLB %d RPN " TARGET_FMT_plx " EPN " TARGET_FMT_lx

              " size " TARGET_FMT_lx " prot %c%c%c%c PID %d\n", __func__,

              (int)entry, tlb->RPN, tlb->EPN, tlb->size,

              tlb->prot & PAGE_READ ? 'r' : '-',

              tlb->prot & PAGE_WRITE ? 'w' : '-',

              tlb->prot & PAGE_EXEC ? 'x' : '-',

              tlb->prot & PAGE_VALID ? 'v' : '-', (int)tlb->PID);

    /* Invalidate new TLB (if valid) */

    if (tlb->prot & PAGE_VALID) {

        end = tlb->EPN + tlb->size;

        LOG_SWTLB("%s: invalidate TLB %d start " TARGET_FMT_lx " end "

                  TARGET_FMT_lx "\n", __func__, (int)entry, tlb->EPN, end);

        for (page = tlb->EPN; page < end; page += TARGET_PAGE_SIZE)

            tlb_flush_page(env, page);

    }

}
