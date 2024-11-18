static void mips_tlb_flush_extra (CPUState *env, int first)

{

    /* Discard entries from env->tlb[first] onwards.  */

    while (env->tlb_in_use > first) {

        invalidate_tlb(--env->tlb_in_use, 0);

    }

}
