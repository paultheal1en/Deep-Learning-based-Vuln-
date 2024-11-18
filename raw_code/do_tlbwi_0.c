void do_tlbwi (void)

{

    /* Discard cached TLB entries.  We could avoid doing this if the

       tlbwi is just upgrading access permissions on the current entry;

       that might be a further win.  */

    mips_tlb_flush_extra (env, MIPS_TLB_NB);



    /* Wildly undefined effects for CP0_index containing a too high value and

       MIPS_TLB_NB not being a power of two.  But so does real silicon.  */

    invalidate_tlb(env->CP0_index & (MIPS_TLB_NB - 1), 0);

    fill_tlb(env->CP0_index & (MIPS_TLB_NB - 1));

}
