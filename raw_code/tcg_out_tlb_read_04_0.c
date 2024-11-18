static void tcg_out_tlb_read (TCGContext *s, int r0, int r1, int r2,

                              int addr_reg, int s_bits, int offset)

{

#ifdef TARGET_LONG_BITS

    tcg_out_rld (s, RLDICL, addr_reg, addr_reg, 0, 32);



    tcg_out32 (s, (RLWINM

                   | RA (r0)

                   | RS (addr_reg)

                   | SH (32 - (TARGET_PAGE_BITS - CPU_TLB_ENTRY_BITS))

                   | MB (32 - (CPU_TLB_BITS + CPU_TLB_ENTRY_BITS))

                   | ME (31 - CPU_TLB_ENTRY_BITS)

                   )

        );

    tcg_out32 (s, ADD | RT (r0) | RA (r0) | RB (TCG_AREG0));

    tcg_out32 (s, (LWZU | RT (r1) | RA (r0) | offset));

    tcg_out32 (s, (RLWINM

                   | RA (r2)

                   | RS (addr_reg)

                   | SH (0)

                   | MB ((32 - s_bits) & 31)

                   | ME (31 - TARGET_PAGE_BITS)

                   )

        );

#else

    tcg_out_rld (s, RLDICL, r0, addr_reg,

                 64 - TARGET_PAGE_BITS,

                 64 - CPU_TLB_BITS);

    tcg_out_rld (s, RLDICR, r0, r0,

                 CPU_TLB_ENTRY_BITS,

                 63 - CPU_TLB_ENTRY_BITS);



    tcg_out32 (s, ADD | TAB (r0, r0, TCG_AREG0));

    tcg_out32 (s, LD_ADDR | RT (r1) | RA (r0) | offset);



    if (!s_bits) {

        tcg_out_rld (s, RLDICR, r2, addr_reg, 0, 63 - TARGET_PAGE_BITS);

    }

    else {

        tcg_out_rld (s, RLDICL, r2, addr_reg,

                     64 - TARGET_PAGE_BITS,

                     TARGET_PAGE_BITS - s_bits);

        tcg_out_rld (s, RLDICL, r2, r2, TARGET_PAGE_BITS, 0);

    }

#endif

}
