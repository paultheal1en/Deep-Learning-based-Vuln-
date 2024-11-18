static void tcg_out_qemu_ld (TCGContext *s, const TCGArg *args, int opc)

{

    int addr_reg, data_reg, data_reg2, r0, r1, rbase, mem_index, s_bits, bswap;

#ifdef CONFIG_SOFTMMU

    int r2;

    void *label1_ptr, *label2_ptr;

#endif

#if TARGET_LONG_BITS == 64

    int addr_reg2;

#endif



    data_reg = *args++;

    if (opc == 3)

        data_reg2 = *args++;

    else

        data_reg2 = 0;

    addr_reg = *args++;

#if TARGET_LONG_BITS == 64

    addr_reg2 = *args++;

#endif

    mem_index = *args;

    s_bits = opc & 3;



#ifdef CONFIG_SOFTMMU

    r0 = 3;

    r1 = 4;

    r2 = 0;

    rbase = 0;



    tcg_out32 (s, (RLWINM

                   | RA (r0)

                   | RS (addr_reg)

                   | SH (32 - (TARGET_PAGE_BITS - CPU_TLB_ENTRY_BITS))

                   | MB (32 - (CPU_TLB_BITS + CPU_TLB_ENTRY_BITS))

                   | ME (31 - CPU_TLB_ENTRY_BITS)

                   )

        );

    tcg_out32 (s, ADD | RT (r0) | RA (r0) | RB (TCG_AREG0));

    tcg_out32 (s, (LWZU

                   | RT (r1)

                   | RA (r0)

                   | offsetof (CPUState, tlb_table[mem_index][0].addr_read)

                   )

        );

    tcg_out32 (s, (RLWINM

                   | RA (r2)

                   | RS (addr_reg)

                   | SH (0)

                   | MB ((32 - s_bits) & 31)

                   | ME (31 - TARGET_PAGE_BITS)

                   )

        );



    tcg_out32 (s, CMP | BF (7) | RA (r2) | RB (r1));

#if TARGET_LONG_BITS == 64

    tcg_out32 (s, LWZ | RT (r1) | RA (r0) | 4);

    tcg_out32 (s, CMP | BF (6) | RA (addr_reg2) | RB (r1));

    tcg_out32 (s, CRAND | BT (7, CR_EQ) | BA (6, CR_EQ) | BB (7, CR_EQ));

#endif



    label1_ptr = s->code_ptr;

#ifdef FAST_PATH

    tcg_out32 (s, BC | BI (7, CR_EQ) | BO_COND_TRUE);

#endif



    /* slow path */

#if TARGET_LONG_BITS == 32

    tcg_out_mov (s, 3, addr_reg);

    tcg_out_movi (s, TCG_TYPE_I32, 4, mem_index);

#else

    tcg_out_mov (s, 3, addr_reg2);

    tcg_out_mov (s, 4, addr_reg);

    tcg_out_movi (s, TCG_TYPE_I32, 5, mem_index);

#endif



    tcg_out_call (s, (tcg_target_long) qemu_ld_helpers[s_bits], 1);

    switch (opc) {

    case 0|4:

        tcg_out32 (s, EXTSB | RA (data_reg) | RS (3));

        break;

    case 1|4:

        tcg_out32 (s, EXTSH | RA (data_reg) | RS (3));

        break;

    case 0:

    case 1:

    case 2:

        if (data_reg != 3)

            tcg_out_mov (s, data_reg, 3);

        break;

    case 3:

        if (data_reg == 3) {

            if (data_reg2 == 4) {

                tcg_out_mov (s, 0, 4);

                tcg_out_mov (s, 4, 3);

                tcg_out_mov (s, 3, 0);

            }

            else {

                tcg_out_mov (s, data_reg2, 3);

                tcg_out_mov (s, 3, 4);

            }

        }

        else {

            if (data_reg != 4) tcg_out_mov (s, data_reg, 4);

            if (data_reg2 != 3) tcg_out_mov (s, data_reg2, 3);

        }

        break;

    }

    label2_ptr = s->code_ptr;

    tcg_out32 (s, B);



    /* label1: fast path */

#ifdef FAST_PATH

    reloc_pc14 (label1_ptr, (tcg_target_long) s->code_ptr);

#endif



    /* r0 now contains &env->tlb_table[mem_index][index].addr_read */

    tcg_out32 (s, (LWZ

                   | RT (r0)

                   | RA (r0)

                   | (ADDEND_OFFSET + offsetof (CPUTLBEntry, addend)

                      - offsetof (CPUTLBEntry, addr_read))

                   ));

    /* r0 = env->tlb_table[mem_index][index].addend */

    tcg_out32 (s, ADD | RT (r0) | RA (r0) | RB (addr_reg));

    /* r0 = env->tlb_table[mem_index][index].addend + addr */



#else  /* !CONFIG_SOFTMMU */

    r0 = addr_reg;

    r1 = 3;

    rbase = GUEST_BASE ? TCG_GUEST_BASE_REG : 0;

#endif



#ifdef TARGET_WORDS_BIGENDIAN

    bswap = 0;

#else

    bswap = 1;

#endif



    switch (opc) {

    default:

    case 0:

        tcg_out32 (s, LBZX | TAB (data_reg, rbase, r0));

        break;

    case 0|4:

        tcg_out32 (s, LBZX | TAB (data_reg, rbase, r0));

        tcg_out32 (s, EXTSB | RA (data_reg) | RS (data_reg));

        break;

    case 1:

        if (bswap)

            tcg_out32 (s, LHBRX | TAB (data_reg, rbase, r0));

        else

            tcg_out32 (s, LHZX | TAB (data_reg, rbase, r0));

        break;

    case 1|4:

        if (bswap) {

            tcg_out32 (s, LHBRX | TAB (data_reg, rbase, r0));

            tcg_out32 (s, EXTSH | RA (data_reg) | RS (data_reg));

        }

        else tcg_out32 (s, LHAX | TAB (data_reg, rbase, r0));

        break;

    case 2:

        if (bswap)

            tcg_out32 (s, LWBRX | TAB (data_reg, rbase, r0));

        else

            tcg_out32 (s, LWZX | TAB (data_reg, rbase, r0));

        break;

    case 3:

        if (bswap) {

            tcg_out32 (s, ADDI | RT (r1) | RA (r0) | 4);

            tcg_out32 (s, LWBRX | TAB (data_reg, rbase, r0));

            tcg_out32 (s, LWBRX | TAB (data_reg2, rbase, r1));

        }

        else {

#ifdef CONFIG_USE_GUEST_BASE

            tcg_out32 (s, ADDI | RT (r1) | RA (r0) | 4);

            tcg_out32 (s, LWZX | TAB (data_reg2, rbase, r0));

            tcg_out32 (s, LWZX | TAB (data_reg, rbase, r1));

#else

            if (r0 == data_reg2) {

                tcg_out32 (s, LWZ | RT (0) | RA (r0));

                tcg_out32 (s, LWZ | RT (data_reg) | RA (r0) | 4);

                tcg_out_mov (s, data_reg2, 0);

            }

            else {

                tcg_out32 (s, LWZ | RT (data_reg2) | RA (r0));

                tcg_out32 (s, LWZ | RT (data_reg) | RA (r0) | 4);

            }

#endif

        }

        break;

    }



#ifdef CONFIG_SOFTMMU

    reloc_pc24 (label2_ptr, (tcg_target_long) s->code_ptr);

#endif

}
