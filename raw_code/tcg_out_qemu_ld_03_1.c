static void tcg_out_qemu_ld(TCGContext *s, const TCGArg *args,

                            int opc)

{

    int addr_regl, addr_reg1, addr_meml;

    int data_regl, data_regh, data_reg1, data_reg2;

    int mem_index, s_bits;

#if defined(CONFIG_SOFTMMU)

    void *label1_ptr, *label2_ptr;

    int sp_args;

#endif

#if TARGET_LONG_BITS == 64

# if defined(CONFIG_SOFTMMU)

    uint8_t *label3_ptr;

# endif

    int addr_regh, addr_reg2, addr_memh;

#endif

    data_regl = *args++;

    if (opc == 3)

        data_regh = *args++;

    else

        data_regh = 0;

    addr_regl = *args++;

#if TARGET_LONG_BITS == 64

    addr_regh = *args++;

#endif

    mem_index = *args;

    s_bits = opc & 3;



    if (opc == 3) {

#if defined(TCG_TARGET_WORDS_BIGENDIAN)

        data_reg1 = data_regh;

        data_reg2 = data_regl;

#else

        data_reg1 = data_regl;

        data_reg2 = data_regh;

#endif

    } else {

        data_reg1 = data_regl;

        data_reg2 = 0;

    }

#if TARGET_LONG_BITS == 64

# if defined(TCG_TARGET_WORDS_BIGENDIAN)

    addr_reg1 = addr_regh;

    addr_reg2 = addr_regl;

    addr_memh = 0;

    addr_meml = 4;

# else

    addr_reg1 = addr_regl;

    addr_reg2 = addr_regh;

    addr_memh = 4;

    addr_meml = 0;

# endif

#else

    addr_reg1 = addr_regl;

    addr_meml = 0;

#endif



#if defined(CONFIG_SOFTMMU)

    tcg_out_opc_sa(s, OPC_SRL, TCG_REG_A0, addr_regl, TARGET_PAGE_BITS - CPU_TLB_ENTRY_BITS);

    tcg_out_opc_imm(s, OPC_ANDI, TCG_REG_A0, TCG_REG_A0, (CPU_TLB_SIZE - 1) << CPU_TLB_ENTRY_BITS);

    tcg_out_opc_reg(s, OPC_ADDU, TCG_REG_A0, TCG_REG_A0, TCG_AREG0);

    tcg_out_opc_imm(s, OPC_LW, TCG_REG_AT, TCG_REG_A0,

                    offsetof(CPUState, tlb_table[mem_index][0].addr_read) + addr_meml);

    tcg_out_movi(s, TCG_TYPE_I32, TCG_REG_T0, TARGET_PAGE_MASK | ((1 << s_bits) - 1));

    tcg_out_opc_reg(s, OPC_AND, TCG_REG_T0, TCG_REG_T0, addr_regl);



# if TARGET_LONG_BITS == 64

    label3_ptr = s->code_ptr;

    tcg_out_opc_imm(s, OPC_BNE, TCG_REG_T0, TCG_REG_AT, 0);

    tcg_out_nop(s);



    tcg_out_opc_imm(s, OPC_LW, TCG_REG_AT, TCG_REG_A0,

                    offsetof(CPUState, tlb_table[mem_index][0].addr_read) + addr_memh);



    label1_ptr = s->code_ptr;

    tcg_out_opc_imm(s, OPC_BEQ, addr_regh, TCG_REG_AT, 0);

    tcg_out_nop(s);



    reloc_pc16(label3_ptr, (tcg_target_long) s->code_ptr);

# else

    label1_ptr = s->code_ptr;

    tcg_out_opc_imm(s, OPC_BEQ, TCG_REG_T0, TCG_REG_AT, 0);

    tcg_out_nop(s);

# endif



    /* slow path */

    sp_args = TCG_REG_A0;

    tcg_out_mov(s, sp_args++, addr_reg1);

# if TARGET_LONG_BITS == 64

    tcg_out_mov(s, sp_args++, addr_reg2);

# endif

    tcg_out_movi(s, TCG_TYPE_I32, sp_args++, mem_index);

    tcg_out_movi(s, TCG_TYPE_I32, TCG_REG_T9, (tcg_target_long)qemu_ld_helpers[s_bits]);

    tcg_out_opc_reg(s, OPC_JALR, TCG_REG_RA, TCG_REG_T9, 0);

    tcg_out_nop(s);



    switch(opc) {

    case 0:

        tcg_out_opc_imm(s, OPC_ANDI, data_reg1, TCG_REG_V0, 0xff);

        break;

    case 0 | 4:

        tcg_out_opc_sa(s, OPC_SLL, TCG_REG_V0, TCG_REG_V0, 24);

        tcg_out_opc_sa(s, OPC_SRA, data_reg1, TCG_REG_V0, 24);

        break;

    case 1:

        tcg_out_opc_imm(s, OPC_ANDI, data_reg1, TCG_REG_V0, 0xffff);

        break;

    case 1 | 4:

        tcg_out_opc_sa(s, OPC_SLL, TCG_REG_V0, TCG_REG_V0, 16);

        tcg_out_opc_sa(s, OPC_SRA, data_reg1, TCG_REG_V0, 16);

        break;

    case 2:

        tcg_out_mov(s, data_reg1, TCG_REG_V0);

        break;

    case 3:

        tcg_out_mov(s, data_reg2, TCG_REG_V1);

        tcg_out_mov(s, data_reg1, TCG_REG_V0);

        break;

    default:

        tcg_abort();

    }



    label2_ptr = s->code_ptr;

    tcg_out_opc_imm(s, OPC_BEQ, TCG_REG_ZERO, TCG_REG_ZERO, 0);

    tcg_out_nop(s);



    /* label1: fast path */

    reloc_pc16(label1_ptr, (tcg_target_long) s->code_ptr);



    tcg_out_opc_imm(s, OPC_LW, TCG_REG_V0, TCG_REG_A0,

                    offsetof(CPUState, tlb_table[mem_index][0].addend) + addr_meml);

    tcg_out_opc_reg(s, OPC_ADDU, TCG_REG_V0, TCG_REG_V0, addr_regl);



    addr_reg1 = TCG_REG_V0;

#endif



    switch(opc) {

    case 0:

        tcg_out_opc_imm(s, OPC_LBU, data_reg1, addr_reg1, 0);

        break;

    case 0 | 4:

        tcg_out_opc_imm(s, OPC_LB, data_reg1, addr_reg1, 0);

        break;

    case 1:

        if (TCG_NEED_BSWAP) {

            tcg_out_opc_imm(s, OPC_LHU, TCG_REG_T0, addr_reg1, 0);

            tcg_out_bswap16(s, data_reg1, TCG_REG_T0);

        } else {

            tcg_out_opc_imm(s, OPC_LHU, data_reg1, addr_reg1, 0);

        }

        break;

    case 1 | 4:

        if (TCG_NEED_BSWAP) {

            tcg_out_opc_imm(s, OPC_LHU, TCG_REG_T0, addr_reg1, 0);

            tcg_out_bswap16s(s, data_reg1, TCG_REG_T0);

        } else {

            tcg_out_opc_imm(s, OPC_LH, data_reg1, addr_reg1, 0);

        }

        break;

    case 2:

        if (TCG_NEED_BSWAP) {

            tcg_out_opc_imm(s, OPC_LW, TCG_REG_T0, addr_reg1, 0);

            tcg_out_bswap32(s, data_reg1, TCG_REG_T0);

        } else {

            tcg_out_opc_imm(s, OPC_LW, data_reg1, addr_reg1, 0);

        }

        break;

    case 3:

#if !defined(CONFIG_SOFTMMU)

        tcg_out_mov(s, TCG_REG_V0, addr_reg1);

        addr_reg1 = TCG_REG_V0;

#endif

        if (TCG_NEED_BSWAP) {

            tcg_out_opc_imm(s, OPC_LW, TCG_REG_T0, addr_reg1, 4);

            tcg_out_bswap32(s, data_reg1, TCG_REG_T0);

            tcg_out_opc_imm(s, OPC_LW, TCG_REG_T0, addr_reg1, 0);

            tcg_out_bswap32(s, data_reg2, TCG_REG_T0);

        } else {

            tcg_out_opc_imm(s, OPC_LW, data_reg1, addr_reg1, 0);

            tcg_out_opc_imm(s, OPC_LW, data_reg2, addr_reg1, 4);

        }

        break;

    default:

        tcg_abort();

    }



#if defined(CONFIG_SOFTMMU)

    reloc_pc16(label2_ptr, (tcg_target_long) s->code_ptr);

#endif

}
