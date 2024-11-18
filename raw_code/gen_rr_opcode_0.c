static TileExcp gen_rr_opcode(DisasContext *dc, unsigned opext,

                              unsigned dest, unsigned srca)

{

    TCGv tdest, tsrca;

    const char *mnemonic;

    TCGMemOp memop;

    TileExcp ret = TILEGX_EXCP_NONE;



    /* Eliminate instructions with no output before doing anything else.  */

    switch (opext) {

    case OE_RR_Y0(NOP):

    case OE_RR_Y1(NOP):

    case OE_RR_X0(NOP):

    case OE_RR_X1(NOP):

        mnemonic = "nop";

        goto done0;

    case OE_RR_Y0(FNOP):

    case OE_RR_Y1(FNOP):

    case OE_RR_X0(FNOP):

    case OE_RR_X1(FNOP):

        mnemonic = "fnop";

        goto done0;

    case OE_RR_X1(DRAIN):

        mnemonic = "drain";

        goto done0;

    case OE_RR_X1(FLUSHWB):

        mnemonic = "flushwb";

        goto done0;

    case OE_RR_X1(ILL):

        if (dest == 0x1c && srca == 0x25) {

            mnemonic = "bpt";

            goto done2;

        }

        /* Fall through */

    case OE_RR_Y1(ILL):

        mnemonic = "ill";

    done2:

        qemu_log_mask(CPU_LOG_TB_IN_ASM, "%s", mnemonic);

        return TILEGX_EXCP_OPCODE_UNKNOWN;

    case OE_RR_X1(MF):

        mnemonic = "mf";

        goto done0;

    case OE_RR_X1(NAP):

        /* ??? This should yield, especially in system mode.  */

        mnemonic = "nap";

        goto done0;

    case OE_RR_X1(SWINT0):

    case OE_RR_X1(SWINT2):

    case OE_RR_X1(SWINT3):

        return TILEGX_EXCP_OPCODE_UNIMPLEMENTED;

    case OE_RR_X1(SWINT1):

        ret = TILEGX_EXCP_SYSCALL;

        mnemonic = "swint1";

    done0:

        if (srca || dest) {

            return TILEGX_EXCP_OPCODE_UNIMPLEMENTED;

        }

        qemu_log_mask(CPU_LOG_TB_IN_ASM, "%s", mnemonic);

        return ret;



    case OE_RR_X1(DTLBPR):

        return TILEGX_EXCP_OPCODE_UNIMPLEMENTED;

    case OE_RR_X1(FINV):

        mnemonic = "finv";

        goto done1;

    case OE_RR_X1(FLUSH):

        mnemonic = "flush";

        goto done1;

    case OE_RR_X1(ICOH):

        mnemonic = "icoh";

        goto done1;

    case OE_RR_X1(INV):

        mnemonic = "inv";

        goto done1;

    case OE_RR_X1(WH64):

        mnemonic = "wh64";

        goto done1;

    case OE_RR_X1(JRP):

    case OE_RR_Y1(JRP):

        mnemonic = "jrp";

        goto do_jr;

    case OE_RR_X1(JR):

    case OE_RR_Y1(JR):

        mnemonic = "jr";

        goto do_jr;

    case OE_RR_X1(JALRP):

    case OE_RR_Y1(JALRP):

        mnemonic = "jalrp";

        goto do_jalr;

    case OE_RR_X1(JALR):

    case OE_RR_Y1(JALR):

        mnemonic = "jalr";

    do_jalr:

        tcg_gen_movi_tl(dest_gr(dc, TILEGX_R_LR),

                        dc->pc + TILEGX_BUNDLE_SIZE_IN_BYTES);

    do_jr:

        dc->jmp.cond = TCG_COND_ALWAYS;

        dc->jmp.dest = tcg_temp_new();

        tcg_gen_andi_tl(dc->jmp.dest, load_gr(dc, srca), ~7);

    done1:

        if (dest) {

            return TILEGX_EXCP_OPCODE_UNIMPLEMENTED;

        }

        qemu_log_mask(CPU_LOG_TB_IN_ASM, "%s %s", mnemonic, reg_names[srca]);

        return ret;

    }



    tdest = dest_gr(dc, dest);

    tsrca = load_gr(dc, srca);



    switch (opext) {

    case OE_RR_X0(CNTLZ):

    case OE_RR_Y0(CNTLZ):

        gen_helper_cntlz(tdest, tsrca);

        mnemonic = "cntlz";

        break;

    case OE_RR_X0(CNTTZ):

    case OE_RR_Y0(CNTTZ):

        gen_helper_cnttz(tdest, tsrca);

        mnemonic = "cnttz";

        break;

    case OE_RR_X0(FSINGLE_PACK1):

    case OE_RR_Y0(FSINGLE_PACK1):

    case OE_RR_X1(IRET):

        return TILEGX_EXCP_OPCODE_UNIMPLEMENTED;

    case OE_RR_X1(LD1S):

        memop = MO_SB;

        mnemonic = "ld1s";

        goto do_load;

    case OE_RR_X1(LD1U):

        memop = MO_UB;

        mnemonic = "ld1u";

        goto do_load;

    case OE_RR_X1(LD2S):

        memop = MO_TESW;

        mnemonic = "ld2s";

        goto do_load;

    case OE_RR_X1(LD2U):

        memop = MO_TEUW;

        mnemonic = "ld2u";

        goto do_load;

    case OE_RR_X1(LD4S):

        memop = MO_TESL;

        mnemonic = "ld4s";

        goto do_load;

    case OE_RR_X1(LD4U):

        memop = MO_TEUL;

        mnemonic = "ld4u";

        goto do_load;

    case OE_RR_X1(LDNT1S):

        memop = MO_SB;

        mnemonic = "ldnt1s";

        goto do_load;

    case OE_RR_X1(LDNT1U):

        memop = MO_UB;

        mnemonic = "ldnt1u";

        goto do_load;

    case OE_RR_X1(LDNT2S):

        memop = MO_TESW;

        mnemonic = "ldnt2s";

        goto do_load;

    case OE_RR_X1(LDNT2U):

        memop = MO_TEUW;

        mnemonic = "ldnt2u";

        goto do_load;

    case OE_RR_X1(LDNT4S):

        memop = MO_TESL;

        mnemonic = "ldnt4s";

        goto do_load;

    case OE_RR_X1(LDNT4U):

        memop = MO_TEUL;

        mnemonic = "ldnt4u";

        goto do_load;

    case OE_RR_X1(LDNT):

        memop = MO_TEQ;

        mnemonic = "ldnt";

        goto do_load;

    case OE_RR_X1(LD):

        memop = MO_TEQ;

        mnemonic = "ld";

    do_load:

        tcg_gen_qemu_ld_tl(tdest, tsrca, dc->mmuidx, memop);

        break;

    case OE_RR_X1(LDNA):

        tcg_gen_andi_tl(tdest, tsrca, ~7);

        tcg_gen_qemu_ld_tl(tdest, tdest, dc->mmuidx, MO_TEQ);

        mnemonic = "ldna";

        break;

    case OE_RR_X1(LNK):

    case OE_RR_Y1(LNK):

        if (srca) {

            return TILEGX_EXCP_OPCODE_UNIMPLEMENTED;

        }

        tcg_gen_movi_tl(tdest, dc->pc + TILEGX_BUNDLE_SIZE_IN_BYTES);

        mnemonic = "lnk";

        break;

    case OE_RR_X0(PCNT):

    case OE_RR_Y0(PCNT):

        gen_helper_pcnt(tdest, tsrca);

        mnemonic = "pcnt";

        break;

    case OE_RR_X0(REVBITS):

    case OE_RR_Y0(REVBITS):

        gen_helper_revbits(tdest, tsrca);

        mnemonic = "revbits";

        break;

    case OE_RR_X0(REVBYTES):

    case OE_RR_Y0(REVBYTES):

        tcg_gen_bswap64_tl(tdest, tsrca);

        mnemonic = "revbytes";

        break;

    case OE_RR_X0(TBLIDXB0):

    case OE_RR_Y0(TBLIDXB0):

        tcg_gen_deposit_tl(tdest, load_gr(dc, dest), tsrca, 2, 8);

        mnemonic = "tblidxb0";

        break;

    case OE_RR_X0(TBLIDXB1):

    case OE_RR_Y0(TBLIDXB1):

        tcg_gen_shri_tl(tdest, tsrca, 8);

        tcg_gen_deposit_tl(tdest, load_gr(dc, dest), tdest, 2, 8);

        mnemonic = "tblidxb1";

        break;

    case OE_RR_X0(TBLIDXB2):

    case OE_RR_Y0(TBLIDXB2):

        tcg_gen_shri_tl(tdest, tsrca, 16);

        tcg_gen_deposit_tl(tdest, load_gr(dc, dest), tdest, 2, 8);

        mnemonic = "tblidxb2";

        break;

    case OE_RR_X0(TBLIDXB3):

    case OE_RR_Y0(TBLIDXB3):

        tcg_gen_shri_tl(tdest, tsrca, 24);

        tcg_gen_deposit_tl(tdest, load_gr(dc, dest), tdest, 2, 8);

        mnemonic = "tblidxb3";

        break;

    default:

        return TILEGX_EXCP_OPCODE_UNIMPLEMENTED;

    }



    qemu_log_mask(CPU_LOG_TB_IN_ASM, "%s %s, %s", mnemonic,

                  reg_names[dest], reg_names[srca]);

    return ret;

}
