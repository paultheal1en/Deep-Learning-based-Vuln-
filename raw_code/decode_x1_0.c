static TileExcp decode_x1(DisasContext *dc, tilegx_bundle_bits bundle)

{

    unsigned opc = get_Opcode_X1(bundle);

    unsigned dest = get_Dest_X1(bundle);

    unsigned srca = get_SrcA_X1(bundle);

    unsigned ext, srcb;

    int imm;



    switch (opc) {

    case RRR_0_OPCODE_X1:

        ext = get_RRROpcodeExtension_X1(bundle);

        srcb = get_SrcB_X1(bundle);

        switch (ext) {

        case UNARY_RRR_0_OPCODE_X1:

            ext = get_UnaryOpcodeExtension_X1(bundle);

            return gen_rr_opcode(dc, OE(opc, ext, X1), dest, srca);

        case ST1_RRR_0_OPCODE_X1:

            return gen_st_opcode(dc, dest, srca, srcb, MO_UB, "st1");

        case ST2_RRR_0_OPCODE_X1:

            return gen_st_opcode(dc, dest, srca, srcb, MO_TEUW, "st2");

        case ST4_RRR_0_OPCODE_X1:

            return gen_st_opcode(dc, dest, srca, srcb, MO_TEUL, "st4");

        case STNT1_RRR_0_OPCODE_X1:

            return gen_st_opcode(dc, dest, srca, srcb, MO_UB, "stnt1");

        case STNT2_RRR_0_OPCODE_X1:

            return gen_st_opcode(dc, dest, srca, srcb, MO_TEUW, "stnt2");

        case STNT4_RRR_0_OPCODE_X1:

            return gen_st_opcode(dc, dest, srca, srcb, MO_TEUL, "stnt4");

        case STNT_RRR_0_OPCODE_X1:

            return gen_st_opcode(dc, dest, srca, srcb, MO_TEQ, "stnt");

        case ST_RRR_0_OPCODE_X1:

            return gen_st_opcode(dc, dest, srca, srcb, MO_TEQ, "st");

        }

        return gen_rrr_opcode(dc, OE(opc, ext, X1), dest, srca, srcb);



    case SHIFT_OPCODE_X1:

        ext = get_ShiftOpcodeExtension_X1(bundle);

        imm = get_ShAmt_X1(bundle);

        return gen_rri_opcode(dc, OE(opc, ext, X1), dest, srca, imm);



    case IMM8_OPCODE_X1:

        ext = get_Imm8OpcodeExtension_X1(bundle);

        imm = (int8_t)get_Dest_Imm8_X1(bundle);

        srcb = get_SrcB_X1(bundle);

        switch (ext) {

        case ST1_ADD_IMM8_OPCODE_X1:

            return gen_st_add_opcode(dc, srca, srcb, imm, MO_UB, "st1_add");

        case ST2_ADD_IMM8_OPCODE_X1:

            return gen_st_add_opcode(dc, srca, srcb, imm, MO_TEUW, "st2_add");

        case ST4_ADD_IMM8_OPCODE_X1:

            return gen_st_add_opcode(dc, srca, srcb, imm, MO_TEUL, "st4_add");

        case STNT1_ADD_IMM8_OPCODE_X1:

            return gen_st_add_opcode(dc, srca, srcb, imm, MO_UB, "stnt1_add");

        case STNT2_ADD_IMM8_OPCODE_X1:

            return gen_st_add_opcode(dc, srca, srcb, imm, MO_TEUW, "stnt2_add");

        case STNT4_ADD_IMM8_OPCODE_X1:

            return gen_st_add_opcode(dc, srca, srcb, imm, MO_TEUL, "stnt4_add");

        case STNT_ADD_IMM8_OPCODE_X1:

            return gen_st_add_opcode(dc, srca, srcb, imm, MO_TEQ, "stnt_add");

        case ST_ADD_IMM8_OPCODE_X1:

            return gen_st_add_opcode(dc, srca, srcb, imm, MO_TEQ, "st_add");

        case MFSPR_IMM8_OPCODE_X1:

            return gen_mfspr_x1(dc, dest, get_MF_Imm14_X1(bundle));

        case MTSPR_IMM8_OPCODE_X1:

            return gen_mtspr_x1(dc, get_MT_Imm14_X1(bundle), srca);

        }

        imm = (int8_t)get_Imm8_X1(bundle);

        return gen_rri_opcode(dc, OE(opc, ext, X1), dest, srca, imm);



    case BRANCH_OPCODE_X1:

        ext = get_BrType_X1(bundle);

        imm = sextract32(get_BrOff_X1(bundle), 0, 17);

        return gen_branch_opcode_x1(dc, ext, srca, imm);



    case JUMP_OPCODE_X1:

        ext = get_JumpOpcodeExtension_X1(bundle);

        imm = sextract32(get_JumpOff_X1(bundle), 0, 27);

        return gen_jump_opcode_x1(dc, ext, imm);



    case ADDLI_OPCODE_X1:

    case SHL16INSLI_OPCODE_X1:

    case ADDXLI_OPCODE_X1:

        imm = (int16_t)get_Imm16_X1(bundle);

        return gen_rri_opcode(dc, OE(opc, 0, X1), dest, srca, imm);



    default:

        return TILEGX_EXCP_OPCODE_UNIMPLEMENTED;

    }

}
