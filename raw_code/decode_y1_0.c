static TileExcp decode_y1(DisasContext *dc, tilegx_bundle_bits bundle)

{

    unsigned opc = get_Opcode_Y1(bundle);

    unsigned ext = get_RRROpcodeExtension_Y1(bundle);

    unsigned dest = get_Dest_Y1(bundle);

    unsigned srca = get_SrcA_Y1(bundle);

    unsigned srcb;

    int imm;



    switch (get_Opcode_Y1(bundle)) {

    case RRR_1_OPCODE_Y1:

        if (ext == UNARY_RRR_1_OPCODE_Y0) {

            ext = get_UnaryOpcodeExtension_Y1(bundle);

            return gen_rr_opcode(dc, OE(opc, ext, Y1), dest, srca);

        }

        /* fallthru */

    case RRR_0_OPCODE_Y1:

    case RRR_2_OPCODE_Y1:

    case RRR_3_OPCODE_Y1:

    case RRR_4_OPCODE_Y1:

    case RRR_5_OPCODE_Y1:

    case RRR_6_OPCODE_Y1:

    case RRR_7_OPCODE_Y1:

        srcb = get_SrcB_Y1(bundle);

        return gen_rrr_opcode(dc, OE(opc, ext, Y1), dest, srca, srcb);



    case SHIFT_OPCODE_Y1:

        ext = get_ShiftOpcodeExtension_Y1(bundle);

        imm = get_ShAmt_Y1(bundle);

        return gen_rri_opcode(dc, OE(opc, ext, Y1), dest, srca, imm);



    case ADDI_OPCODE_Y1:

    case ADDXI_OPCODE_Y1:

    case ANDI_OPCODE_Y1:

    case CMPEQI_OPCODE_Y1:

    case CMPLTSI_OPCODE_Y1:

        imm = (int8_t)get_Imm8_Y1(bundle);

        return gen_rri_opcode(dc, OE(opc, 0, Y1), dest, srca, imm);



    default:

        return TILEGX_EXCP_OPCODE_UNIMPLEMENTED;

    }

}
