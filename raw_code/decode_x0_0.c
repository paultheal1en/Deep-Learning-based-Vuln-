static TileExcp decode_x0(DisasContext *dc, tilegx_bundle_bits bundle)

{

    unsigned opc = get_Opcode_X0(bundle);

    unsigned dest = get_Dest_X0(bundle);

    unsigned srca = get_SrcA_X0(bundle);

    unsigned ext, srcb, bfs, bfe;

    int imm;



    switch (opc) {

    case RRR_0_OPCODE_X0:

        ext = get_RRROpcodeExtension_X0(bundle);

        if (ext == UNARY_RRR_0_OPCODE_X0) {

            ext = get_UnaryOpcodeExtension_X0(bundle);

            return gen_rr_opcode(dc, OE(opc, ext, X0), dest, srca);

        }

        srcb = get_SrcB_X0(bundle);

        return gen_rrr_opcode(dc, OE(opc, ext, X0), dest, srca, srcb);



    case SHIFT_OPCODE_X0:

        ext = get_ShiftOpcodeExtension_X0(bundle);

        imm = get_ShAmt_X0(bundle);

        return gen_rri_opcode(dc, OE(opc, ext, X0), dest, srca, imm);



    case IMM8_OPCODE_X0:

        ext = get_Imm8OpcodeExtension_X0(bundle);

        imm = (int8_t)get_Imm8_X0(bundle);

        return gen_rri_opcode(dc, OE(opc, ext, X0), dest, srca, imm);



    case BF_OPCODE_X0:

        ext = get_BFOpcodeExtension_X0(bundle);

        bfs = get_BFStart_X0(bundle);

        bfe = get_BFEnd_X0(bundle);

        return gen_bf_opcode_x0(dc, ext, dest, srca, bfs, bfe);



    case ADDLI_OPCODE_X0:

    case SHL16INSLI_OPCODE_X0:

    case ADDXLI_OPCODE_X0:

        imm = (int16_t)get_Imm16_X0(bundle);

        return gen_rri_opcode(dc, OE(opc, 0, X0), dest, srca, imm);



    default:

        return TILEGX_EXCP_OPCODE_UNIMPLEMENTED;

    }

}
