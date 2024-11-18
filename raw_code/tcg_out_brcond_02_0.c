static inline void tcg_out_brcond(TCGContext *s, TCGCond cond, TCGReg arg1,

                                  TCGReg arg2, int label_index, int cmp4)

{

    TCGLabel *l = &s->labels[label_index];

    uint64_t imm;



    /* We pay attention here to not modify the branch target by reading

       the existing value and using it again. This ensure that caches and

       memory are kept coherent during retranslation. */

    if (l->has_value) {

        imm = l->u.value_ptr - s->code_ptr;

    } else {

        imm = get_reloc_pcrel21b_slot2(s->code_ptr);

        tcg_out_reloc(s, s->code_ptr, R_IA64_PCREL21B, label_index, 0);

    }



    tcg_out_bundle(s, miB,

                   INSN_NOP_M,

                   tcg_opc_cmp_a(TCG_REG_P0, cond, arg1, arg2, cmp4),

                   tcg_opc_b1(TCG_REG_P6, OPC_BR_DPTK_FEW_B1, imm));

}
