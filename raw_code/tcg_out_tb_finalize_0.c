static void tcg_out_tb_finalize(TCGContext *s)

{

    static const void * const helpers[8] = {

        helper_ret_stb_mmu,

        helper_le_stw_mmu,

        helper_le_stl_mmu,

        helper_le_stq_mmu,

        helper_ret_ldub_mmu,

        helper_le_lduw_mmu,

        helper_le_ldul_mmu,

        helper_le_ldq_mmu,

    };

    tcg_insn_unit *thunks[8] = { };

    TCGLabelQemuLdst *l;



    for (l = s->be->labels; l != NULL; l = l->next) {

        long x = l->is_ld * 4 + l->size;

        tcg_insn_unit *dest = thunks[x];



        /* The out-of-line thunks are all the same; load the return address

           from B0, load the GP, and branch to the code.  Note that we are

           always post-call, so the register window has rolled, so we're

           using incoming parameter register numbers, not outgoing.  */

        if (dest == NULL) {

            uintptr_t *desc = (uintptr_t *)helpers[x];

            uintptr_t func = desc[0], gp = desc[1], disp;



            thunks[x] = dest = s->code_ptr;



            tcg_out_bundle(s, mlx,

                           INSN_NOP_M,

                           tcg_opc_l2 (gp),

                           tcg_opc_x2 (TCG_REG_P0, OPC_MOVL_X2,

                                       TCG_REG_R1, gp));

            tcg_out_bundle(s, mii,

                           INSN_NOP_M,

                           INSN_NOP_I,

                           tcg_opc_i22(TCG_REG_P0, OPC_MOV_I22,

                                       l->is_ld ? TCG_REG_R35 : TCG_REG_R36,

                                       TCG_REG_B0));

            disp = (tcg_insn_unit *)func - s->code_ptr;

            tcg_out_bundle(s, mLX,

                           INSN_NOP_M,

                           tcg_opc_l3 (disp),

                           tcg_opc_x3 (TCG_REG_P0, OPC_BRL_SPTK_MANY_X3, disp));

        }



        reloc_pcrel21b_slot2(l->label_ptr, dest);

    }

}
