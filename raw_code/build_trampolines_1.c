static void build_trampolines(TCGContext *s)

{

    static void * const qemu_ld_helpers[16] = {

        [MO_UB]   = helper_ret_ldub_mmu,

        [MO_SB]   = helper_ret_ldsb_mmu,

        [MO_LEUW] = helper_le_lduw_mmu,

        [MO_LESW] = helper_le_ldsw_mmu,

        [MO_LEUL] = helper_le_ldul_mmu,

        [MO_LEQ]  = helper_le_ldq_mmu,

        [MO_BEUW] = helper_be_lduw_mmu,

        [MO_BESW] = helper_be_ldsw_mmu,

        [MO_BEUL] = helper_be_ldul_mmu,

        [MO_BEQ]  = helper_be_ldq_mmu,

    };

    static void * const qemu_st_helpers[16] = {

        [MO_UB]   = helper_ret_stb_mmu,

        [MO_LEUW] = helper_le_stw_mmu,

        [MO_LEUL] = helper_le_stl_mmu,

        [MO_LEQ]  = helper_le_stq_mmu,

        [MO_BEUW] = helper_be_stw_mmu,

        [MO_BEUL] = helper_be_stl_mmu,

        [MO_BEQ]  = helper_be_stq_mmu,

    };



    int i;

    TCGReg ra;



    for (i = 0; i < 16; ++i) {

        if (qemu_ld_helpers[i] == NULL) {

            continue;

        }



        /* May as well align the trampoline.  */

        while ((uintptr_t)s->code_ptr & 15) {

            tcg_out_nop(s);

        }

        qemu_ld_trampoline[i] = s->code_ptr;



        if (SPARC64 || TARGET_LONG_BITS == 32) {

            ra = TCG_REG_O3;

        } else {

            /* Install the high part of the address.  */

            tcg_out_arithi(s, TCG_REG_O1, TCG_REG_O2, 32, SHIFT_SRLX);

            ra = TCG_REG_O4;

        }



        /* Set the retaddr operand.  */

        tcg_out_mov(s, TCG_TYPE_PTR, ra, TCG_REG_O7);

        /* Set the env operand.  */

        tcg_out_mov(s, TCG_TYPE_PTR, TCG_REG_O0, TCG_AREG0);

        /* Tail call.  */

        tcg_out_call_nodelay(s, qemu_ld_helpers[i]);

        tcg_out_mov(s, TCG_TYPE_PTR, TCG_REG_O7, ra);

    }



    for (i = 0; i < 16; ++i) {

        if (qemu_st_helpers[i] == NULL) {

            continue;

        }



        /* May as well align the trampoline.  */

        while ((uintptr_t)s->code_ptr & 15) {

            tcg_out_nop(s);

        }

        qemu_st_trampoline[i] = s->code_ptr;



        if (SPARC64) {

            emit_extend(s, TCG_REG_O2, i);

            ra = TCG_REG_O4;

        } else {

            ra = TCG_REG_O1;

            if (TARGET_LONG_BITS == 64) {

                /* Install the high part of the address.  */

                tcg_out_arithi(s, ra, ra + 1, 32, SHIFT_SRLX);

                ra += 2;

            } else {

                ra += 1;

            }

            if ((i & MO_SIZE) == MO_64) {

                /* Install the high part of the data.  */

                tcg_out_arithi(s, ra, ra + 1, 32, SHIFT_SRLX);

                ra += 2;

            } else {


                ra += 1;

            }

            /* Skip the oi argument.  */

            ra += 1;

        }

                

        /* Set the retaddr operand.  */

        if (ra >= TCG_REG_O6) {

            tcg_out_st(s, TCG_TYPE_PTR, TCG_REG_O7, TCG_REG_CALL_STACK,

                       TCG_TARGET_CALL_STACK_OFFSET);

            ra = TCG_REG_G1;

        }

        tcg_out_mov(s, TCG_TYPE_PTR, ra, TCG_REG_O7);

        /* Set the env operand.  */

        tcg_out_mov(s, TCG_TYPE_PTR, TCG_REG_O0, TCG_AREG0);

        /* Tail call.  */

        tcg_out_call_nodelay(s, qemu_st_helpers[i]);

        tcg_out_mov(s, TCG_TYPE_PTR, TCG_REG_O7, ra);

    }

}