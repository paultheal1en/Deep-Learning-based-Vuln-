static void disas_ldst_reg_imm9(DisasContext *s, uint32_t insn)

{

    int rt = extract32(insn, 0, 5);

    int rn = extract32(insn, 5, 5);

    int imm9 = sextract32(insn, 12, 9);

    int opc = extract32(insn, 22, 2);

    int size = extract32(insn, 30, 2);

    int idx = extract32(insn, 10, 2);

    bool is_signed = false;

    bool is_store = false;

    bool is_extended = false;

    bool is_unpriv = (idx == 2);

    bool is_vector = extract32(insn, 26, 1);

    bool post_index;

    bool writeback;



    TCGv_i64 tcg_addr;



    if (is_vector) {

        size |= (opc & 2) << 1;

        if (size > 4 || is_unpriv) {

            unallocated_encoding(s);

            return;

        }

        is_store = ((opc & 1) == 0);

        if (!fp_access_check(s)) {

            return;

        }

    } else {

        if (size == 3 && opc == 2) {

            /* PRFM - prefetch */

            if (is_unpriv) {

                unallocated_encoding(s);

                return;

            }

            return;

        }

        if (opc == 3 && size > 1) {

            unallocated_encoding(s);

            return;

        }

        is_store = (opc == 0);

        is_signed = opc & (1<<1);

        is_extended = (size < 3) && (opc & 1);

    }



    switch (idx) {

    case 0:

    case 2:

        post_index = false;

        writeback = false;

        break;

    case 1:

        post_index = true;

        writeback = true;

        break;

    case 3:

        post_index = false;

        writeback = true;

        break;

    }



    if (rn == 31) {

        gen_check_sp_alignment(s);

    }

    tcg_addr = read_cpu_reg_sp(s, rn, 1);



    if (!post_index) {

        tcg_gen_addi_i64(tcg_addr, tcg_addr, imm9);

    }



    if (is_vector) {

        if (is_store) {

            do_fp_st(s, rt, tcg_addr, size);

        } else {

            do_fp_ld(s, rt, tcg_addr, size);

        }

    } else {

        TCGv_i64 tcg_rt = cpu_reg(s, rt);

        int memidx = is_unpriv ? 1 : get_mem_index(s);



        if (is_store) {

            do_gpr_st_memidx(s, tcg_rt, tcg_addr, size, memidx);

        } else {

            do_gpr_ld_memidx(s, tcg_rt, tcg_addr, size,

                             is_signed, is_extended, memidx);

        }

    }



    if (writeback) {

        TCGv_i64 tcg_rn = cpu_reg_sp(s, rn);

        if (post_index) {

            tcg_gen_addi_i64(tcg_addr, tcg_addr, imm9);

        }

        tcg_gen_mov_i64(tcg_rn, tcg_addr);

    }

}
