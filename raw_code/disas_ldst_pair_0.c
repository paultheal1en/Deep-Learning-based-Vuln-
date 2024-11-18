static void disas_ldst_pair(DisasContext *s, uint32_t insn)

{

    int rt = extract32(insn, 0, 5);

    int rn = extract32(insn, 5, 5);

    int rt2 = extract32(insn, 10, 5);

    int64_t offset = sextract32(insn, 15, 7);

    int index = extract32(insn, 23, 2);

    bool is_vector = extract32(insn, 26, 1);

    bool is_load = extract32(insn, 22, 1);

    int opc = extract32(insn, 30, 2);



    bool is_signed = false;

    bool postindex = false;

    bool wback = false;



    TCGv_i64 tcg_addr; /* calculated address */

    int size;



    if (opc == 3) {

        unallocated_encoding(s);

        return;

    }



    if (is_vector) {

        size = 2 + opc;

    } else {

        size = 2 + extract32(opc, 1, 1);

        is_signed = extract32(opc, 0, 1);

        if (!is_load && is_signed) {

            unallocated_encoding(s);

            return;

        }

    }



    switch (index) {

    case 1: /* post-index */

        postindex = true;

        wback = true;

        break;

    case 0:

        /* signed offset with "non-temporal" hint. Since we don't emulate

         * caches we don't care about hints to the cache system about

         * data access patterns, and handle this identically to plain

         * signed offset.

         */

        if (is_signed) {

            /* There is no non-temporal-hint version of LDPSW */

            unallocated_encoding(s);

            return;

        }

        postindex = false;

        break;

    case 2: /* signed offset, rn not updated */

        postindex = false;

        break;

    case 3: /* pre-index */

        postindex = false;

        wback = true;

        break;

    }



    if (is_vector && !fp_access_check(s)) {

        return;

    }



    offset <<= size;



    if (rn == 31) {

        gen_check_sp_alignment(s);

    }



    tcg_addr = read_cpu_reg_sp(s, rn, 1);



    if (!postindex) {

        tcg_gen_addi_i64(tcg_addr, tcg_addr, offset);

    }



    if (is_vector) {

        if (is_load) {

            do_fp_ld(s, rt, tcg_addr, size);

        } else {

            do_fp_st(s, rt, tcg_addr, size);

        }

    } else {

        TCGv_i64 tcg_rt = cpu_reg(s, rt);

        if (is_load) {

            do_gpr_ld(s, tcg_rt, tcg_addr, size, is_signed, false);

        } else {

            do_gpr_st(s, tcg_rt, tcg_addr, size);

        }

    }

    tcg_gen_addi_i64(tcg_addr, tcg_addr, 1 << size);

    if (is_vector) {

        if (is_load) {

            do_fp_ld(s, rt2, tcg_addr, size);

        } else {

            do_fp_st(s, rt2, tcg_addr, size);

        }

    } else {

        TCGv_i64 tcg_rt2 = cpu_reg(s, rt2);

        if (is_load) {

            do_gpr_ld(s, tcg_rt2, tcg_addr, size, is_signed, false);

        } else {

            do_gpr_st(s, tcg_rt2, tcg_addr, size);

        }

    }



    if (wback) {

        if (postindex) {

            tcg_gen_addi_i64(tcg_addr, tcg_addr, offset - (1 << size));

        } else {

            tcg_gen_subi_i64(tcg_addr, tcg_addr, 1 << size);

        }

        tcg_gen_mov_i64(cpu_reg_sp(s, rn), tcg_addr);

    }

}
