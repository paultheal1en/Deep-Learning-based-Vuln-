static void disas_simd_3same_logic(DisasContext *s, uint32_t insn)

{

    int rd = extract32(insn, 0, 5);

    int rn = extract32(insn, 5, 5);

    int rm = extract32(insn, 16, 5);

    int size = extract32(insn, 22, 2);

    bool is_u = extract32(insn, 29, 1);

    bool is_q = extract32(insn, 30, 1);

    TCGv_i64 tcg_op1 = tcg_temp_new_i64();

    TCGv_i64 tcg_op2 = tcg_temp_new_i64();

    TCGv_i64 tcg_res[2];

    int pass;



    tcg_res[0] = tcg_temp_new_i64();

    tcg_res[1] = tcg_temp_new_i64();



    for (pass = 0; pass < (is_q ? 2 : 1); pass++) {

        read_vec_element(s, tcg_op1, rn, pass, MO_64);

        read_vec_element(s, tcg_op2, rm, pass, MO_64);



        if (!is_u) {

            switch (size) {

            case 0: /* AND */

                tcg_gen_and_i64(tcg_res[pass], tcg_op1, tcg_op2);

                break;

            case 1: /* BIC */

                tcg_gen_andc_i64(tcg_res[pass], tcg_op1, tcg_op2);

                break;

            case 2: /* ORR */

                tcg_gen_or_i64(tcg_res[pass], tcg_op1, tcg_op2);

                break;

            case 3: /* ORN */

                tcg_gen_orc_i64(tcg_res[pass], tcg_op1, tcg_op2);

                break;

            }

        } else {

            if (size != 0) {

                /* B* ops need res loaded to operate on */

                read_vec_element(s, tcg_res[pass], rd, pass, MO_64);

            }



            switch (size) {

            case 0: /* EOR */

                tcg_gen_xor_i64(tcg_res[pass], tcg_op1, tcg_op2);

                break;

            case 1: /* BSL bitwise select */

                tcg_gen_xor_i64(tcg_op1, tcg_op1, tcg_op2);

                tcg_gen_and_i64(tcg_op1, tcg_op1, tcg_res[pass]);

                tcg_gen_xor_i64(tcg_res[pass], tcg_op2, tcg_op1);

                break;

            case 2: /* BIT, bitwise insert if true */

                tcg_gen_xor_i64(tcg_op1, tcg_op1, tcg_res[pass]);

                tcg_gen_and_i64(tcg_op1, tcg_op1, tcg_op2);

                tcg_gen_xor_i64(tcg_res[pass], tcg_res[pass], tcg_op1);

                break;

            case 3: /* BIF, bitwise insert if false */

                tcg_gen_xor_i64(tcg_op1, tcg_op1, tcg_res[pass]);

                tcg_gen_andc_i64(tcg_op1, tcg_op1, tcg_op2);

                tcg_gen_xor_i64(tcg_res[pass], tcg_res[pass], tcg_op1);

                break;

            }

        }

    }



    write_vec_element(s, tcg_res[0], rd, 0, MO_64);

    if (!is_q) {

        tcg_gen_movi_i64(tcg_res[1], 0);

    }

    write_vec_element(s, tcg_res[1], rd, 1, MO_64);



    tcg_temp_free_i64(tcg_op1);

    tcg_temp_free_i64(tcg_op2);

    tcg_temp_free_i64(tcg_res[0]);

    tcg_temp_free_i64(tcg_res[1]);

}
