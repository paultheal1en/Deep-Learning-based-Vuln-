static void disas_simd_mod_imm(DisasContext *s, uint32_t insn)

{

    int rd = extract32(insn, 0, 5);

    int cmode = extract32(insn, 12, 4);

    int cmode_3_1 = extract32(cmode, 1, 3);

    int cmode_0 = extract32(cmode, 0, 1);

    int o2 = extract32(insn, 11, 1);

    uint64_t abcdefgh = extract32(insn, 5, 5) | (extract32(insn, 16, 3) << 5);

    bool is_neg = extract32(insn, 29, 1);

    bool is_q = extract32(insn, 30, 1);

    uint64_t imm = 0;

    TCGv_i64 tcg_rd, tcg_imm;

    int i;



    if (o2 != 0 || ((cmode == 0xf) && is_neg && !is_q)) {

        unallocated_encoding(s);

        return;

    }



    if (!fp_access_check(s)) {

        return;

    }



    /* See AdvSIMDExpandImm() in ARM ARM */

    switch (cmode_3_1) {

    case 0: /* Replicate(Zeros(24):imm8, 2) */

    case 1: /* Replicate(Zeros(16):imm8:Zeros(8), 2) */

    case 2: /* Replicate(Zeros(8):imm8:Zeros(16), 2) */

    case 3: /* Replicate(imm8:Zeros(24), 2) */

    {

        int shift = cmode_3_1 * 8;

        imm = bitfield_replicate(abcdefgh << shift, 32);

        break;

    }

    case 4: /* Replicate(Zeros(8):imm8, 4) */

    case 5: /* Replicate(imm8:Zeros(8), 4) */

    {

        int shift = (cmode_3_1 & 0x1) * 8;

        imm = bitfield_replicate(abcdefgh << shift, 16);

        break;

    }

    case 6:

        if (cmode_0) {

            /* Replicate(Zeros(8):imm8:Ones(16), 2) */

            imm = (abcdefgh << 16) | 0xffff;

        } else {

            /* Replicate(Zeros(16):imm8:Ones(8), 2) */

            imm = (abcdefgh << 8) | 0xff;

        }

        imm = bitfield_replicate(imm, 32);

        break;

    case 7:

        if (!cmode_0 && !is_neg) {

            imm = bitfield_replicate(abcdefgh, 8);

        } else if (!cmode_0 && is_neg) {

            int i;

            imm = 0;

            for (i = 0; i < 8; i++) {

                if ((abcdefgh) & (1 << i)) {

                    imm |= 0xffULL << (i * 8);

                }

            }

        } else if (cmode_0) {

            if (is_neg) {

                imm = (abcdefgh & 0x3f) << 48;

                if (abcdefgh & 0x80) {

                    imm |= 0x8000000000000000ULL;

                }

                if (abcdefgh & 0x40) {

                    imm |= 0x3fc0000000000000ULL;

                } else {

                    imm |= 0x4000000000000000ULL;

                }

            } else {

                imm = (abcdefgh & 0x3f) << 19;

                if (abcdefgh & 0x80) {

                    imm |= 0x80000000;

                }

                if (abcdefgh & 0x40) {

                    imm |= 0x3e000000;

                } else {

                    imm |= 0x40000000;

                }

                imm |= (imm << 32);

            }

        }

        break;

    }



    if (cmode_3_1 != 7 && is_neg) {

        imm = ~imm;

    }



    tcg_imm = tcg_const_i64(imm);

    tcg_rd = new_tmp_a64(s);



    for (i = 0; i < 2; i++) {

        int foffs = i ? fp_reg_hi_offset(rd) : fp_reg_offset(rd, MO_64);



        if (i == 1 && !is_q) {

            /* non-quad ops clear high half of vector */

            tcg_gen_movi_i64(tcg_rd, 0);

        } else if ((cmode & 0x9) == 0x1 || (cmode & 0xd) == 0x9) {

            tcg_gen_ld_i64(tcg_rd, cpu_env, foffs);

            if (is_neg) {

                /* AND (BIC) */

                tcg_gen_and_i64(tcg_rd, tcg_rd, tcg_imm);

            } else {

                /* ORR */

                tcg_gen_or_i64(tcg_rd, tcg_rd, tcg_imm);

            }

        } else {

            /* MOVI */

            tcg_gen_mov_i64(tcg_rd, tcg_imm);

        }

        tcg_gen_st_i64(tcg_rd, cpu_env, foffs);

    }



    tcg_temp_free_i64(tcg_imm);

}
