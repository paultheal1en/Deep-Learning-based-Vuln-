static int disas_neon_ls_insn(CPUState * env, DisasContext *s, uint32_t insn)

{

    int rd, rn, rm;

    int op;

    int nregs;

    int interleave;

    int spacing;

    int stride;

    int size;

    int reg;

    int pass;

    int load;

    int shift;

    int n;

    TCGv addr;

    TCGv tmp;

    TCGv tmp2;

    TCGv_i64 tmp64;



    if (!s->vfp_enabled)

      return 1;

    VFP_DREG_D(rd, insn);

    rn = (insn >> 16) & 0xf;

    rm = insn & 0xf;

    load = (insn & (1 << 21)) != 0;

    addr = tcg_temp_new_i32();

    if ((insn & (1 << 23)) == 0) {

        /* Load store all elements.  */

        op = (insn >> 8) & 0xf;

        size = (insn >> 6) & 3;

        if (op > 10)

            return 1;

        nregs = neon_ls_element_type[op].nregs;

        interleave = neon_ls_element_type[op].interleave;

        spacing = neon_ls_element_type[op].spacing;

        if (size == 3 && (interleave | spacing) != 1)

            return 1;

        load_reg_var(s, addr, rn);

        stride = (1 << size) * interleave;

        for (reg = 0; reg < nregs; reg++) {

            if (interleave > 2 || (interleave == 2 && nregs == 2)) {

                load_reg_var(s, addr, rn);

                tcg_gen_addi_i32(addr, addr, (1 << size) * reg);

            } else if (interleave == 2 && nregs == 4 && reg == 2) {

                load_reg_var(s, addr, rn);

                tcg_gen_addi_i32(addr, addr, 1 << size);

            }

            if (size == 3) {

                if (load) {

                    tmp64 = gen_ld64(addr, IS_USER(s));

                    neon_store_reg64(tmp64, rd);

                    tcg_temp_free_i64(tmp64);

                } else {

                    tmp64 = tcg_temp_new_i64();

                    neon_load_reg64(tmp64, rd);

                    gen_st64(tmp64, addr, IS_USER(s));

                }

                tcg_gen_addi_i32(addr, addr, stride);

            } else {

                for (pass = 0; pass < 2; pass++) {

                    if (size == 2) {

                        if (load) {

                            tmp = gen_ld32(addr, IS_USER(s));

                            neon_store_reg(rd, pass, tmp);

                        } else {

                            tmp = neon_load_reg(rd, pass);

                            gen_st32(tmp, addr, IS_USER(s));

                        }

                        tcg_gen_addi_i32(addr, addr, stride);

                    } else if (size == 1) {

                        if (load) {

                            tmp = gen_ld16u(addr, IS_USER(s));

                            tcg_gen_addi_i32(addr, addr, stride);

                            tmp2 = gen_ld16u(addr, IS_USER(s));

                            tcg_gen_addi_i32(addr, addr, stride);

                            tcg_gen_shli_i32(tmp2, tmp2, 16);

                            tcg_gen_or_i32(tmp, tmp, tmp2);

                            tcg_temp_free_i32(tmp2);

                            neon_store_reg(rd, pass, tmp);

                        } else {

                            tmp = neon_load_reg(rd, pass);

                            tmp2 = tcg_temp_new_i32();

                            tcg_gen_shri_i32(tmp2, tmp, 16);

                            gen_st16(tmp, addr, IS_USER(s));

                            tcg_gen_addi_i32(addr, addr, stride);

                            gen_st16(tmp2, addr, IS_USER(s));

                            tcg_gen_addi_i32(addr, addr, stride);

                        }

                    } else /* size == 0 */ {

                        if (load) {

                            TCGV_UNUSED(tmp2);

                            for (n = 0; n < 4; n++) {

                                tmp = gen_ld8u(addr, IS_USER(s));

                                tcg_gen_addi_i32(addr, addr, stride);

                                if (n == 0) {

                                    tmp2 = tmp;

                                } else {

                                    tcg_gen_shli_i32(tmp, tmp, n * 8);

                                    tcg_gen_or_i32(tmp2, tmp2, tmp);

                                    tcg_temp_free_i32(tmp);

                                }

                            }

                            neon_store_reg(rd, pass, tmp2);

                        } else {

                            tmp2 = neon_load_reg(rd, pass);

                            for (n = 0; n < 4; n++) {

                                tmp = tcg_temp_new_i32();

                                if (n == 0) {

                                    tcg_gen_mov_i32(tmp, tmp2);

                                } else {

                                    tcg_gen_shri_i32(tmp, tmp2, n * 8);

                                }

                                gen_st8(tmp, addr, IS_USER(s));

                                tcg_gen_addi_i32(addr, addr, stride);

                            }

                            tcg_temp_free_i32(tmp2);

                        }

                    }

                }

            }

            rd += spacing;

        }

        stride = nregs * 8;

    } else {

        size = (insn >> 10) & 3;

        if (size == 3) {

            /* Load single element to all lanes.  */

            int a = (insn >> 4) & 1;

            if (!load) {

                return 1;

            }

            size = (insn >> 6) & 3;

            nregs = ((insn >> 8) & 3) + 1;



            if (size == 3) {

                if (nregs != 4 || a == 0) {

                    return 1;

                }

                /* For VLD4 size==3 a == 1 means 32 bits at 16 byte alignment */

                size = 2;

            }

            if (nregs == 1 && a == 1 && size == 0) {

                return 1;

            }

            if (nregs == 3 && a == 1) {

                return 1;

            }

            load_reg_var(s, addr, rn);

            if (nregs == 1) {

                /* VLD1 to all lanes: bit 5 indicates how many Dregs to write */

                tmp = gen_load_and_replicate(s, addr, size);

                tcg_gen_st_i32(tmp, cpu_env, neon_reg_offset(rd, 0));

                tcg_gen_st_i32(tmp, cpu_env, neon_reg_offset(rd, 1));

                if (insn & (1 << 5)) {

                    tcg_gen_st_i32(tmp, cpu_env, neon_reg_offset(rd + 1, 0));

                    tcg_gen_st_i32(tmp, cpu_env, neon_reg_offset(rd + 1, 1));

                }

                tcg_temp_free_i32(tmp);

            } else {

                /* VLD2/3/4 to all lanes: bit 5 indicates register stride */

                stride = (insn & (1 << 5)) ? 2 : 1;

                for (reg = 0; reg < nregs; reg++) {

                    tmp = gen_load_and_replicate(s, addr, size);

                    tcg_gen_st_i32(tmp, cpu_env, neon_reg_offset(rd, 0));

                    tcg_gen_st_i32(tmp, cpu_env, neon_reg_offset(rd, 1));

                    tcg_temp_free_i32(tmp);

                    tcg_gen_addi_i32(addr, addr, 1 << size);

                    rd += stride;

                }

            }

            stride = (1 << size) * nregs;

        } else {

            /* Single element.  */

            pass = (insn >> 7) & 1;

            switch (size) {

            case 0:

                shift = ((insn >> 5) & 3) * 8;

                stride = 1;

                break;

            case 1:

                shift = ((insn >> 6) & 1) * 16;

                stride = (insn & (1 << 5)) ? 2 : 1;

                break;

            case 2:

                shift = 0;

                stride = (insn & (1 << 6)) ? 2 : 1;

                break;

            default:

                abort();

            }

            nregs = ((insn >> 8) & 3) + 1;

            load_reg_var(s, addr, rn);

            for (reg = 0; reg < nregs; reg++) {

                if (load) {

                    switch (size) {

                    case 0:

                        tmp = gen_ld8u(addr, IS_USER(s));

                        break;

                    case 1:

                        tmp = gen_ld16u(addr, IS_USER(s));

                        break;

                    case 2:

                        tmp = gen_ld32(addr, IS_USER(s));

                        break;

                    default: /* Avoid compiler warnings.  */

                        abort();

                    }

                    if (size != 2) {

                        tmp2 = neon_load_reg(rd, pass);

                        gen_bfi(tmp, tmp2, tmp, shift, size ? 0xffff : 0xff);

                        tcg_temp_free_i32(tmp2);

                    }

                    neon_store_reg(rd, pass, tmp);

                } else { /* Store */

                    tmp = neon_load_reg(rd, pass);

                    if (shift)

                        tcg_gen_shri_i32(tmp, tmp, shift);

                    switch (size) {

                    case 0:

                        gen_st8(tmp, addr, IS_USER(s));

                        break;

                    case 1:

                        gen_st16(tmp, addr, IS_USER(s));

                        break;

                    case 2:

                        gen_st32(tmp, addr, IS_USER(s));

                        break;

                    }

                }

                rd += stride;

                tcg_gen_addi_i32(addr, addr, 1 << size);

            }

            stride = nregs * (1 << size);

        }

    }

    tcg_temp_free_i32(addr);

    if (rm != 15) {

        TCGv base;



        base = load_reg(s, rn);

        if (rm == 13) {

            tcg_gen_addi_i32(base, base, stride);

        } else {

            TCGv index;

            index = load_reg(s, rm);

            tcg_gen_add_i32(base, base, index);

            tcg_temp_free_i32(index);

        }

        store_reg(s, rn, base);

    }

    return 0;

}
