static void dec_misc(DisasContext *dc, uint32_t insn)

{

    uint32_t op0, op1;

    uint32_t ra, rb, rd;

#ifdef OPENRISC_DISAS

    uint32_t L6, K5;

#endif

    uint32_t I16, I5, I11, N26, tmp;

    TCGMemOp mop;



    op0 = extract32(insn, 26, 6);

    op1 = extract32(insn, 24, 2);

    ra = extract32(insn, 16, 5);

    rb = extract32(insn, 11, 5);

    rd = extract32(insn, 21, 5);

#ifdef OPENRISC_DISAS

    L6 = extract32(insn, 5, 6);

    K5 = extract32(insn, 0, 5);

#endif

    I16 = extract32(insn, 0, 16);

    I5 = extract32(insn, 21, 5);

    I11 = extract32(insn, 0, 11);

    N26 = extract32(insn, 0, 26);

    tmp = (I5<<11) + I11;



    switch (op0) {

    case 0x00:    /* l.j */

        LOG_DIS("l.j %d\n", N26);

        gen_jump(dc, N26, 0, op0);

        break;



    case 0x01:    /* l.jal */

        LOG_DIS("l.jal %d\n", N26);

        gen_jump(dc, N26, 0, op0);

        break;



    case 0x03:    /* l.bnf */

        LOG_DIS("l.bnf %d\n", N26);

        gen_jump(dc, N26, 0, op0);

        break;



    case 0x04:    /* l.bf */

        LOG_DIS("l.bf %d\n", N26);

        gen_jump(dc, N26, 0, op0);

        break;



    case 0x05:

        switch (op1) {

        case 0x01:    /* l.nop */

            LOG_DIS("l.nop %d\n", I16);

            break;



        default:

            gen_illegal_exception(dc);

            break;

        }

        break;



    case 0x11:    /* l.jr */

        LOG_DIS("l.jr r%d\n", rb);

         gen_jump(dc, 0, rb, op0);

         break;



    case 0x12:    /* l.jalr */

        LOG_DIS("l.jalr r%d\n", rb);

        gen_jump(dc, 0, rb, op0);

        break;



    case 0x13:    /* l.maci */

        LOG_DIS("l.maci %d, r%d, %d\n", I5, ra, I11);

        {

            TCGv_i64 t1 = tcg_temp_new_i64();

            TCGv_i64 t2 = tcg_temp_new_i64();

            TCGv_i32 dst = tcg_temp_new_i32();

            TCGv ttmp = tcg_const_tl(tmp);

            tcg_gen_mul_tl(dst, cpu_R[ra], ttmp);

            tcg_gen_ext_i32_i64(t1, dst);

            tcg_gen_concat_i32_i64(t2, maclo, machi);

            tcg_gen_add_i64(t2, t2, t1);

            tcg_gen_trunc_i64_i32(maclo, t2);

            tcg_gen_shri_i64(t2, t2, 32);

            tcg_gen_trunc_i64_i32(machi, t2);

            tcg_temp_free_i32(dst);

            tcg_temp_free(ttmp);

            tcg_temp_free_i64(t1);

            tcg_temp_free_i64(t2);

        }

        break;



    case 0x09:    /* l.rfe */

        LOG_DIS("l.rfe\n");

        {

#if defined(CONFIG_USER_ONLY)

            return;

#else

            if (dc->mem_idx == MMU_USER_IDX) {

                gen_illegal_exception(dc);

                return;

            }

            gen_helper_rfe(cpu_env);

            dc->is_jmp = DISAS_UPDATE;

#endif

        }

        break;



    case 0x1c:    /* l.cust1 */

        LOG_DIS("l.cust1\n");

        break;



    case 0x1d:    /* l.cust2 */

        LOG_DIS("l.cust2\n");

        break;



    case 0x1e:    /* l.cust3 */

        LOG_DIS("l.cust3\n");

        break;



    case 0x1f:    /* l.cust4 */

        LOG_DIS("l.cust4\n");

        break;



    case 0x3c:    /* l.cust5 */

        LOG_DIS("l.cust5 r%d, r%d, r%d, %d, %d\n", rd, ra, rb, L6, K5);

        break;



    case 0x3d:    /* l.cust6 */

        LOG_DIS("l.cust6\n");

        break;



    case 0x3e:    /* l.cust7 */

        LOG_DIS("l.cust7\n");

        break;



    case 0x3f:    /* l.cust8 */

        LOG_DIS("l.cust8\n");

        break;



/* not used yet, open it when we need or64.  */

/*#ifdef TARGET_OPENRISC64

    case 0x20:     l.ld

        LOG_DIS("l.ld r%d, r%d, %d\n", rd, ra, I16);

        check_ob64s(dc);

        mop = MO_TEQ;

        goto do_load;

#endif*/



    case 0x21:    /* l.lwz */

        LOG_DIS("l.lwz r%d, r%d, %d\n", rd, ra, I16);

        mop = MO_TEUL;

        goto do_load;



    case 0x22:    /* l.lws */

        LOG_DIS("l.lws r%d, r%d, %d\n", rd, ra, I16);

        mop = MO_TESL;

        goto do_load;



    case 0x23:    /* l.lbz */

        LOG_DIS("l.lbz r%d, r%d, %d\n", rd, ra, I16);

        mop = MO_UB;

        goto do_load;



    case 0x24:    /* l.lbs */

        LOG_DIS("l.lbs r%d, r%d, %d\n", rd, ra, I16);

        mop = MO_SB;

        goto do_load;



    case 0x25:    /* l.lhz */

        LOG_DIS("l.lhz r%d, r%d, %d\n", rd, ra, I16);

        mop = MO_TEUW;

        goto do_load;



    case 0x26:    /* l.lhs */

        LOG_DIS("l.lhs r%d, r%d, %d\n", rd, ra, I16);

        mop = MO_TESW;

        goto do_load;



    do_load:

        {

            TCGv t0 = tcg_temp_new();

            tcg_gen_addi_tl(t0, cpu_R[ra], sign_extend(I16, 16));

            tcg_gen_qemu_ld_tl(cpu_R[rd], t0, dc->mem_idx, mop);

            tcg_temp_free(t0);

        }

        break;



    case 0x27:    /* l.addi */

        LOG_DIS("l.addi r%d, r%d, %d\n", rd, ra, I16);

        {

            if (I16 == 0) {

                tcg_gen_mov_tl(cpu_R[rd], cpu_R[ra]);

            } else {

                int lab = gen_new_label();

                TCGv_i64 ta = tcg_temp_new_i64();

                TCGv_i64 td = tcg_temp_local_new_i64();

                TCGv_i32 res = tcg_temp_local_new_i32();

                TCGv_i32 sr_ove = tcg_temp_local_new_i32();

                tcg_gen_extu_i32_i64(ta, cpu_R[ra]);

                tcg_gen_addi_i64(td, ta, sign_extend(I16, 16));

                tcg_gen_trunc_i64_i32(res, td);

                tcg_gen_shri_i64(td, td, 32);

                tcg_gen_andi_i64(td, td, 0x3);

                /* Jump to lab when no overflow.  */

                tcg_gen_brcondi_i64(TCG_COND_EQ, td, 0x0, lab);

                tcg_gen_brcondi_i64(TCG_COND_EQ, td, 0x3, lab);

                tcg_gen_ori_i32(cpu_sr, cpu_sr, (SR_OV | SR_CY));

                tcg_gen_andi_i32(sr_ove, cpu_sr, SR_OVE);

                tcg_gen_brcondi_i32(TCG_COND_NE, sr_ove, SR_OVE, lab);

                gen_exception(dc, EXCP_RANGE);

                gen_set_label(lab);

                tcg_gen_mov_i32(cpu_R[rd], res);

                tcg_temp_free_i64(ta);

                tcg_temp_free_i64(td);

                tcg_temp_free_i32(res);

                tcg_temp_free_i32(sr_ove);

            }

        }

        break;



    case 0x28:    /* l.addic */

        LOG_DIS("l.addic r%d, r%d, %d\n", rd, ra, I16);

        {

            int lab = gen_new_label();

            TCGv_i64 ta = tcg_temp_new_i64();

            TCGv_i64 td = tcg_temp_local_new_i64();

            TCGv_i64 tcy = tcg_temp_local_new_i64();

            TCGv_i32 res = tcg_temp_local_new_i32();

            TCGv_i32 sr_cy = tcg_temp_local_new_i32();

            TCGv_i32 sr_ove = tcg_temp_local_new_i32();

            tcg_gen_extu_i32_i64(ta, cpu_R[ra]);

            tcg_gen_andi_i32(sr_cy, cpu_sr, SR_CY);

            tcg_gen_shri_i32(sr_cy, sr_cy, 10);

            tcg_gen_extu_i32_i64(tcy, sr_cy);

            tcg_gen_addi_i64(td, ta, sign_extend(I16, 16));

            tcg_gen_add_i64(td, td, tcy);

            tcg_gen_trunc_i64_i32(res, td);

            tcg_gen_shri_i64(td, td, 32);

            tcg_gen_andi_i64(td, td, 0x3);

            /* Jump to lab when no overflow.  */

            tcg_gen_brcondi_i64(TCG_COND_EQ, td, 0x0, lab);

            tcg_gen_brcondi_i64(TCG_COND_EQ, td, 0x3, lab);

            tcg_gen_ori_i32(cpu_sr, cpu_sr, (SR_OV | SR_CY));

            tcg_gen_andi_i32(sr_ove, cpu_sr, SR_OVE);

            tcg_gen_brcondi_i32(TCG_COND_NE, sr_ove, SR_OVE, lab);

            gen_exception(dc, EXCP_RANGE);

            gen_set_label(lab);

            tcg_gen_mov_i32(cpu_R[rd], res);

            tcg_temp_free_i64(ta);

            tcg_temp_free_i64(td);

            tcg_temp_free_i64(tcy);

            tcg_temp_free_i32(res);

            tcg_temp_free_i32(sr_cy);

            tcg_temp_free_i32(sr_ove);

        }

        break;



    case 0x29:    /* l.andi */

        LOG_DIS("l.andi r%d, r%d, %d\n", rd, ra, I16);

        tcg_gen_andi_tl(cpu_R[rd], cpu_R[ra], zero_extend(I16, 16));

        break;



    case 0x2a:    /* l.ori */

        LOG_DIS("l.ori r%d, r%d, %d\n", rd, ra, I16);

        tcg_gen_ori_tl(cpu_R[rd], cpu_R[ra], zero_extend(I16, 16));

        break;



    case 0x2b:    /* l.xori */

        LOG_DIS("l.xori r%d, r%d, %d\n", rd, ra, I16);

        tcg_gen_xori_tl(cpu_R[rd], cpu_R[ra], sign_extend(I16, 16));

        break;



    case 0x2c:    /* l.muli */

        LOG_DIS("l.muli r%d, r%d, %d\n", rd, ra, I16);

        if (ra != 0 && I16 != 0) {

            TCGv_i32 im = tcg_const_i32(I16);

            gen_helper_mul32(cpu_R[rd], cpu_env, cpu_R[ra], im);

            tcg_temp_free_i32(im);

        } else {

            tcg_gen_movi_tl(cpu_R[rd], 0x0);

        }

        break;



    case 0x2d:    /* l.mfspr */

        LOG_DIS("l.mfspr r%d, r%d, %d\n", rd, ra, I16);

        {

#if defined(CONFIG_USER_ONLY)

            return;

#else

            TCGv_i32 ti = tcg_const_i32(I16);

            if (dc->mem_idx == MMU_USER_IDX) {

                gen_illegal_exception(dc);

                return;

            }

            gen_helper_mfspr(cpu_R[rd], cpu_env, cpu_R[rd], cpu_R[ra], ti);

            tcg_temp_free_i32(ti);

#endif

        }

        break;



    case 0x30:    /* l.mtspr */

        LOG_DIS("l.mtspr %d, r%d, r%d, %d\n", I5, ra, rb, I11);

        {

#if defined(CONFIG_USER_ONLY)

            return;

#else

            TCGv_i32 im = tcg_const_i32(tmp);

            if (dc->mem_idx == MMU_USER_IDX) {

                gen_illegal_exception(dc);

                return;

            }

            gen_helper_mtspr(cpu_env, cpu_R[ra], cpu_R[rb], im);

            tcg_temp_free_i32(im);

#endif

        }

        break;



/* not used yet, open it when we need or64.  */

/*#ifdef TARGET_OPENRISC64

    case 0x34:     l.sd

        LOG_DIS("l.sd %d, r%d, r%d, %d\n", I5, ra, rb, I11);

        check_ob64s(dc);

        mop = MO_TEQ;

        goto do_store;

#endif*/



    case 0x35:    /* l.sw */

        LOG_DIS("l.sw %d, r%d, r%d, %d\n", I5, ra, rb, I11);

        mop = MO_TEUL;

        goto do_store;



    case 0x36:    /* l.sb */

        LOG_DIS("l.sb %d, r%d, r%d, %d\n", I5, ra, rb, I11);

        mop = MO_UB;

        goto do_store;



    case 0x37:    /* l.sh */

        LOG_DIS("l.sh %d, r%d, r%d, %d\n", I5, ra, rb, I11);

        mop = MO_TEUW;

        goto do_store;



    do_store:

        {

            TCGv t0 = tcg_temp_new();

            tcg_gen_addi_tl(t0, cpu_R[ra], sign_extend(tmp, 16));

            tcg_gen_qemu_st_tl(cpu_R[rb], t0, dc->mem_idx, mop);

            tcg_temp_free(t0);

        }

        break;



    default:

        gen_illegal_exception(dc);

        break;

    }

}
