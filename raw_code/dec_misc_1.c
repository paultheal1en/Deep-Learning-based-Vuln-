static void dec_misc(DisasContext *dc, uint32_t insn)

{

    uint32_t op0, op1;

    uint32_t ra, rb, rd;

    uint32_t L6, K5, K16, K5_11;

    int32_t I16, I5_11, N26;

    TCGMemOp mop;

    TCGv t0;



    op0 = extract32(insn, 26, 6);

    op1 = extract32(insn, 24, 2);

    ra = extract32(insn, 16, 5);

    rb = extract32(insn, 11, 5);

    rd = extract32(insn, 21, 5);

    L6 = extract32(insn, 5, 6);

    K5 = extract32(insn, 0, 5);

    K16 = extract32(insn, 0, 16);

    I16 = (int16_t)K16;

    N26 = sextract32(insn, 0, 26);

    K5_11 = (extract32(insn, 21, 5) << 11) | extract32(insn, 0, 11);

    I5_11 = (int16_t)K5_11;



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

        LOG_DIS("l.maci r%d, %d\n", ra, I16);

        t0 = tcg_const_tl(I16);

        gen_mac(dc, cpu_R[ra], t0);

        tcg_temp_free(t0);

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



    case 0x1b: /* l.lwa */

        LOG_DIS("l.lwa r%d, r%d, %d\n", rd, ra, I16);

        gen_lwa(dc, cpu_R[rd], cpu_R[ra], I16);

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

            tcg_gen_addi_tl(t0, cpu_R[ra], I16);

            tcg_gen_qemu_ld_tl(cpu_R[rd], t0, dc->mem_idx, mop);

            tcg_temp_free(t0);

        }

        break;



    case 0x27:    /* l.addi */

        LOG_DIS("l.addi r%d, r%d, %d\n", rd, ra, I16);

        t0 = tcg_const_tl(I16);

        gen_add(dc, cpu_R[rd], cpu_R[ra], t0);

        tcg_temp_free(t0);

        break;



    case 0x28:    /* l.addic */

        LOG_DIS("l.addic r%d, r%d, %d\n", rd, ra, I16);

        t0 = tcg_const_tl(I16);

        gen_addc(dc, cpu_R[rd], cpu_R[ra], t0);

        tcg_temp_free(t0);

        break;



    case 0x29:    /* l.andi */

        LOG_DIS("l.andi r%d, r%d, %d\n", rd, ra, K16);

        tcg_gen_andi_tl(cpu_R[rd], cpu_R[ra], K16);

        break;



    case 0x2a:    /* l.ori */

        LOG_DIS("l.ori r%d, r%d, %d\n", rd, ra, K16);

        tcg_gen_ori_tl(cpu_R[rd], cpu_R[ra], K16);

        break;



    case 0x2b:    /* l.xori */

        LOG_DIS("l.xori r%d, r%d, %d\n", rd, ra, I16);

        tcg_gen_xori_tl(cpu_R[rd], cpu_R[ra], I16);

        break;



    case 0x2c:    /* l.muli */

        LOG_DIS("l.muli r%d, r%d, %d\n", rd, ra, I16);

        t0 = tcg_const_tl(I16);

        gen_mul(dc, cpu_R[rd], cpu_R[ra], t0);

        tcg_temp_free(t0);

        break;



    case 0x2d:    /* l.mfspr */

        LOG_DIS("l.mfspr r%d, r%d, %d\n", rd, ra, K16);

        {

#if defined(CONFIG_USER_ONLY)

            return;

#else

            TCGv_i32 ti = tcg_const_i32(K16);

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

        LOG_DIS("l.mtspr r%d, r%d, %d\n", ra, rb, K5_11);

        {

#if defined(CONFIG_USER_ONLY)

            return;

#else

            TCGv_i32 im = tcg_const_i32(K5_11);

            if (dc->mem_idx == MMU_USER_IDX) {

                gen_illegal_exception(dc);

                return;

            }

            gen_helper_mtspr(cpu_env, cpu_R[ra], cpu_R[rb], im);

            tcg_temp_free_i32(im);

#endif

        }

        break;



    case 0x33: /* l.swa */

        LOG_DIS("l.swa r%d, r%d, %d\n", ra, rb, I5_11);

        gen_swa(dc, cpu_R[rb], cpu_R[ra], I5_11);

        break;



/* not used yet, open it when we need or64.  */

/*#ifdef TARGET_OPENRISC64

    case 0x34:     l.sd

        LOG_DIS("l.sd r%d, r%d, %d\n", ra, rb, I5_11);

        check_ob64s(dc);

        mop = MO_TEQ;

        goto do_store;

#endif*/



    case 0x35:    /* l.sw */

        LOG_DIS("l.sw r%d, r%d, %d\n", ra, rb, I5_11);

        mop = MO_TEUL;

        goto do_store;



    case 0x36:    /* l.sb */

        LOG_DIS("l.sb r%d, r%d, %d\n", ra, rb, I5_11);

        mop = MO_UB;

        goto do_store;



    case 0x37:    /* l.sh */

        LOG_DIS("l.sh r%d, r%d, %d\n", ra, rb, I5_11);

        mop = MO_TEUW;

        goto do_store;



    do_store:

        {

            TCGv t0 = tcg_temp_new();

            tcg_gen_addi_tl(t0, cpu_R[ra], I5_11);

            tcg_gen_qemu_st_tl(cpu_R[rb], t0, dc->mem_idx, mop);

            tcg_temp_free(t0);

        }

        break;



    default:

        gen_illegal_exception(dc);

        break;

    }

}
