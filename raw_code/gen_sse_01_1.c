static void gen_sse(CPUX86State *env, DisasContext *s, int b,

                    target_ulong pc_start, int rex_r)

{

    int b1, op1_offset, op2_offset, is_xmm, val;

    int modrm, mod, rm, reg;

    SSEFunc_0_epp sse_fn_epp;

    SSEFunc_0_eppi sse_fn_eppi;

    SSEFunc_0_ppi sse_fn_ppi;

    SSEFunc_0_eppt sse_fn_eppt;

    TCGMemOp ot;



    b &= 0xff;

    if (s->prefix & PREFIX_DATA)

        b1 = 1;

    else if (s->prefix & PREFIX_REPZ)

        b1 = 2;

    else if (s->prefix & PREFIX_REPNZ)

        b1 = 3;

    else

        b1 = 0;

    sse_fn_epp = sse_op_table1[b][b1];

    if (!sse_fn_epp) {

        goto illegal_op;

    }

    if ((b <= 0x5f && b >= 0x10) || b == 0xc6 || b == 0xc2) {

        is_xmm = 1;

    } else {

        if (b1 == 0) {

            /* MMX case */

            is_xmm = 0;

        } else {

            is_xmm = 1;

        }

    }

    /* simple MMX/SSE operation */

    if (s->flags & HF_TS_MASK) {

        gen_exception(s, EXCP07_PREX, pc_start - s->cs_base);

        return;

    }

    if (s->flags & HF_EM_MASK) {

    illegal_op:

        gen_exception(s, EXCP06_ILLOP, pc_start - s->cs_base);

        return;

    }

    if (is_xmm && !(s->flags & HF_OSFXSR_MASK))

        if ((b != 0x38 && b != 0x3a) || (s->prefix & PREFIX_DATA))

            goto illegal_op;

    if (b == 0x0e) {

        if (!(s->cpuid_ext2_features & CPUID_EXT2_3DNOW))

            goto illegal_op;

        /* femms */

        gen_helper_emms(cpu_env);

        return;

    }

    if (b == 0x77) {

        /* emms */

        gen_helper_emms(cpu_env);

        return;

    }

    /* prepare MMX state (XXX: optimize by storing fptt and fptags in

       the static cpu state) */

    if (!is_xmm) {

        gen_helper_enter_mmx(cpu_env);

    }



    modrm = cpu_ldub_code(env, s->pc++);

    reg = ((modrm >> 3) & 7);

    if (is_xmm)

        reg |= rex_r;

    mod = (modrm >> 6) & 3;

    if (sse_fn_epp == SSE_SPECIAL) {

        b |= (b1 << 8);

        switch(b) {

        case 0x0e7: /* movntq */

            if (mod == 3)

                goto illegal_op;

            gen_lea_modrm(env, s, modrm);

            gen_stq_env_A0(s, offsetof(CPUX86State, fpregs[reg].mmx));

            break;

        case 0x1e7: /* movntdq */

        case 0x02b: /* movntps */

        case 0x12b: /* movntps */

            if (mod == 3)

                goto illegal_op;

            gen_lea_modrm(env, s, modrm);

            gen_sto_env_A0(s, offsetof(CPUX86State, xmm_regs[reg]));

            break;

        case 0x3f0: /* lddqu */

            if (mod == 3)

                goto illegal_op;

            gen_lea_modrm(env, s, modrm);

            gen_ldo_env_A0(s, offsetof(CPUX86State, xmm_regs[reg]));

            break;

        case 0x22b: /* movntss */

        case 0x32b: /* movntsd */

            if (mod == 3)

                goto illegal_op;

            gen_lea_modrm(env, s, modrm);

            if (b1 & 1) {

                gen_stq_env_A0(s, offsetof(CPUX86State,

                                           xmm_regs[reg].XMM_Q(0)));

            } else {

                tcg_gen_ld32u_tl(cpu_T[0], cpu_env, offsetof(CPUX86State,

                    xmm_regs[reg].XMM_L(0)));

                gen_op_st_v(s, MO_32, cpu_T[0], cpu_A0);

            }

            break;

        case 0x6e: /* movd mm, ea */

#ifdef TARGET_X86_64

            if (s->dflag == MO_64) {

                gen_ldst_modrm(env, s, modrm, MO_64, OR_TMP0, 0);

                tcg_gen_st_tl(cpu_T[0], cpu_env, offsetof(CPUX86State,fpregs[reg].mmx));

            } else

#endif

            {

                gen_ldst_modrm(env, s, modrm, MO_32, OR_TMP0, 0);

                tcg_gen_addi_ptr(cpu_ptr0, cpu_env, 

                                 offsetof(CPUX86State,fpregs[reg].mmx));

                tcg_gen_trunc_tl_i32(cpu_tmp2_i32, cpu_T[0]);

                gen_helper_movl_mm_T0_mmx(cpu_ptr0, cpu_tmp2_i32);

            }

            break;

        case 0x16e: /* movd xmm, ea */

#ifdef TARGET_X86_64

            if (s->dflag == MO_64) {

                gen_ldst_modrm(env, s, modrm, MO_64, OR_TMP0, 0);

                tcg_gen_addi_ptr(cpu_ptr0, cpu_env, 

                                 offsetof(CPUX86State,xmm_regs[reg]));

                gen_helper_movq_mm_T0_xmm(cpu_ptr0, cpu_T[0]);

            } else

#endif

            {

                gen_ldst_modrm(env, s, modrm, MO_32, OR_TMP0, 0);

                tcg_gen_addi_ptr(cpu_ptr0, cpu_env, 

                                 offsetof(CPUX86State,xmm_regs[reg]));

                tcg_gen_trunc_tl_i32(cpu_tmp2_i32, cpu_T[0]);

                gen_helper_movl_mm_T0_xmm(cpu_ptr0, cpu_tmp2_i32);

            }

            break;

        case 0x6f: /* movq mm, ea */

            if (mod != 3) {

                gen_lea_modrm(env, s, modrm);

                gen_ldq_env_A0(s, offsetof(CPUX86State, fpregs[reg].mmx));

            } else {

                rm = (modrm & 7);

                tcg_gen_ld_i64(cpu_tmp1_i64, cpu_env,

                               offsetof(CPUX86State,fpregs[rm].mmx));

                tcg_gen_st_i64(cpu_tmp1_i64, cpu_env,

                               offsetof(CPUX86State,fpregs[reg].mmx));

            }

            break;

        case 0x010: /* movups */

        case 0x110: /* movupd */

        case 0x028: /* movaps */

        case 0x128: /* movapd */

        case 0x16f: /* movdqa xmm, ea */

        case 0x26f: /* movdqu xmm, ea */

            if (mod != 3) {

                gen_lea_modrm(env, s, modrm);

                gen_ldo_env_A0(s, offsetof(CPUX86State, xmm_regs[reg]));

            } else {

                rm = (modrm & 7) | REX_B(s);

                gen_op_movo(offsetof(CPUX86State,xmm_regs[reg]),

                            offsetof(CPUX86State,xmm_regs[rm]));

            }

            break;

        case 0x210: /* movss xmm, ea */

            if (mod != 3) {

                gen_lea_modrm(env, s, modrm);

                gen_op_ld_v(s, MO_32, cpu_T[0], cpu_A0);

                tcg_gen_st32_tl(cpu_T[0], cpu_env, offsetof(CPUX86State,xmm_regs[reg].XMM_L(0)));

                tcg_gen_movi_tl(cpu_T[0], 0);

                tcg_gen_st32_tl(cpu_T[0], cpu_env, offsetof(CPUX86State,xmm_regs[reg].XMM_L(1)));

                tcg_gen_st32_tl(cpu_T[0], cpu_env, offsetof(CPUX86State,xmm_regs[reg].XMM_L(2)));

                tcg_gen_st32_tl(cpu_T[0], cpu_env, offsetof(CPUX86State,xmm_regs[reg].XMM_L(3)));

            } else {

                rm = (modrm & 7) | REX_B(s);

                gen_op_movl(offsetof(CPUX86State,xmm_regs[reg].XMM_L(0)),

                            offsetof(CPUX86State,xmm_regs[rm].XMM_L(0)));

            }

            break;

        case 0x310: /* movsd xmm, ea */

            if (mod != 3) {

                gen_lea_modrm(env, s, modrm);

                gen_ldq_env_A0(s, offsetof(CPUX86State,

                                           xmm_regs[reg].XMM_Q(0)));

                tcg_gen_movi_tl(cpu_T[0], 0);

                tcg_gen_st32_tl(cpu_T[0], cpu_env, offsetof(CPUX86State,xmm_regs[reg].XMM_L(2)));

                tcg_gen_st32_tl(cpu_T[0], cpu_env, offsetof(CPUX86State,xmm_regs[reg].XMM_L(3)));

            } else {

                rm = (modrm & 7) | REX_B(s);

                gen_op_movq(offsetof(CPUX86State,xmm_regs[reg].XMM_Q(0)),

                            offsetof(CPUX86State,xmm_regs[rm].XMM_Q(0)));

            }

            break;

        case 0x012: /* movlps */

        case 0x112: /* movlpd */

            if (mod != 3) {

                gen_lea_modrm(env, s, modrm);

                gen_ldq_env_A0(s, offsetof(CPUX86State,

                                           xmm_regs[reg].XMM_Q(0)));

            } else {

                /* movhlps */

                rm = (modrm & 7) | REX_B(s);

                gen_op_movq(offsetof(CPUX86State,xmm_regs[reg].XMM_Q(0)),

                            offsetof(CPUX86State,xmm_regs[rm].XMM_Q(1)));

            }

            break;

        case 0x212: /* movsldup */

            if (mod != 3) {

                gen_lea_modrm(env, s, modrm);

                gen_ldo_env_A0(s, offsetof(CPUX86State, xmm_regs[reg]));

            } else {

                rm = (modrm & 7) | REX_B(s);

                gen_op_movl(offsetof(CPUX86State,xmm_regs[reg].XMM_L(0)),

                            offsetof(CPUX86State,xmm_regs[rm].XMM_L(0)));

                gen_op_movl(offsetof(CPUX86State,xmm_regs[reg].XMM_L(2)),

                            offsetof(CPUX86State,xmm_regs[rm].XMM_L(2)));

            }

            gen_op_movl(offsetof(CPUX86State,xmm_regs[reg].XMM_L(1)),

                        offsetof(CPUX86State,xmm_regs[reg].XMM_L(0)));

            gen_op_movl(offsetof(CPUX86State,xmm_regs[reg].XMM_L(3)),

                        offsetof(CPUX86State,xmm_regs[reg].XMM_L(2)));

            break;

        case 0x312: /* movddup */

            if (mod != 3) {

                gen_lea_modrm(env, s, modrm);

                gen_ldq_env_A0(s, offsetof(CPUX86State,

                                           xmm_regs[reg].XMM_Q(0)));

            } else {

                rm = (modrm & 7) | REX_B(s);

                gen_op_movq(offsetof(CPUX86State,xmm_regs[reg].XMM_Q(0)),

                            offsetof(CPUX86State,xmm_regs[rm].XMM_Q(0)));

            }

            gen_op_movq(offsetof(CPUX86State,xmm_regs[reg].XMM_Q(1)),

                        offsetof(CPUX86State,xmm_regs[reg].XMM_Q(0)));

            break;

        case 0x016: /* movhps */

        case 0x116: /* movhpd */

            if (mod != 3) {

                gen_lea_modrm(env, s, modrm);

                gen_ldq_env_A0(s, offsetof(CPUX86State,

                                           xmm_regs[reg].XMM_Q(1)));

            } else {

                /* movlhps */

                rm = (modrm & 7) | REX_B(s);

                gen_op_movq(offsetof(CPUX86State,xmm_regs[reg].XMM_Q(1)),

                            offsetof(CPUX86State,xmm_regs[rm].XMM_Q(0)));

            }

            break;

        case 0x216: /* movshdup */

            if (mod != 3) {

                gen_lea_modrm(env, s, modrm);

                gen_ldo_env_A0(s, offsetof(CPUX86State, xmm_regs[reg]));

            } else {

                rm = (modrm & 7) | REX_B(s);

                gen_op_movl(offsetof(CPUX86State,xmm_regs[reg].XMM_L(1)),

                            offsetof(CPUX86State,xmm_regs[rm].XMM_L(1)));

                gen_op_movl(offsetof(CPUX86State,xmm_regs[reg].XMM_L(3)),

                            offsetof(CPUX86State,xmm_regs[rm].XMM_L(3)));

            }

            gen_op_movl(offsetof(CPUX86State,xmm_regs[reg].XMM_L(0)),

                        offsetof(CPUX86State,xmm_regs[reg].XMM_L(1)));

            gen_op_movl(offsetof(CPUX86State,xmm_regs[reg].XMM_L(2)),

                        offsetof(CPUX86State,xmm_regs[reg].XMM_L(3)));

            break;

        case 0x178:

        case 0x378:

            {

                int bit_index, field_length;



                if (b1 == 1 && reg != 0)

                    goto illegal_op;

                field_length = cpu_ldub_code(env, s->pc++) & 0x3F;

                bit_index = cpu_ldub_code(env, s->pc++) & 0x3F;

                tcg_gen_addi_ptr(cpu_ptr0, cpu_env,

                    offsetof(CPUX86State,xmm_regs[reg]));

                if (b1 == 1)

                    gen_helper_extrq_i(cpu_env, cpu_ptr0,

                                       tcg_const_i32(bit_index),

                                       tcg_const_i32(field_length));

                else

                    gen_helper_insertq_i(cpu_env, cpu_ptr0,

                                         tcg_const_i32(bit_index),

                                         tcg_const_i32(field_length));

            }

            break;

        case 0x7e: /* movd ea, mm */

#ifdef TARGET_X86_64

            if (s->dflag == MO_64) {

                tcg_gen_ld_i64(cpu_T[0], cpu_env, 

                               offsetof(CPUX86State,fpregs[reg].mmx));

                gen_ldst_modrm(env, s, modrm, MO_64, OR_TMP0, 1);

            } else

#endif

            {

                tcg_gen_ld32u_tl(cpu_T[0], cpu_env, 

                                 offsetof(CPUX86State,fpregs[reg].mmx.MMX_L(0)));

                gen_ldst_modrm(env, s, modrm, MO_32, OR_TMP0, 1);

            }

            break;

        case 0x17e: /* movd ea, xmm */

#ifdef TARGET_X86_64

            if (s->dflag == MO_64) {

                tcg_gen_ld_i64(cpu_T[0], cpu_env, 

                               offsetof(CPUX86State,xmm_regs[reg].XMM_Q(0)));

                gen_ldst_modrm(env, s, modrm, MO_64, OR_TMP0, 1);

            } else

#endif

            {

                tcg_gen_ld32u_tl(cpu_T[0], cpu_env, 

                                 offsetof(CPUX86State,xmm_regs[reg].XMM_L(0)));

                gen_ldst_modrm(env, s, modrm, MO_32, OR_TMP0, 1);

            }

            break;

        case 0x27e: /* movq xmm, ea */

            if (mod != 3) {

                gen_lea_modrm(env, s, modrm);

                gen_ldq_env_A0(s, offsetof(CPUX86State,

                                           xmm_regs[reg].XMM_Q(0)));

            } else {

                rm = (modrm & 7) | REX_B(s);

                gen_op_movq(offsetof(CPUX86State,xmm_regs[reg].XMM_Q(0)),

                            offsetof(CPUX86State,xmm_regs[rm].XMM_Q(0)));

            }

            gen_op_movq_env_0(offsetof(CPUX86State,xmm_regs[reg].XMM_Q(1)));

            break;

        case 0x7f: /* movq ea, mm */

            if (mod != 3) {

                gen_lea_modrm(env, s, modrm);

                gen_stq_env_A0(s, offsetof(CPUX86State, fpregs[reg].mmx));

            } else {

                rm = (modrm & 7);

                gen_op_movq(offsetof(CPUX86State,fpregs[rm].mmx),

                            offsetof(CPUX86State,fpregs[reg].mmx));

            }

            break;

        case 0x011: /* movups */

        case 0x111: /* movupd */

        case 0x029: /* movaps */

        case 0x129: /* movapd */

        case 0x17f: /* movdqa ea, xmm */

        case 0x27f: /* movdqu ea, xmm */

            if (mod != 3) {

                gen_lea_modrm(env, s, modrm);

                gen_sto_env_A0(s, offsetof(CPUX86State, xmm_regs[reg]));

            } else {

                rm = (modrm & 7) | REX_B(s);

                gen_op_movo(offsetof(CPUX86State,xmm_regs[rm]),

                            offsetof(CPUX86State,xmm_regs[reg]));

            }

            break;

        case 0x211: /* movss ea, xmm */

            if (mod != 3) {

                gen_lea_modrm(env, s, modrm);

                tcg_gen_ld32u_tl(cpu_T[0], cpu_env, offsetof(CPUX86State,xmm_regs[reg].XMM_L(0)));

                gen_op_st_v(s, MO_32, cpu_T[0], cpu_A0);

            } else {

                rm = (modrm & 7) | REX_B(s);

                gen_op_movl(offsetof(CPUX86State,xmm_regs[rm].XMM_L(0)),

                            offsetof(CPUX86State,xmm_regs[reg].XMM_L(0)));

            }

            break;

        case 0x311: /* movsd ea, xmm */

            if (mod != 3) {

                gen_lea_modrm(env, s, modrm);

                gen_stq_env_A0(s, offsetof(CPUX86State,

                                           xmm_regs[reg].XMM_Q(0)));

            } else {

                rm = (modrm & 7) | REX_B(s);

                gen_op_movq(offsetof(CPUX86State,xmm_regs[rm].XMM_Q(0)),

                            offsetof(CPUX86State,xmm_regs[reg].XMM_Q(0)));

            }

            break;

        case 0x013: /* movlps */

        case 0x113: /* movlpd */

            if (mod != 3) {

                gen_lea_modrm(env, s, modrm);

                gen_stq_env_A0(s, offsetof(CPUX86State,

                                           xmm_regs[reg].XMM_Q(0)));

            } else {

                goto illegal_op;

            }

            break;

        case 0x017: /* movhps */

        case 0x117: /* movhpd */

            if (mod != 3) {

                gen_lea_modrm(env, s, modrm);

                gen_stq_env_A0(s, offsetof(CPUX86State,

                                           xmm_regs[reg].XMM_Q(1)));

            } else {

                goto illegal_op;

            }

            break;

        case 0x71: /* shift mm, im */

        case 0x72:

        case 0x73:

        case 0x171: /* shift xmm, im */

        case 0x172:

        case 0x173:

            if (b1 >= 2) {

	        goto illegal_op;

            }

            val = cpu_ldub_code(env, s->pc++);

            if (is_xmm) {

                tcg_gen_movi_tl(cpu_T[0], val);

                tcg_gen_st32_tl(cpu_T[0], cpu_env, offsetof(CPUX86State,xmm_t0.XMM_L(0)));

                tcg_gen_movi_tl(cpu_T[0], 0);

                tcg_gen_st32_tl(cpu_T[0], cpu_env, offsetof(CPUX86State,xmm_t0.XMM_L(1)));

                op1_offset = offsetof(CPUX86State,xmm_t0);

            } else {

                tcg_gen_movi_tl(cpu_T[0], val);

                tcg_gen_st32_tl(cpu_T[0], cpu_env, offsetof(CPUX86State,mmx_t0.MMX_L(0)));

                tcg_gen_movi_tl(cpu_T[0], 0);

                tcg_gen_st32_tl(cpu_T[0], cpu_env, offsetof(CPUX86State,mmx_t0.MMX_L(1)));

                op1_offset = offsetof(CPUX86State,mmx_t0);

            }

            sse_fn_epp = sse_op_table2[((b - 1) & 3) * 8 +

                                       (((modrm >> 3)) & 7)][b1];

            if (!sse_fn_epp) {

                goto illegal_op;

            }

            if (is_xmm) {

                rm = (modrm & 7) | REX_B(s);

                op2_offset = offsetof(CPUX86State,xmm_regs[rm]);

            } else {

                rm = (modrm & 7);

                op2_offset = offsetof(CPUX86State,fpregs[rm].mmx);

            }

            tcg_gen_addi_ptr(cpu_ptr0, cpu_env, op2_offset);

            tcg_gen_addi_ptr(cpu_ptr1, cpu_env, op1_offset);

            sse_fn_epp(cpu_env, cpu_ptr0, cpu_ptr1);

            break;

        case 0x050: /* movmskps */

            rm = (modrm & 7) | REX_B(s);

            tcg_gen_addi_ptr(cpu_ptr0, cpu_env, 

                             offsetof(CPUX86State,xmm_regs[rm]));

            gen_helper_movmskps(cpu_tmp2_i32, cpu_env, cpu_ptr0);

            tcg_gen_extu_i32_tl(cpu_regs[reg], cpu_tmp2_i32);

            break;

        case 0x150: /* movmskpd */

            rm = (modrm & 7) | REX_B(s);

            tcg_gen_addi_ptr(cpu_ptr0, cpu_env, 

                             offsetof(CPUX86State,xmm_regs[rm]));

            gen_helper_movmskpd(cpu_tmp2_i32, cpu_env, cpu_ptr0);

            tcg_gen_extu_i32_tl(cpu_regs[reg], cpu_tmp2_i32);

            break;

        case 0x02a: /* cvtpi2ps */

        case 0x12a: /* cvtpi2pd */

            gen_helper_enter_mmx(cpu_env);

            if (mod != 3) {

                gen_lea_modrm(env, s, modrm);

                op2_offset = offsetof(CPUX86State,mmx_t0);

                gen_ldq_env_A0(s, op2_offset);

            } else {

                rm = (modrm & 7);

                op2_offset = offsetof(CPUX86State,fpregs[rm].mmx);

            }

            op1_offset = offsetof(CPUX86State,xmm_regs[reg]);

            tcg_gen_addi_ptr(cpu_ptr0, cpu_env, op1_offset);

            tcg_gen_addi_ptr(cpu_ptr1, cpu_env, op2_offset);

            switch(b >> 8) {

            case 0x0:

                gen_helper_cvtpi2ps(cpu_env, cpu_ptr0, cpu_ptr1);

                break;

            default:

            case 0x1:

                gen_helper_cvtpi2pd(cpu_env, cpu_ptr0, cpu_ptr1);

                break;

            }

            break;

        case 0x22a: /* cvtsi2ss */

        case 0x32a: /* cvtsi2sd */

            ot = mo_64_32(s->dflag);

            gen_ldst_modrm(env, s, modrm, ot, OR_TMP0, 0);

            op1_offset = offsetof(CPUX86State,xmm_regs[reg]);

            tcg_gen_addi_ptr(cpu_ptr0, cpu_env, op1_offset);

            if (ot == MO_32) {

                SSEFunc_0_epi sse_fn_epi = sse_op_table3ai[(b >> 8) & 1];

                tcg_gen_trunc_tl_i32(cpu_tmp2_i32, cpu_T[0]);

                sse_fn_epi(cpu_env, cpu_ptr0, cpu_tmp2_i32);

            } else {

#ifdef TARGET_X86_64

                SSEFunc_0_epl sse_fn_epl = sse_op_table3aq[(b >> 8) & 1];

                sse_fn_epl(cpu_env, cpu_ptr0, cpu_T[0]);

#else

                goto illegal_op;

#endif

            }

            break;

        case 0x02c: /* cvttps2pi */

        case 0x12c: /* cvttpd2pi */

        case 0x02d: /* cvtps2pi */

        case 0x12d: /* cvtpd2pi */

            gen_helper_enter_mmx(cpu_env);

            if (mod != 3) {

                gen_lea_modrm(env, s, modrm);

                op2_offset = offsetof(CPUX86State,xmm_t0);

                gen_ldo_env_A0(s, op2_offset);

            } else {

                rm = (modrm & 7) | REX_B(s);

                op2_offset = offsetof(CPUX86State,xmm_regs[rm]);

            }

            op1_offset = offsetof(CPUX86State,fpregs[reg & 7].mmx);

            tcg_gen_addi_ptr(cpu_ptr0, cpu_env, op1_offset);

            tcg_gen_addi_ptr(cpu_ptr1, cpu_env, op2_offset);

            switch(b) {

            case 0x02c:

                gen_helper_cvttps2pi(cpu_env, cpu_ptr0, cpu_ptr1);

                break;

            case 0x12c:

                gen_helper_cvttpd2pi(cpu_env, cpu_ptr0, cpu_ptr1);

                break;

            case 0x02d:

                gen_helper_cvtps2pi(cpu_env, cpu_ptr0, cpu_ptr1);

                break;

            case 0x12d:

                gen_helper_cvtpd2pi(cpu_env, cpu_ptr0, cpu_ptr1);

                break;

            }

            break;

        case 0x22c: /* cvttss2si */

        case 0x32c: /* cvttsd2si */

        case 0x22d: /* cvtss2si */

        case 0x32d: /* cvtsd2si */

            ot = mo_64_32(s->dflag);

            if (mod != 3) {

                gen_lea_modrm(env, s, modrm);

                if ((b >> 8) & 1) {

                    gen_ldq_env_A0(s, offsetof(CPUX86State, xmm_t0.XMM_Q(0)));

                } else {

                    gen_op_ld_v(s, MO_32, cpu_T[0], cpu_A0);

                    tcg_gen_st32_tl(cpu_T[0], cpu_env, offsetof(CPUX86State,xmm_t0.XMM_L(0)));

                }

                op2_offset = offsetof(CPUX86State,xmm_t0);

            } else {

                rm = (modrm & 7) | REX_B(s);

                op2_offset = offsetof(CPUX86State,xmm_regs[rm]);

            }

            tcg_gen_addi_ptr(cpu_ptr0, cpu_env, op2_offset);

            if (ot == MO_32) {

                SSEFunc_i_ep sse_fn_i_ep =

                    sse_op_table3bi[((b >> 7) & 2) | (b & 1)];

                sse_fn_i_ep(cpu_tmp2_i32, cpu_env, cpu_ptr0);

                tcg_gen_extu_i32_tl(cpu_T[0], cpu_tmp2_i32);

            } else {

#ifdef TARGET_X86_64

                SSEFunc_l_ep sse_fn_l_ep =

                    sse_op_table3bq[((b >> 7) & 2) | (b & 1)];

                sse_fn_l_ep(cpu_T[0], cpu_env, cpu_ptr0);

#else

                goto illegal_op;

#endif

            }

            gen_op_mov_reg_v(ot, reg, cpu_T[0]);

            break;

        case 0xc4: /* pinsrw */

        case 0x1c4:

            s->rip_offset = 1;

            gen_ldst_modrm(env, s, modrm, MO_16, OR_TMP0, 0);

            val = cpu_ldub_code(env, s->pc++);

            if (b1) {

                val &= 7;

                tcg_gen_st16_tl(cpu_T[0], cpu_env,

                                offsetof(CPUX86State,xmm_regs[reg].XMM_W(val)));

            } else {

                val &= 3;

                tcg_gen_st16_tl(cpu_T[0], cpu_env,

                                offsetof(CPUX86State,fpregs[reg].mmx.MMX_W(val)));

            }

            break;

        case 0xc5: /* pextrw */

        case 0x1c5:

            if (mod != 3)

                goto illegal_op;

            ot = mo_64_32(s->dflag);

            val = cpu_ldub_code(env, s->pc++);

            if (b1) {

                val &= 7;

                rm = (modrm & 7) | REX_B(s);

                tcg_gen_ld16u_tl(cpu_T[0], cpu_env,

                                 offsetof(CPUX86State,xmm_regs[rm].XMM_W(val)));

            } else {

                val &= 3;

                rm = (modrm & 7);

                tcg_gen_ld16u_tl(cpu_T[0], cpu_env,

                                offsetof(CPUX86State,fpregs[rm].mmx.MMX_W(val)));

            }

            reg = ((modrm >> 3) & 7) | rex_r;

            gen_op_mov_reg_v(ot, reg, cpu_T[0]);

            break;

        case 0x1d6: /* movq ea, xmm */

            if (mod != 3) {

                gen_lea_modrm(env, s, modrm);

                gen_stq_env_A0(s, offsetof(CPUX86State,

                                           xmm_regs[reg].XMM_Q(0)));

            } else {

                rm = (modrm & 7) | REX_B(s);

                gen_op_movq(offsetof(CPUX86State,xmm_regs[rm].XMM_Q(0)),

                            offsetof(CPUX86State,xmm_regs[reg].XMM_Q(0)));

                gen_op_movq_env_0(offsetof(CPUX86State,xmm_regs[rm].XMM_Q(1)));

            }

            break;

        case 0x2d6: /* movq2dq */

            gen_helper_enter_mmx(cpu_env);

            rm = (modrm & 7);

            gen_op_movq(offsetof(CPUX86State,xmm_regs[reg].XMM_Q(0)),

                        offsetof(CPUX86State,fpregs[rm].mmx));

            gen_op_movq_env_0(offsetof(CPUX86State,xmm_regs[reg].XMM_Q(1)));

            break;

        case 0x3d6: /* movdq2q */

            gen_helper_enter_mmx(cpu_env);

            rm = (modrm & 7) | REX_B(s);

            gen_op_movq(offsetof(CPUX86State,fpregs[reg & 7].mmx),

                        offsetof(CPUX86State,xmm_regs[rm].XMM_Q(0)));

            break;

        case 0xd7: /* pmovmskb */

        case 0x1d7:

            if (mod != 3)

                goto illegal_op;

            if (b1) {

                rm = (modrm & 7) | REX_B(s);

                tcg_gen_addi_ptr(cpu_ptr0, cpu_env, offsetof(CPUX86State,xmm_regs[rm]));

                gen_helper_pmovmskb_xmm(cpu_tmp2_i32, cpu_env, cpu_ptr0);

            } else {

                rm = (modrm & 7);

                tcg_gen_addi_ptr(cpu_ptr0, cpu_env, offsetof(CPUX86State,fpregs[rm].mmx));

                gen_helper_pmovmskb_mmx(cpu_tmp2_i32, cpu_env, cpu_ptr0);

            }

            reg = ((modrm >> 3) & 7) | rex_r;

            tcg_gen_extu_i32_tl(cpu_regs[reg], cpu_tmp2_i32);

            break;



        case 0x138:

        case 0x038:

            b = modrm;

            if ((b & 0xf0) == 0xf0) {

                goto do_0f_38_fx;

            }

            modrm = cpu_ldub_code(env, s->pc++);

            rm = modrm & 7;

            reg = ((modrm >> 3) & 7) | rex_r;

            mod = (modrm >> 6) & 3;

            if (b1 >= 2) {

                goto illegal_op;

            }



            sse_fn_epp = sse_op_table6[b].op[b1];

            if (!sse_fn_epp) {

                goto illegal_op;

            }

            if (!(s->cpuid_ext_features & sse_op_table6[b].ext_mask))

                goto illegal_op;



            if (b1) {

                op1_offset = offsetof(CPUX86State,xmm_regs[reg]);

                if (mod == 3) {

                    op2_offset = offsetof(CPUX86State,xmm_regs[rm | REX_B(s)]);

                } else {

                    op2_offset = offsetof(CPUX86State,xmm_t0);

                    gen_lea_modrm(env, s, modrm);

                    switch (b) {

                    case 0x20: case 0x30: /* pmovsxbw, pmovzxbw */

                    case 0x23: case 0x33: /* pmovsxwd, pmovzxwd */

                    case 0x25: case 0x35: /* pmovsxdq, pmovzxdq */

                        gen_ldq_env_A0(s, op2_offset +

                                        offsetof(XMMReg, XMM_Q(0)));

                        break;

                    case 0x21: case 0x31: /* pmovsxbd, pmovzxbd */

                    case 0x24: case 0x34: /* pmovsxwq, pmovzxwq */

                        tcg_gen_qemu_ld_i32(cpu_tmp2_i32, cpu_A0,

                                            s->mem_index, MO_LEUL);

                        tcg_gen_st_i32(cpu_tmp2_i32, cpu_env, op2_offset +

                                        offsetof(XMMReg, XMM_L(0)));

                        break;

                    case 0x22: case 0x32: /* pmovsxbq, pmovzxbq */

                        tcg_gen_qemu_ld_tl(cpu_tmp0, cpu_A0,

                                           s->mem_index, MO_LEUW);

                        tcg_gen_st16_tl(cpu_tmp0, cpu_env, op2_offset +

                                        offsetof(XMMReg, XMM_W(0)));

                        break;

                    case 0x2a:            /* movntqda */

                        gen_ldo_env_A0(s, op1_offset);

                        return;

                    default:

                        gen_ldo_env_A0(s, op2_offset);

                    }

                }

            } else {

                op1_offset = offsetof(CPUX86State,fpregs[reg].mmx);

                if (mod == 3) {

                    op2_offset = offsetof(CPUX86State,fpregs[rm].mmx);

                } else {

                    op2_offset = offsetof(CPUX86State,mmx_t0);

                    gen_lea_modrm(env, s, modrm);

                    gen_ldq_env_A0(s, op2_offset);

                }

            }

            if (sse_fn_epp == SSE_SPECIAL) {

                goto illegal_op;

            }



            tcg_gen_addi_ptr(cpu_ptr0, cpu_env, op1_offset);

            tcg_gen_addi_ptr(cpu_ptr1, cpu_env, op2_offset);

            sse_fn_epp(cpu_env, cpu_ptr0, cpu_ptr1);



            if (b == 0x17) {

                set_cc_op(s, CC_OP_EFLAGS);

            }

            break;



        case 0x238:

        case 0x338:

        do_0f_38_fx:

            /* Various integer extensions at 0f 38 f[0-f].  */

            b = modrm | (b1 << 8);

            modrm = cpu_ldub_code(env, s->pc++);

            reg = ((modrm >> 3) & 7) | rex_r;



            switch (b) {

            case 0x3f0: /* crc32 Gd,Eb */

            case 0x3f1: /* crc32 Gd,Ey */

            do_crc32:

                if (!(s->cpuid_ext_features & CPUID_EXT_SSE42)) {

                    goto illegal_op;

                }

                if ((b & 0xff) == 0xf0) {

                    ot = MO_8;

                } else if (s->dflag != MO_64) {

                    ot = (s->prefix & PREFIX_DATA ? MO_16 : MO_32);

                } else {

                    ot = MO_64;

                }



                tcg_gen_trunc_tl_i32(cpu_tmp2_i32, cpu_regs[reg]);

                gen_ldst_modrm(env, s, modrm, ot, OR_TMP0, 0);

                gen_helper_crc32(cpu_T[0], cpu_tmp2_i32,

                                 cpu_T[0], tcg_const_i32(8 << ot));



                ot = mo_64_32(s->dflag);

                gen_op_mov_reg_v(ot, reg, cpu_T[0]);

                break;



            case 0x1f0: /* crc32 or movbe */

            case 0x1f1:

                /* For these insns, the f3 prefix is supposed to have priority

                   over the 66 prefix, but that's not what we implement above

                   setting b1.  */

                if (s->prefix & PREFIX_REPNZ) {

                    goto do_crc32;

                }

                /* FALLTHRU */

            case 0x0f0: /* movbe Gy,My */

            case 0x0f1: /* movbe My,Gy */

                if (!(s->cpuid_ext_features & CPUID_EXT_MOVBE)) {

                    goto illegal_op;

                }

                if (s->dflag != MO_64) {

                    ot = (s->prefix & PREFIX_DATA ? MO_16 : MO_32);

                } else {

                    ot = MO_64;

                }



                gen_lea_modrm(env, s, modrm);

                if ((b & 1) == 0) {

                    tcg_gen_qemu_ld_tl(cpu_T[0], cpu_A0,

                                       s->mem_index, ot | MO_BE);

                    gen_op_mov_reg_v(ot, reg, cpu_T[0]);

                } else {

                    tcg_gen_qemu_st_tl(cpu_regs[reg], cpu_A0,

                                       s->mem_index, ot | MO_BE);

                }

                break;



            case 0x0f2: /* andn Gy, By, Ey */

                if (!(s->cpuid_7_0_ebx_features & CPUID_7_0_EBX_BMI1)

                    || !(s->prefix & PREFIX_VEX)

                    || s->vex_l != 0) {

                    goto illegal_op;

                }

                ot = mo_64_32(s->dflag);

                gen_ldst_modrm(env, s, modrm, ot, OR_TMP0, 0);

                tcg_gen_andc_tl(cpu_T[0], cpu_regs[s->vex_v], cpu_T[0]);

                gen_op_mov_reg_v(ot, reg, cpu_T[0]);

                gen_op_update1_cc();

                set_cc_op(s, CC_OP_LOGICB + ot);

                break;



            case 0x0f7: /* bextr Gy, Ey, By */

                if (!(s->cpuid_7_0_ebx_features & CPUID_7_0_EBX_BMI1)

                    || !(s->prefix & PREFIX_VEX)

                    || s->vex_l != 0) {

                    goto illegal_op;

                }

                ot = mo_64_32(s->dflag);

                {

                    TCGv bound, zero;



                    gen_ldst_modrm(env, s, modrm, ot, OR_TMP0, 0);

                    /* Extract START, and shift the operand.

                       Shifts larger than operand size get zeros.  */

                    tcg_gen_ext8u_tl(cpu_A0, cpu_regs[s->vex_v]);

                    tcg_gen_shr_tl(cpu_T[0], cpu_T[0], cpu_A0);



                    bound = tcg_const_tl(ot == MO_64 ? 63 : 31);

                    zero = tcg_const_tl(0);

                    tcg_gen_movcond_tl(TCG_COND_LEU, cpu_T[0], cpu_A0, bound,

                                       cpu_T[0], zero);

                    tcg_temp_free(zero);



                    /* Extract the LEN into a mask.  Lengths larger than

                       operand size get all ones.  */

                    tcg_gen_shri_tl(cpu_A0, cpu_regs[s->vex_v], 8);

                    tcg_gen_ext8u_tl(cpu_A0, cpu_A0);

                    tcg_gen_movcond_tl(TCG_COND_LEU, cpu_A0, cpu_A0, bound,

                                       cpu_A0, bound);

                    tcg_temp_free(bound);

                    tcg_gen_movi_tl(cpu_T[1], 1);

                    tcg_gen_shl_tl(cpu_T[1], cpu_T[1], cpu_A0);

                    tcg_gen_subi_tl(cpu_T[1], cpu_T[1], 1);

                    tcg_gen_and_tl(cpu_T[0], cpu_T[0], cpu_T[1]);



                    gen_op_mov_reg_v(ot, reg, cpu_T[0]);

                    gen_op_update1_cc();

                    set_cc_op(s, CC_OP_LOGICB + ot);

                }

                break;



            case 0x0f5: /* bzhi Gy, Ey, By */

                if (!(s->cpuid_7_0_ebx_features & CPUID_7_0_EBX_BMI2)

                    || !(s->prefix & PREFIX_VEX)

                    || s->vex_l != 0) {

                    goto illegal_op;

                }

                ot = mo_64_32(s->dflag);

                gen_ldst_modrm(env, s, modrm, ot, OR_TMP0, 0);

                tcg_gen_ext8u_tl(cpu_T[1], cpu_regs[s->vex_v]);

                {

                    TCGv bound = tcg_const_tl(ot == MO_64 ? 63 : 31);

                    /* Note that since we're using BMILG (in order to get O

                       cleared) we need to store the inverse into C.  */

                    tcg_gen_setcond_tl(TCG_COND_LT, cpu_cc_src,

                                       cpu_T[1], bound);

                    tcg_gen_movcond_tl(TCG_COND_GT, cpu_T[1], cpu_T[1],

                                       bound, bound, cpu_T[1]);

                    tcg_temp_free(bound);

                }

                tcg_gen_movi_tl(cpu_A0, -1);

                tcg_gen_shl_tl(cpu_A0, cpu_A0, cpu_T[1]);

                tcg_gen_andc_tl(cpu_T[0], cpu_T[0], cpu_A0);

                gen_op_mov_reg_v(ot, reg, cpu_T[0]);

                gen_op_update1_cc();

                set_cc_op(s, CC_OP_BMILGB + ot);

                break;



            case 0x3f6: /* mulx By, Gy, rdx, Ey */

                if (!(s->cpuid_7_0_ebx_features & CPUID_7_0_EBX_BMI2)

                    || !(s->prefix & PREFIX_VEX)

                    || s->vex_l != 0) {

                    goto illegal_op;

                }

                ot = mo_64_32(s->dflag);

                gen_ldst_modrm(env, s, modrm, ot, OR_TMP0, 0);

                switch (ot) {

                default:

                    tcg_gen_trunc_tl_i32(cpu_tmp2_i32, cpu_T[0]);

                    tcg_gen_trunc_tl_i32(cpu_tmp3_i32, cpu_regs[R_EDX]);

                    tcg_gen_mulu2_i32(cpu_tmp2_i32, cpu_tmp3_i32,

                                      cpu_tmp2_i32, cpu_tmp3_i32);

                    tcg_gen_extu_i32_tl(cpu_regs[s->vex_v], cpu_tmp2_i32);

                    tcg_gen_extu_i32_tl(cpu_regs[reg], cpu_tmp3_i32);

                    break;

#ifdef TARGET_X86_64

                case MO_64:

                    tcg_gen_mulu2_i64(cpu_regs[s->vex_v], cpu_regs[reg],

                                      cpu_T[0], cpu_regs[R_EDX]);

                    break;

#endif

                }

                break;



            case 0x3f5: /* pdep Gy, By, Ey */

                if (!(s->cpuid_7_0_ebx_features & CPUID_7_0_EBX_BMI2)

                    || !(s->prefix & PREFIX_VEX)

                    || s->vex_l != 0) {

                    goto illegal_op;

                }

                ot = mo_64_32(s->dflag);

                gen_ldst_modrm(env, s, modrm, ot, OR_TMP0, 0);

                /* Note that by zero-extending the mask operand, we

                   automatically handle zero-extending the result.  */

                if (ot == MO_64) {

                    tcg_gen_mov_tl(cpu_T[1], cpu_regs[s->vex_v]);

                } else {

                    tcg_gen_ext32u_tl(cpu_T[1], cpu_regs[s->vex_v]);

                }

                gen_helper_pdep(cpu_regs[reg], cpu_T[0], cpu_T[1]);

                break;



            case 0x2f5: /* pext Gy, By, Ey */

                if (!(s->cpuid_7_0_ebx_features & CPUID_7_0_EBX_BMI2)

                    || !(s->prefix & PREFIX_VEX)

                    || s->vex_l != 0) {

                    goto illegal_op;

                }

                ot = mo_64_32(s->dflag);

                gen_ldst_modrm(env, s, modrm, ot, OR_TMP0, 0);

                /* Note that by zero-extending the mask operand, we

                   automatically handle zero-extending the result.  */

                if (ot == MO_64) {

                    tcg_gen_mov_tl(cpu_T[1], cpu_regs[s->vex_v]);

                } else {

                    tcg_gen_ext32u_tl(cpu_T[1], cpu_regs[s->vex_v]);

                }

                gen_helper_pext(cpu_regs[reg], cpu_T[0], cpu_T[1]);

                break;



            case 0x1f6: /* adcx Gy, Ey */

            case 0x2f6: /* adox Gy, Ey */

                if (!(s->cpuid_7_0_ebx_features & CPUID_7_0_EBX_ADX)) {

                    goto illegal_op;

                } else {

                    TCGv carry_in, carry_out, zero;

                    int end_op;



                    ot = mo_64_32(s->dflag);

                    gen_ldst_modrm(env, s, modrm, ot, OR_TMP0, 0);



                    /* Re-use the carry-out from a previous round.  */

                    TCGV_UNUSED(carry_in);

                    carry_out = (b == 0x1f6 ? cpu_cc_dst : cpu_cc_src2);

                    switch (s->cc_op) {

                    case CC_OP_ADCX:

                        if (b == 0x1f6) {

                            carry_in = cpu_cc_dst;

                            end_op = CC_OP_ADCX;

                        } else {

                            end_op = CC_OP_ADCOX;

                        }

                        break;

                    case CC_OP_ADOX:

                        if (b == 0x1f6) {

                            end_op = CC_OP_ADCOX;

                        } else {

                            carry_in = cpu_cc_src2;

                            end_op = CC_OP_ADOX;

                        }

                        break;

                    case CC_OP_ADCOX:

                        end_op = CC_OP_ADCOX;

                        carry_in = carry_out;

                        break;

                    default:

                        end_op = (b == 0x1f6 ? CC_OP_ADCX : CC_OP_ADOX);

                        break;

                    }

                    /* If we can't reuse carry-out, get it out of EFLAGS.  */

                    if (TCGV_IS_UNUSED(carry_in)) {

                        if (s->cc_op != CC_OP_ADCX && s->cc_op != CC_OP_ADOX) {

                            gen_compute_eflags(s);

                        }

                        carry_in = cpu_tmp0;

                        tcg_gen_shri_tl(carry_in, cpu_cc_src,

                                        ctz32(b == 0x1f6 ? CC_C : CC_O));

                        tcg_gen_andi_tl(carry_in, carry_in, 1);

                    }



                    switch (ot) {

#ifdef TARGET_X86_64

                    case MO_32:

                        /* If we know TL is 64-bit, and we want a 32-bit

                           result, just do everything in 64-bit arithmetic.  */

                        tcg_gen_ext32u_i64(cpu_regs[reg], cpu_regs[reg]);

                        tcg_gen_ext32u_i64(cpu_T[0], cpu_T[0]);

                        tcg_gen_add_i64(cpu_T[0], cpu_T[0], cpu_regs[reg]);

                        tcg_gen_add_i64(cpu_T[0], cpu_T[0], carry_in);

                        tcg_gen_ext32u_i64(cpu_regs[reg], cpu_T[0]);

                        tcg_gen_shri_i64(carry_out, cpu_T[0], 32);

                        break;

#endif

                    default:

                        /* Otherwise compute the carry-out in two steps.  */

                        zero = tcg_const_tl(0);

                        tcg_gen_add2_tl(cpu_T[0], carry_out,

                                        cpu_T[0], zero,

                                        carry_in, zero);

                        tcg_gen_add2_tl(cpu_regs[reg], carry_out,

                                        cpu_regs[reg], carry_out,

                                        cpu_T[0], zero);

                        tcg_temp_free(zero);

                        break;

                    }

                    set_cc_op(s, end_op);

                }

                break;



            case 0x1f7: /* shlx Gy, Ey, By */

            case 0x2f7: /* sarx Gy, Ey, By */

            case 0x3f7: /* shrx Gy, Ey, By */

                if (!(s->cpuid_7_0_ebx_features & CPUID_7_0_EBX_BMI2)

                    || !(s->prefix & PREFIX_VEX)

                    || s->vex_l != 0) {

                    goto illegal_op;

                }

                ot = mo_64_32(s->dflag);

                gen_ldst_modrm(env, s, modrm, ot, OR_TMP0, 0);

                if (ot == MO_64) {

                    tcg_gen_andi_tl(cpu_T[1], cpu_regs[s->vex_v], 63);

                } else {

                    tcg_gen_andi_tl(cpu_T[1], cpu_regs[s->vex_v], 31);

                }

                if (b == 0x1f7) {

                    tcg_gen_shl_tl(cpu_T[0], cpu_T[0], cpu_T[1]);

                } else if (b == 0x2f7) {

                    if (ot != MO_64) {

                        tcg_gen_ext32s_tl(cpu_T[0], cpu_T[0]);

                    }

                    tcg_gen_sar_tl(cpu_T[0], cpu_T[0], cpu_T[1]);

                } else {

                    if (ot != MO_64) {

                        tcg_gen_ext32u_tl(cpu_T[0], cpu_T[0]);

                    }

                    tcg_gen_shr_tl(cpu_T[0], cpu_T[0], cpu_T[1]);

                }

                gen_op_mov_reg_v(ot, reg, cpu_T[0]);

                break;



            case 0x0f3:

            case 0x1f3:

            case 0x2f3:

            case 0x3f3: /* Group 17 */

                if (!(s->cpuid_7_0_ebx_features & CPUID_7_0_EBX_BMI1)

                    || !(s->prefix & PREFIX_VEX)

                    || s->vex_l != 0) {

                    goto illegal_op;

                }

                ot = mo_64_32(s->dflag);

                gen_ldst_modrm(env, s, modrm, ot, OR_TMP0, 0);



                switch (reg & 7) {

                case 1: /* blsr By,Ey */

                    tcg_gen_neg_tl(cpu_T[1], cpu_T[0]);

                    tcg_gen_and_tl(cpu_T[0], cpu_T[0], cpu_T[1]);

                    gen_op_mov_reg_v(ot, s->vex_v, cpu_T[0]);

                    gen_op_update2_cc();

                    set_cc_op(s, CC_OP_BMILGB + ot);

                    break;



                case 2: /* blsmsk By,Ey */

                    tcg_gen_mov_tl(cpu_cc_src, cpu_T[0]);

                    tcg_gen_subi_tl(cpu_T[0], cpu_T[0], 1);

                    tcg_gen_xor_tl(cpu_T[0], cpu_T[0], cpu_cc_src);

                    tcg_gen_mov_tl(cpu_cc_dst, cpu_T[0]);

                    set_cc_op(s, CC_OP_BMILGB + ot);

                    break;



                case 3: /* blsi By, Ey */

                    tcg_gen_mov_tl(cpu_cc_src, cpu_T[0]);

                    tcg_gen_subi_tl(cpu_T[0], cpu_T[0], 1);

                    tcg_gen_and_tl(cpu_T[0], cpu_T[0], cpu_cc_src);

                    tcg_gen_mov_tl(cpu_cc_dst, cpu_T[0]);

                    set_cc_op(s, CC_OP_BMILGB + ot);

                    break;



                default:

                    goto illegal_op;

                }

                break;



            default:

                goto illegal_op;

            }

            break;



        case 0x03a:

        case 0x13a:

            b = modrm;

            modrm = cpu_ldub_code(env, s->pc++);

            rm = modrm & 7;

            reg = ((modrm >> 3) & 7) | rex_r;

            mod = (modrm >> 6) & 3;

            if (b1 >= 2) {

                goto illegal_op;

            }



            sse_fn_eppi = sse_op_table7[b].op[b1];

            if (!sse_fn_eppi) {

                goto illegal_op;

            }

            if (!(s->cpuid_ext_features & sse_op_table7[b].ext_mask))

                goto illegal_op;



            if (sse_fn_eppi == SSE_SPECIAL) {

                ot = mo_64_32(s->dflag);

                rm = (modrm & 7) | REX_B(s);

                if (mod != 3)

                    gen_lea_modrm(env, s, modrm);

                reg = ((modrm >> 3) & 7) | rex_r;

                val = cpu_ldub_code(env, s->pc++);

                switch (b) {

                case 0x14: /* pextrb */

                    tcg_gen_ld8u_tl(cpu_T[0], cpu_env, offsetof(CPUX86State,

                                            xmm_regs[reg].XMM_B(val & 15)));

                    if (mod == 3) {

                        gen_op_mov_reg_v(ot, rm, cpu_T[0]);

                    } else {

                        tcg_gen_qemu_st_tl(cpu_T[0], cpu_A0,

                                           s->mem_index, MO_UB);

                    }

                    break;

                case 0x15: /* pextrw */

                    tcg_gen_ld16u_tl(cpu_T[0], cpu_env, offsetof(CPUX86State,

                                            xmm_regs[reg].XMM_W(val & 7)));

                    if (mod == 3) {

                        gen_op_mov_reg_v(ot, rm, cpu_T[0]);

                    } else {

                        tcg_gen_qemu_st_tl(cpu_T[0], cpu_A0,

                                           s->mem_index, MO_LEUW);

                    }

                    break;

                case 0x16:

                    if (ot == MO_32) { /* pextrd */

                        tcg_gen_ld_i32(cpu_tmp2_i32, cpu_env,

                                        offsetof(CPUX86State,

                                                xmm_regs[reg].XMM_L(val & 3)));

                        if (mod == 3) {

                            tcg_gen_extu_i32_tl(cpu_regs[rm], cpu_tmp2_i32);

                        } else {

                            tcg_gen_qemu_st_i32(cpu_tmp2_i32, cpu_A0,

                                                s->mem_index, MO_LEUL);

                        }

                    } else { /* pextrq */

#ifdef TARGET_X86_64

                        tcg_gen_ld_i64(cpu_tmp1_i64, cpu_env,

                                        offsetof(CPUX86State,

                                                xmm_regs[reg].XMM_Q(val & 1)));

                        if (mod == 3) {

                            tcg_gen_mov_i64(cpu_regs[rm], cpu_tmp1_i64);

                        } else {

                            tcg_gen_qemu_st_i64(cpu_tmp1_i64, cpu_A0,

                                                s->mem_index, MO_LEQ);

                        }

#else

                        goto illegal_op;

#endif

                    }

                    break;

                case 0x17: /* extractps */

                    tcg_gen_ld32u_tl(cpu_T[0], cpu_env, offsetof(CPUX86State,

                                            xmm_regs[reg].XMM_L(val & 3)));

                    if (mod == 3) {

                        gen_op_mov_reg_v(ot, rm, cpu_T[0]);

                    } else {

                        tcg_gen_qemu_st_tl(cpu_T[0], cpu_A0,

                                           s->mem_index, MO_LEUL);

                    }

                    break;

                case 0x20: /* pinsrb */

                    if (mod == 3) {

                        gen_op_mov_v_reg(MO_32, cpu_T[0], rm);

                    } else {

                        tcg_gen_qemu_ld_tl(cpu_T[0], cpu_A0,

                                           s->mem_index, MO_UB);

                    }

                    tcg_gen_st8_tl(cpu_T[0], cpu_env, offsetof(CPUX86State,

                                            xmm_regs[reg].XMM_B(val & 15)));

                    break;

                case 0x21: /* insertps */

                    if (mod == 3) {

                        tcg_gen_ld_i32(cpu_tmp2_i32, cpu_env,

                                        offsetof(CPUX86State,xmm_regs[rm]

                                                .XMM_L((val >> 6) & 3)));

                    } else {

                        tcg_gen_qemu_ld_i32(cpu_tmp2_i32, cpu_A0,

                                            s->mem_index, MO_LEUL);

                    }

                    tcg_gen_st_i32(cpu_tmp2_i32, cpu_env,

                                    offsetof(CPUX86State,xmm_regs[reg]

                                            .XMM_L((val >> 4) & 3)));

                    if ((val >> 0) & 1)

                        tcg_gen_st_i32(tcg_const_i32(0 /*float32_zero*/),

                                        cpu_env, offsetof(CPUX86State,

                                                xmm_regs[reg].XMM_L(0)));

                    if ((val >> 1) & 1)

                        tcg_gen_st_i32(tcg_const_i32(0 /*float32_zero*/),

                                        cpu_env, offsetof(CPUX86State,

                                                xmm_regs[reg].XMM_L(1)));

                    if ((val >> 2) & 1)

                        tcg_gen_st_i32(tcg_const_i32(0 /*float32_zero*/),

                                        cpu_env, offsetof(CPUX86State,

                                                xmm_regs[reg].XMM_L(2)));

                    if ((val >> 3) & 1)

                        tcg_gen_st_i32(tcg_const_i32(0 /*float32_zero*/),

                                        cpu_env, offsetof(CPUX86State,

                                                xmm_regs[reg].XMM_L(3)));

                    break;

                case 0x22:

                    if (ot == MO_32) { /* pinsrd */

                        if (mod == 3) {

                            tcg_gen_trunc_tl_i32(cpu_tmp2_i32, cpu_regs[rm]);

                        } else {

                            tcg_gen_qemu_ld_i32(cpu_tmp2_i32, cpu_A0,

                                                s->mem_index, MO_LEUL);

                        }

                        tcg_gen_st_i32(cpu_tmp2_i32, cpu_env,

                                        offsetof(CPUX86State,

                                                xmm_regs[reg].XMM_L(val & 3)));

                    } else { /* pinsrq */

#ifdef TARGET_X86_64

                        if (mod == 3) {

                            gen_op_mov_v_reg(ot, cpu_tmp1_i64, rm);

                        } else {

                            tcg_gen_qemu_ld_i64(cpu_tmp1_i64, cpu_A0,

                                                s->mem_index, MO_LEQ);

                        }

                        tcg_gen_st_i64(cpu_tmp1_i64, cpu_env,

                                        offsetof(CPUX86State,

                                                xmm_regs[reg].XMM_Q(val & 1)));

#else

                        goto illegal_op;

#endif

                    }

                    break;

                }

                return;

            }



            if (b1) {

                op1_offset = offsetof(CPUX86State,xmm_regs[reg]);

                if (mod == 3) {

                    op2_offset = offsetof(CPUX86State,xmm_regs[rm | REX_B(s)]);

                } else {

                    op2_offset = offsetof(CPUX86State,xmm_t0);

                    gen_lea_modrm(env, s, modrm);

                    gen_ldo_env_A0(s, op2_offset);

                }

            } else {

                op1_offset = offsetof(CPUX86State,fpregs[reg].mmx);

                if (mod == 3) {

                    op2_offset = offsetof(CPUX86State,fpregs[rm].mmx);

                } else {

                    op2_offset = offsetof(CPUX86State,mmx_t0);

                    gen_lea_modrm(env, s, modrm);

                    gen_ldq_env_A0(s, op2_offset);

                }

            }

            val = cpu_ldub_code(env, s->pc++);



            if ((b & 0xfc) == 0x60) { /* pcmpXstrX */

                set_cc_op(s, CC_OP_EFLAGS);



                if (s->dflag == MO_64) {

                    /* The helper must use entire 64-bit gp registers */

                    val |= 1 << 8;

                }

            }



            tcg_gen_addi_ptr(cpu_ptr0, cpu_env, op1_offset);

            tcg_gen_addi_ptr(cpu_ptr1, cpu_env, op2_offset);

            sse_fn_eppi(cpu_env, cpu_ptr0, cpu_ptr1, tcg_const_i32(val));

            break;



        case 0x33a:

            /* Various integer extensions at 0f 3a f[0-f].  */

            b = modrm | (b1 << 8);

            modrm = cpu_ldub_code(env, s->pc++);

            reg = ((modrm >> 3) & 7) | rex_r;



            switch (b) {

            case 0x3f0: /* rorx Gy,Ey, Ib */

                if (!(s->cpuid_7_0_ebx_features & CPUID_7_0_EBX_BMI2)

                    || !(s->prefix & PREFIX_VEX)

                    || s->vex_l != 0) {

                    goto illegal_op;

                }

                ot = mo_64_32(s->dflag);

                gen_ldst_modrm(env, s, modrm, ot, OR_TMP0, 0);

                b = cpu_ldub_code(env, s->pc++);

                if (ot == MO_64) {

                    tcg_gen_rotri_tl(cpu_T[0], cpu_T[0], b & 63);

                } else {

                    tcg_gen_trunc_tl_i32(cpu_tmp2_i32, cpu_T[0]);

                    tcg_gen_rotri_i32(cpu_tmp2_i32, cpu_tmp2_i32, b & 31);

                    tcg_gen_extu_i32_tl(cpu_T[0], cpu_tmp2_i32);

                }

                gen_op_mov_reg_v(ot, reg, cpu_T[0]);

                break;



            default:

                goto illegal_op;

            }

            break;



        default:

            goto illegal_op;

        }

    } else {

        /* generic MMX or SSE operation */

        switch(b) {

        case 0x70: /* pshufx insn */

        case 0xc6: /* pshufx insn */

        case 0xc2: /* compare insns */

            s->rip_offset = 1;

            break;

        default:

            break;

        }

        if (is_xmm) {

            op1_offset = offsetof(CPUX86State,xmm_regs[reg]);

            if (mod != 3) {

                int sz = 4;



                gen_lea_modrm(env, s, modrm);

                op2_offset = offsetof(CPUX86State,xmm_t0);



                switch (b) {

                case 0x50 ... 0x5a:

                case 0x5c ... 0x5f:

                case 0xc2:

                    /* Most sse scalar operations.  */

                    if (b1 == 2) {

                        sz = 2;

                    } else if (b1 == 3) {

                        sz = 3;

                    }

                    break;



                case 0x2e:  /* ucomis[sd] */

                case 0x2f:  /* comis[sd] */

                    if (b1 == 0) {

                        sz = 2;

                    } else {

                        sz = 3;

                    }

                    break;

                }



                switch (sz) {

                case 2:

                    /* 32 bit access */

                    gen_op_ld_v(s, MO_32, cpu_T[0], cpu_A0);

                    tcg_gen_st32_tl(cpu_T[0], cpu_env,

                                    offsetof(CPUX86State,xmm_t0.XMM_L(0)));

                    break;

                case 3:

                    /* 64 bit access */

                    gen_ldq_env_A0(s, offsetof(CPUX86State, xmm_t0.XMM_D(0)));

                    break;

                default:

                    /* 128 bit access */

                    gen_ldo_env_A0(s, op2_offset);

                    break;

                }

            } else {

                rm = (modrm & 7) | REX_B(s);

                op2_offset = offsetof(CPUX86State,xmm_regs[rm]);

            }

        } else {

            op1_offset = offsetof(CPUX86State,fpregs[reg].mmx);

            if (mod != 3) {

                gen_lea_modrm(env, s, modrm);

                op2_offset = offsetof(CPUX86State,mmx_t0);

                gen_ldq_env_A0(s, op2_offset);

            } else {

                rm = (modrm & 7);

                op2_offset = offsetof(CPUX86State,fpregs[rm].mmx);

            }

        }

        switch(b) {

        case 0x0f: /* 3DNow! data insns */

            if (!(s->cpuid_ext2_features & CPUID_EXT2_3DNOW))

                goto illegal_op;

            val = cpu_ldub_code(env, s->pc++);

            sse_fn_epp = sse_op_table5[val];

            if (!sse_fn_epp) {

                goto illegal_op;

            }

            tcg_gen_addi_ptr(cpu_ptr0, cpu_env, op1_offset);

            tcg_gen_addi_ptr(cpu_ptr1, cpu_env, op2_offset);

            sse_fn_epp(cpu_env, cpu_ptr0, cpu_ptr1);

            break;

        case 0x70: /* pshufx insn */

        case 0xc6: /* pshufx insn */

            val = cpu_ldub_code(env, s->pc++);

            tcg_gen_addi_ptr(cpu_ptr0, cpu_env, op1_offset);

            tcg_gen_addi_ptr(cpu_ptr1, cpu_env, op2_offset);

            /* XXX: introduce a new table? */

            sse_fn_ppi = (SSEFunc_0_ppi)sse_fn_epp;

            sse_fn_ppi(cpu_ptr0, cpu_ptr1, tcg_const_i32(val));

            break;

        case 0xc2:

            /* compare insns */

            val = cpu_ldub_code(env, s->pc++);

            if (val >= 8)

                goto illegal_op;

            sse_fn_epp = sse_op_table4[val][b1];



            tcg_gen_addi_ptr(cpu_ptr0, cpu_env, op1_offset);

            tcg_gen_addi_ptr(cpu_ptr1, cpu_env, op2_offset);

            sse_fn_epp(cpu_env, cpu_ptr0, cpu_ptr1);

            break;

        case 0xf7:

            /* maskmov : we must prepare A0 */

            if (mod != 3)

                goto illegal_op;

            tcg_gen_mov_tl(cpu_A0, cpu_regs[R_EDI]);

            gen_extu(s->aflag, cpu_A0);

            gen_add_A0_ds_seg(s);



            tcg_gen_addi_ptr(cpu_ptr0, cpu_env, op1_offset);

            tcg_gen_addi_ptr(cpu_ptr1, cpu_env, op2_offset);

            /* XXX: introduce a new table? */

            sse_fn_eppt = (SSEFunc_0_eppt)sse_fn_epp;

            sse_fn_eppt(cpu_env, cpu_ptr0, cpu_ptr1, cpu_A0);

            break;

        default:

            tcg_gen_addi_ptr(cpu_ptr0, cpu_env, op1_offset);

            tcg_gen_addi_ptr(cpu_ptr1, cpu_env, op2_offset);

            sse_fn_epp(cpu_env, cpu_ptr0, cpu_ptr1);

            break;

        }

        if (b == 0x2e || b == 0x2f) {

            set_cc_op(s, CC_OP_EFLAGS);

        }

    }

}
