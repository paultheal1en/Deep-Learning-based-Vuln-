static inline void helper_ret_protected(CPUX86State *env, int shift,

                                        int is_iret, int addend)

{

    uint32_t new_cs, new_eflags, new_ss;

    uint32_t new_es, new_ds, new_fs, new_gs;

    uint32_t e1, e2, ss_e1, ss_e2;

    int cpl, dpl, rpl, eflags_mask, iopl;

    target_ulong ssp, sp, new_eip, new_esp, sp_mask;



#ifdef TARGET_X86_64

    if (shift == 2) {

        sp_mask = -1;

    } else

#endif

    {

        sp_mask = get_sp_mask(env->segs[R_SS].flags);

    }

    sp = env->regs[R_ESP];

    ssp = env->segs[R_SS].base;

    new_eflags = 0; /* avoid warning */

#ifdef TARGET_X86_64

    if (shift == 2) {

        POPQ(sp, new_eip);

        POPQ(sp, new_cs);

        new_cs &= 0xffff;

        if (is_iret) {

            POPQ(sp, new_eflags);

        }

    } else

#endif

    {

        if (shift == 1) {

            /* 32 bits */

            POPL(ssp, sp, sp_mask, new_eip);

            POPL(ssp, sp, sp_mask, new_cs);

            new_cs &= 0xffff;

            if (is_iret) {

                POPL(ssp, sp, sp_mask, new_eflags);

                if (new_eflags & VM_MASK) {

                    goto return_to_vm86;

                }

            }

        } else {

            /* 16 bits */

            POPW(ssp, sp, sp_mask, new_eip);

            POPW(ssp, sp, sp_mask, new_cs);

            if (is_iret) {

                POPW(ssp, sp, sp_mask, new_eflags);

            }

        }

    }

    LOG_PCALL("lret new %04x:" TARGET_FMT_lx " s=%d addend=0x%x\n",

              new_cs, new_eip, shift, addend);

    LOG_PCALL_STATE(CPU(x86_env_get_cpu(env)));

    if ((new_cs & 0xfffc) == 0) {

        raise_exception_err(env, EXCP0D_GPF, new_cs & 0xfffc);

    }

    if (load_segment(env, &e1, &e2, new_cs) != 0) {

        raise_exception_err(env, EXCP0D_GPF, new_cs & 0xfffc);

    }

    if (!(e2 & DESC_S_MASK) ||

        !(e2 & DESC_CS_MASK)) {

        raise_exception_err(env, EXCP0D_GPF, new_cs & 0xfffc);

    }

    cpl = env->hflags & HF_CPL_MASK;

    rpl = new_cs & 3;

    if (rpl < cpl) {

        raise_exception_err(env, EXCP0D_GPF, new_cs & 0xfffc);

    }

    dpl = (e2 >> DESC_DPL_SHIFT) & 3;

    if (e2 & DESC_C_MASK) {

        if (dpl > rpl) {

            raise_exception_err(env, EXCP0D_GPF, new_cs & 0xfffc);

        }

    } else {

        if (dpl != rpl) {

            raise_exception_err(env, EXCP0D_GPF, new_cs & 0xfffc);

        }

    }

    if (!(e2 & DESC_P_MASK)) {

        raise_exception_err(env, EXCP0B_NOSEG, new_cs & 0xfffc);

    }



    sp += addend;

    if (rpl == cpl && (!(env->hflags & HF_CS64_MASK) ||

                       ((env->hflags & HF_CS64_MASK) && !is_iret))) {

        /* return to same privilege level */

        cpu_x86_load_seg_cache(env, R_CS, new_cs,

                       get_seg_base(e1, e2),

                       get_seg_limit(e1, e2),

                       e2);

    } else {

        /* return to different privilege level */

#ifdef TARGET_X86_64

        if (shift == 2) {

            POPQ(sp, new_esp);

            POPQ(sp, new_ss);

            new_ss &= 0xffff;

        } else

#endif

        {

            if (shift == 1) {

                /* 32 bits */

                POPL(ssp, sp, sp_mask, new_esp);

                POPL(ssp, sp, sp_mask, new_ss);

                new_ss &= 0xffff;

            } else {

                /* 16 bits */

                POPW(ssp, sp, sp_mask, new_esp);

                POPW(ssp, sp, sp_mask, new_ss);

            }

        }

        LOG_PCALL("new ss:esp=%04x:" TARGET_FMT_lx "\n",

                  new_ss, new_esp);

        if ((new_ss & 0xfffc) == 0) {

#ifdef TARGET_X86_64

            /* NULL ss is allowed in long mode if cpl != 3 */

            /* XXX: test CS64? */

            if ((env->hflags & HF_LMA_MASK) && rpl != 3) {

                cpu_x86_load_seg_cache(env, R_SS, new_ss,

                                       0, 0xffffffff,

                                       DESC_G_MASK | DESC_B_MASK | DESC_P_MASK |

                                       DESC_S_MASK | (rpl << DESC_DPL_SHIFT) |

                                       DESC_W_MASK | DESC_A_MASK);

                ss_e2 = DESC_B_MASK; /* XXX: should not be needed? */

            } else

#endif

            {

                raise_exception_err(env, EXCP0D_GPF, 0);

            }

        } else {

            if ((new_ss & 3) != rpl) {

                raise_exception_err(env, EXCP0D_GPF, new_ss & 0xfffc);

            }

            if (load_segment(env, &ss_e1, &ss_e2, new_ss) != 0) {

                raise_exception_err(env, EXCP0D_GPF, new_ss & 0xfffc);

            }

            if (!(ss_e2 & DESC_S_MASK) ||

                (ss_e2 & DESC_CS_MASK) ||

                !(ss_e2 & DESC_W_MASK)) {

                raise_exception_err(env, EXCP0D_GPF, new_ss & 0xfffc);

            }

            dpl = (ss_e2 >> DESC_DPL_SHIFT) & 3;

            if (dpl != rpl) {

                raise_exception_err(env, EXCP0D_GPF, new_ss & 0xfffc);

            }

            if (!(ss_e2 & DESC_P_MASK)) {

                raise_exception_err(env, EXCP0B_NOSEG, new_ss & 0xfffc);

            }

            cpu_x86_load_seg_cache(env, R_SS, new_ss,

                                   get_seg_base(ss_e1, ss_e2),

                                   get_seg_limit(ss_e1, ss_e2),

                                   ss_e2);

        }



        cpu_x86_load_seg_cache(env, R_CS, new_cs,

                       get_seg_base(e1, e2),

                       get_seg_limit(e1, e2),

                       e2);

        cpu_x86_set_cpl(env, rpl);

        sp = new_esp;

#ifdef TARGET_X86_64

        if (env->hflags & HF_CS64_MASK) {

            sp_mask = -1;

        } else

#endif

        {

            sp_mask = get_sp_mask(ss_e2);

        }



        /* validate data segments */

        validate_seg(env, R_ES, rpl);

        validate_seg(env, R_DS, rpl);

        validate_seg(env, R_FS, rpl);

        validate_seg(env, R_GS, rpl);



        sp += addend;

    }

    SET_ESP(sp, sp_mask);

    env->eip = new_eip;

    if (is_iret) {

        /* NOTE: 'cpl' is the _old_ CPL */

        eflags_mask = TF_MASK | AC_MASK | ID_MASK | RF_MASK | NT_MASK;

        if (cpl == 0) {

            eflags_mask |= IOPL_MASK;

        }

        iopl = (env->eflags >> IOPL_SHIFT) & 3;

        if (cpl <= iopl) {

            eflags_mask |= IF_MASK;

        }

        if (shift == 0) {

            eflags_mask &= 0xffff;

        }

        cpu_load_eflags(env, new_eflags, eflags_mask);

    }

    return;



 return_to_vm86:

    POPL(ssp, sp, sp_mask, new_esp);

    POPL(ssp, sp, sp_mask, new_ss);

    POPL(ssp, sp, sp_mask, new_es);

    POPL(ssp, sp, sp_mask, new_ds);

    POPL(ssp, sp, sp_mask, new_fs);

    POPL(ssp, sp, sp_mask, new_gs);



    /* modify processor state */

    cpu_load_eflags(env, new_eflags, TF_MASK | AC_MASK | ID_MASK |

                    IF_MASK | IOPL_MASK | VM_MASK | NT_MASK | VIF_MASK |

                    VIP_MASK);

    load_seg_vm(env, R_CS, new_cs & 0xffff);

    cpu_x86_set_cpl(env, 3);

    load_seg_vm(env, R_SS, new_ss & 0xffff);

    load_seg_vm(env, R_ES, new_es & 0xffff);

    load_seg_vm(env, R_DS, new_ds & 0xffff);

    load_seg_vm(env, R_FS, new_fs & 0xffff);

    load_seg_vm(env, R_GS, new_gs & 0xffff);



    env->eip = new_eip & 0xffff;

    env->regs[R_ESP] = new_esp;

}
