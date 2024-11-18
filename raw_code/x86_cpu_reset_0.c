static void x86_cpu_reset(CPUState *s)

{

    X86CPU *cpu = X86_CPU(s);

    X86CPUClass *xcc = X86_CPU_GET_CLASS(cpu);

    CPUX86State *env = &cpu->env;

    target_ulong cr4;

    uint64_t xcr0;

    int i;



    xcc->parent_reset(s);



    memset(env, 0, offsetof(CPUX86State, end_reset_fields));



    tlb_flush(s, 1);



    env->old_exception = -1;



    /* init to reset state */



    env->hflags2 |= HF2_GIF_MASK;



    cpu_x86_update_cr0(env, 0x60000010);

    env->a20_mask = ~0x0;

    env->smbase = 0x30000;



    env->idt.limit = 0xffff;

    env->gdt.limit = 0xffff;

    env->ldt.limit = 0xffff;

    env->ldt.flags = DESC_P_MASK | (2 << DESC_TYPE_SHIFT);

    env->tr.limit = 0xffff;

    env->tr.flags = DESC_P_MASK | (11 << DESC_TYPE_SHIFT);



    cpu_x86_load_seg_cache(env, R_CS, 0xf000, 0xffff0000, 0xffff,

                           DESC_P_MASK | DESC_S_MASK | DESC_CS_MASK |

                           DESC_R_MASK | DESC_A_MASK);

    cpu_x86_load_seg_cache(env, R_DS, 0, 0, 0xffff,

                           DESC_P_MASK | DESC_S_MASK | DESC_W_MASK |

                           DESC_A_MASK);

    cpu_x86_load_seg_cache(env, R_ES, 0, 0, 0xffff,

                           DESC_P_MASK | DESC_S_MASK | DESC_W_MASK |

                           DESC_A_MASK);

    cpu_x86_load_seg_cache(env, R_SS, 0, 0, 0xffff,

                           DESC_P_MASK | DESC_S_MASK | DESC_W_MASK |

                           DESC_A_MASK);

    cpu_x86_load_seg_cache(env, R_FS, 0, 0, 0xffff,

                           DESC_P_MASK | DESC_S_MASK | DESC_W_MASK |

                           DESC_A_MASK);

    cpu_x86_load_seg_cache(env, R_GS, 0, 0, 0xffff,

                           DESC_P_MASK | DESC_S_MASK | DESC_W_MASK |

                           DESC_A_MASK);



    env->eip = 0xfff0;

    env->regs[R_EDX] = env->cpuid_version;



    env->eflags = 0x2;



    /* FPU init */

    for (i = 0; i < 8; i++) {

        env->fptags[i] = 1;

    }

    cpu_set_fpuc(env, 0x37f);



    env->mxcsr = 0x1f80;

    /* All units are in INIT state.  */

    env->xstate_bv = 0;



    env->pat = 0x0007040600070406ULL;

    env->msr_ia32_misc_enable = MSR_IA32_MISC_ENABLE_DEFAULT;



    memset(env->dr, 0, sizeof(env->dr));

    env->dr[6] = DR6_FIXED_1;

    env->dr[7] = DR7_FIXED_1;

    cpu_breakpoint_remove_all(s, BP_CPU);

    cpu_watchpoint_remove_all(s, BP_CPU);



    cr4 = 0;

    xcr0 = XSTATE_FP_MASK;



#ifdef CONFIG_USER_ONLY

    /* Enable all the features for user-mode.  */

    if (env->features[FEAT_1_EDX] & CPUID_SSE) {

        xcr0 |= XSTATE_SSE_MASK;

    }

    for (i = 2; i < ARRAY_SIZE(x86_ext_save_areas); i++) {

        const ExtSaveArea *esa = &x86_ext_save_areas[i];

        if ((env->features[esa->feature] & esa->bits) == esa->bits) {

            xcr0 |= 1ull << i;

        }

    }



    if (env->features[FEAT_1_ECX] & CPUID_EXT_XSAVE) {

        cr4 |= CR4_OSFXSR_MASK | CR4_OSXSAVE_MASK;

    }

    if (env->features[FEAT_7_0_EBX] & CPUID_7_0_EBX_FSGSBASE) {

        cr4 |= CR4_FSGSBASE_MASK;

    }

#endif



    env->xcr0 = xcr0;

    cpu_x86_update_cr4(env, cr4);



    /*

     * SDM 11.11.5 requires:

     *  - IA32_MTRR_DEF_TYPE MSR.E = 0

     *  - IA32_MTRR_PHYSMASKn.V = 0

     * All other bits are undefined.  For simplification, zero it all.

     */

    env->mtrr_deftype = 0;

    memset(env->mtrr_var, 0, sizeof(env->mtrr_var));

    memset(env->mtrr_fixed, 0, sizeof(env->mtrr_fixed));



#if !defined(CONFIG_USER_ONLY)

    /* We hard-wire the BSP to the first CPU. */

    apic_designate_bsp(cpu->apic_state, s->cpu_index == 0);



    s->halted = !cpu_is_bsp(cpu);



    if (kvm_enabled()) {

        kvm_arch_reset_vcpu(cpu);

    }

#endif

}
