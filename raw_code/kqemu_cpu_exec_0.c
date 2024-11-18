int kqemu_cpu_exec(CPUState *env)

{

    struct kqemu_cpu_state kcpu_state, *kenv = &kcpu_state;

    int ret, cpl, i;

#ifdef CONFIG_PROFILER

    int64_t ti;

#endif

#ifdef _WIN32

    DWORD temp;

#endif



#ifdef CONFIG_PROFILER

    ti = profile_getclock();

#endif

    LOG_INT("kqemu: cpu_exec: enter\n");

    LOG_INT_STATE(env);

    for(i = 0; i < CPU_NB_REGS; i++)

        kenv->regs[i] = env->regs[i];

    kenv->eip = env->eip;

    kenv->eflags = env->eflags;

    for(i = 0; i < 6; i++)

        kqemu_load_seg(&kenv->segs[i], &env->segs[i]);

    kqemu_load_seg(&kenv->ldt, &env->ldt);

    kqemu_load_seg(&kenv->tr, &env->tr);

    kqemu_load_seg(&kenv->gdt, &env->gdt);

    kqemu_load_seg(&kenv->idt, &env->idt);

    kenv->cr0 = env->cr[0];

    kenv->cr2 = env->cr[2];

    kenv->cr3 = env->cr[3];

    kenv->cr4 = env->cr[4];

    kenv->a20_mask = env->a20_mask;

    kenv->efer = env->efer;

    kenv->tsc_offset = 0;

    kenv->star = env->star;

    kenv->sysenter_cs = env->sysenter_cs;

    kenv->sysenter_esp = env->sysenter_esp;

    kenv->sysenter_eip = env->sysenter_eip;

#ifdef TARGET_X86_64

    kenv->lstar = env->lstar;

    kenv->cstar = env->cstar;

    kenv->fmask = env->fmask;

    kenv->kernelgsbase = env->kernelgsbase;

#endif

    if (env->dr[7] & 0xff) {

        kenv->dr7 = env->dr[7];

        kenv->dr0 = env->dr[0];

        kenv->dr1 = env->dr[1];

        kenv->dr2 = env->dr[2];

        kenv->dr3 = env->dr[3];

    } else {

        kenv->dr7 = 0;

    }

    kenv->dr6 = env->dr[6];

    cpl = (env->hflags & HF_CPL_MASK);

    kenv->cpl = cpl;

    kenv->nb_pages_to_flush = nb_pages_to_flush;

    kenv->user_only = (env->kqemu_enabled == 1);

    kenv->nb_ram_pages_to_update = nb_ram_pages_to_update;

    nb_ram_pages_to_update = 0;

    kenv->nb_modified_ram_pages = nb_modified_ram_pages;



    kqemu_reset_modified_ram_pages();



    if (env->cpuid_features & CPUID_FXSR)

        restore_native_fp_fxrstor(env);

    else

        restore_native_fp_frstor(env);



#ifdef _WIN32

    if (DeviceIoControl(kqemu_fd, KQEMU_EXEC,

                        kenv, sizeof(struct kqemu_cpu_state),

                        kenv, sizeof(struct kqemu_cpu_state),

                        &temp, NULL)) {

        ret = kenv->retval;

    } else {

        ret = -1;

    }

#else

    ioctl(kqemu_fd, KQEMU_EXEC, kenv);

    ret = kenv->retval;

#endif

    if (env->cpuid_features & CPUID_FXSR)

        save_native_fp_fxsave(env);

    else

        save_native_fp_fsave(env);



    for(i = 0; i < CPU_NB_REGS; i++)

        env->regs[i] = kenv->regs[i];

    env->eip = kenv->eip;

    env->eflags = kenv->eflags;

    for(i = 0; i < 6; i++)

        kqemu_save_seg(&env->segs[i], &kenv->segs[i]);

    cpu_x86_set_cpl(env, kenv->cpl);

    kqemu_save_seg(&env->ldt, &kenv->ldt);

    env->cr[0] = kenv->cr0;

    env->cr[4] = kenv->cr4;

    env->cr[3] = kenv->cr3;

    env->cr[2] = kenv->cr2;

    env->dr[6] = kenv->dr6;

#ifdef TARGET_X86_64

    env->kernelgsbase = kenv->kernelgsbase;

#endif



    /* flush pages as indicated by kqemu */

    if (kenv->nb_pages_to_flush >= KQEMU_FLUSH_ALL) {

        tlb_flush(env, 1);

    } else {

        for(i = 0; i < kenv->nb_pages_to_flush; i++) {

            tlb_flush_page(env, pages_to_flush[i]);

        }

    }

    nb_pages_to_flush = 0;



#ifdef CONFIG_PROFILER

    kqemu_time += profile_getclock() - ti;

    kqemu_exec_count++;

#endif



    if (kenv->nb_ram_pages_to_update > 0) {

        cpu_tlb_update_dirty(env);

    }



    if (kenv->nb_modified_ram_pages > 0) {

        for(i = 0; i < kenv->nb_modified_ram_pages; i++) {

            unsigned long addr;

            addr = modified_ram_pages[i];

            tb_invalidate_phys_page_range(addr, addr + TARGET_PAGE_SIZE, 0);

        }

    }



    /* restore the hidden flags */

    {

        unsigned int new_hflags;

#ifdef TARGET_X86_64

        if ((env->hflags & HF_LMA_MASK) &&

            (env->segs[R_CS].flags & DESC_L_MASK)) {

            /* long mode */

            new_hflags = HF_CS32_MASK | HF_SS32_MASK | HF_CS64_MASK;

        } else

#endif

        {

            /* legacy / compatibility case */

            new_hflags = (env->segs[R_CS].flags & DESC_B_MASK)

                >> (DESC_B_SHIFT - HF_CS32_SHIFT);

            new_hflags |= (env->segs[R_SS].flags & DESC_B_MASK)

                >> (DESC_B_SHIFT - HF_SS32_SHIFT);

            if (!(env->cr[0] & CR0_PE_MASK) ||

                   (env->eflags & VM_MASK) ||

                   !(env->hflags & HF_CS32_MASK)) {

                /* XXX: try to avoid this test. The problem comes from the

                   fact that is real mode or vm86 mode we only modify the

                   'base' and 'selector' fields of the segment cache to go

                   faster. A solution may be to force addseg to one in

                   translate-i386.c. */

                new_hflags |= HF_ADDSEG_MASK;

            } else {

                new_hflags |= ((env->segs[R_DS].base |

                                env->segs[R_ES].base |

                                env->segs[R_SS].base) != 0) <<

                    HF_ADDSEG_SHIFT;

            }

        }

        env->hflags = (env->hflags &

           ~(HF_CS32_MASK | HF_SS32_MASK | HF_CS64_MASK | HF_ADDSEG_MASK)) |

            new_hflags;

    }

    /* update FPU flags */

    env->hflags = (env->hflags & ~(HF_MP_MASK | HF_EM_MASK | HF_TS_MASK)) |

        ((env->cr[0] << (HF_MP_SHIFT - 1)) & (HF_MP_MASK | HF_EM_MASK | HF_TS_MASK));

    if (env->cr[4] & CR4_OSFXSR_MASK)

        env->hflags |= HF_OSFXSR_MASK;

    else

        env->hflags &= ~HF_OSFXSR_MASK;



    LOG_INT("kqemu: kqemu_cpu_exec: ret=0x%x\n", ret);

    if (ret == KQEMU_RET_SYSCALL) {

        /* syscall instruction */

        return do_syscall(env, kenv);

    } else

    if ((ret & 0xff00) == KQEMU_RET_INT) {

        env->exception_index = ret & 0xff;

        env->error_code = 0;

        env->exception_is_int = 1;

        env->exception_next_eip = kenv->next_eip;

#ifdef CONFIG_PROFILER

        kqemu_ret_int_count++;

#endif

        LOG_INT("kqemu: interrupt v=%02x:\n", env->exception_index);

        LOG_INT_STATE(env);

        return 1;

    } else if ((ret & 0xff00) == KQEMU_RET_EXCEPTION) {

        env->exception_index = ret & 0xff;

        env->error_code = kenv->error_code;

        env->exception_is_int = 0;

        env->exception_next_eip = 0;

#ifdef CONFIG_PROFILER

        kqemu_ret_excp_count++;

#endif

        LOG_INT("kqemu: exception v=%02x e=%04x:\n",

                    env->exception_index, env->error_code);

        LOG_INT_STATE(env);

        return 1;

    } else if (ret == KQEMU_RET_INTR) {

#ifdef CONFIG_PROFILER

        kqemu_ret_intr_count++;

#endif

        LOG_INT_STATE(env);

        return 0;

    } else if (ret == KQEMU_RET_SOFTMMU) {

#ifdef CONFIG_PROFILER

        {

            unsigned long pc = env->eip + env->segs[R_CS].base;

            kqemu_record_pc(pc);

        }

#endif

        LOG_INT_STATE(env);

        return 2;

    } else {

        cpu_dump_state(env, stderr, fprintf, 0);

        fprintf(stderr, "Unsupported return value: 0x%x\n", ret);

        exit(1);

    }

    return 0;

}
