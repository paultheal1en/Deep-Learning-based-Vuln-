void helper_vmexit(CPUX86State *env, uint32_t exit_code, uint64_t exit_info_1)

{

    CPUState *cs = CPU(x86_env_get_cpu(env));

    uint32_t int_ctl;



    qemu_log_mask(CPU_LOG_TB_IN_ASM, "vmexit(%08x, %016" PRIx64 ", %016"

                  PRIx64 ", " TARGET_FMT_lx ")!\n",

                  exit_code, exit_info_1,

                  ldq_phys(cs->as, env->vm_vmcb + offsetof(struct vmcb,

                                                   control.exit_info_2)),

                  env->eip);



    if (env->hflags & HF_INHIBIT_IRQ_MASK) {

        stl_phys(cs->as,

                 env->vm_vmcb + offsetof(struct vmcb, control.int_state),

                 SVM_INTERRUPT_SHADOW_MASK);

        env->hflags &= ~HF_INHIBIT_IRQ_MASK;

    } else {

        stl_phys(cs->as,

                 env->vm_vmcb + offsetof(struct vmcb, control.int_state), 0);

    }



    /* Save the VM state in the vmcb */

    svm_save_seg(env, env->vm_vmcb + offsetof(struct vmcb, save.es),

                 &env->segs[R_ES]);

    svm_save_seg(env, env->vm_vmcb + offsetof(struct vmcb, save.cs),

                 &env->segs[R_CS]);

    svm_save_seg(env, env->vm_vmcb + offsetof(struct vmcb, save.ss),

                 &env->segs[R_SS]);

    svm_save_seg(env, env->vm_vmcb + offsetof(struct vmcb, save.ds),

                 &env->segs[R_DS]);



    stq_phys(cs->as, env->vm_vmcb + offsetof(struct vmcb, save.gdtr.base),

             env->gdt.base);

    stl_phys(cs->as, env->vm_vmcb + offsetof(struct vmcb, save.gdtr.limit),

             env->gdt.limit);



    stq_phys(cs->as, env->vm_vmcb + offsetof(struct vmcb, save.idtr.base),

             env->idt.base);

    stl_phys(cs->as, env->vm_vmcb + offsetof(struct vmcb, save.idtr.limit),

             env->idt.limit);



    stq_phys(cs->as,

             env->vm_vmcb + offsetof(struct vmcb, save.efer), env->efer);

    stq_phys(cs->as,

             env->vm_vmcb + offsetof(struct vmcb, save.cr0), env->cr[0]);

    stq_phys(cs->as,

             env->vm_vmcb + offsetof(struct vmcb, save.cr2), env->cr[2]);

    stq_phys(cs->as,

             env->vm_vmcb + offsetof(struct vmcb, save.cr3), env->cr[3]);

    stq_phys(cs->as,

             env->vm_vmcb + offsetof(struct vmcb, save.cr4), env->cr[4]);



    int_ctl = ldl_phys(cs->as,

                       env->vm_vmcb + offsetof(struct vmcb, control.int_ctl));

    int_ctl &= ~(V_TPR_MASK | V_IRQ_MASK);

    int_ctl |= env->v_tpr & V_TPR_MASK;

    if (cs->interrupt_request & CPU_INTERRUPT_VIRQ) {

        int_ctl |= V_IRQ_MASK;

    }

    stl_phys(cs->as,

             env->vm_vmcb + offsetof(struct vmcb, control.int_ctl), int_ctl);



    stq_phys(cs->as, env->vm_vmcb + offsetof(struct vmcb, save.rflags),

             cpu_compute_eflags(env));

    stq_phys(cs->as, env->vm_vmcb + offsetof(struct vmcb, save.rip),

             env->eip);

    stq_phys(cs->as,

             env->vm_vmcb + offsetof(struct vmcb, save.rsp), env->regs[R_ESP]);

    stq_phys(cs->as,

             env->vm_vmcb + offsetof(struct vmcb, save.rax), env->regs[R_EAX]);

    stq_phys(cs->as,

             env->vm_vmcb + offsetof(struct vmcb, save.dr7), env->dr[7]);

    stq_phys(cs->as,

             env->vm_vmcb + offsetof(struct vmcb, save.dr6), env->dr[6]);

    stb_phys(cs->as, env->vm_vmcb + offsetof(struct vmcb, save.cpl),

             env->hflags & HF_CPL_MASK);



    /* Reload the host state from vm_hsave */

    env->hflags2 &= ~(HF2_HIF_MASK | HF2_VINTR_MASK);

    env->hflags &= ~HF_SVMI_MASK;

    env->intercept = 0;

    env->intercept_exceptions = 0;

    cs->interrupt_request &= ~CPU_INTERRUPT_VIRQ;

    env->tsc_offset = 0;



    env->gdt.base  = ldq_phys(cs->as, env->vm_hsave + offsetof(struct vmcb,

                                                       save.gdtr.base));

    env->gdt.limit = ldl_phys(cs->as, env->vm_hsave + offsetof(struct vmcb,

                                                       save.gdtr.limit));



    env->idt.base  = ldq_phys(cs->as, env->vm_hsave + offsetof(struct vmcb,

                                                       save.idtr.base));

    env->idt.limit = ldl_phys(cs->as, env->vm_hsave + offsetof(struct vmcb,

                                                       save.idtr.limit));



    cpu_x86_update_cr0(env, ldq_phys(cs->as,

                                     env->vm_hsave + offsetof(struct vmcb,

                                                              save.cr0)) |

                       CR0_PE_MASK);

    cpu_x86_update_cr4(env, ldq_phys(cs->as,

                                     env->vm_hsave + offsetof(struct vmcb,

                                                              save.cr4)));

    cpu_x86_update_cr3(env, ldq_phys(cs->as,

                                     env->vm_hsave + offsetof(struct vmcb,

                                                              save.cr3)));

    /* we need to set the efer after the crs so the hidden flags get

       set properly */

    cpu_load_efer(env, ldq_phys(cs->as, env->vm_hsave + offsetof(struct vmcb,

                                                         save.efer)));

    env->eflags = 0;

    cpu_load_eflags(env, ldq_phys(cs->as,

                                  env->vm_hsave + offsetof(struct vmcb,

                                                           save.rflags)),

                    ~(CC_O | CC_S | CC_Z | CC_A | CC_P | CC_C | DF_MASK |

                      VM_MASK));

    CC_OP = CC_OP_EFLAGS;



    svm_load_seg_cache(env, env->vm_hsave + offsetof(struct vmcb, save.es),

                       R_ES);

    svm_load_seg_cache(env, env->vm_hsave + offsetof(struct vmcb, save.cs),

                       R_CS);

    svm_load_seg_cache(env, env->vm_hsave + offsetof(struct vmcb, save.ss),

                       R_SS);

    svm_load_seg_cache(env, env->vm_hsave + offsetof(struct vmcb, save.ds),

                       R_DS);



    env->eip = ldq_phys(cs->as,

                        env->vm_hsave + offsetof(struct vmcb, save.rip));

    env->regs[R_ESP] = ldq_phys(cs->as, env->vm_hsave +

                                offsetof(struct vmcb, save.rsp));

    env->regs[R_EAX] = ldq_phys(cs->as, env->vm_hsave +

                                offsetof(struct vmcb, save.rax));



    env->dr[6] = ldq_phys(cs->as,

                          env->vm_hsave + offsetof(struct vmcb, save.dr6));

    env->dr[7] = ldq_phys(cs->as,

                          env->vm_hsave + offsetof(struct vmcb, save.dr7));



    /* other setups */

    cpu_x86_set_cpl(env, 0);

    stq_phys(cs->as, env->vm_vmcb + offsetof(struct vmcb, control.exit_code),

             exit_code);

    stq_phys(cs->as, env->vm_vmcb + offsetof(struct vmcb, control.exit_info_1),

             exit_info_1);



    stl_phys(cs->as,

             env->vm_vmcb + offsetof(struct vmcb, control.exit_int_info),

             ldl_phys(cs->as, env->vm_vmcb + offsetof(struct vmcb,

                                              control.event_inj)));

    stl_phys(cs->as,

             env->vm_vmcb + offsetof(struct vmcb, control.exit_int_info_err),

             ldl_phys(cs->as, env->vm_vmcb + offsetof(struct vmcb,

                                              control.event_inj_err)));

    stl_phys(cs->as,

             env->vm_vmcb + offsetof(struct vmcb, control.event_inj), 0);



    env->hflags2 &= ~HF2_GIF_MASK;

    /* FIXME: Resets the current ASID register to zero (host ASID). */



    /* Clears the V_IRQ and V_INTR_MASKING bits inside the processor. */



    /* Clears the TSC_OFFSET inside the processor. */



    /* If the host is in PAE mode, the processor reloads the host's PDPEs

       from the page table indicated the host's CR3. If the PDPEs contain

       illegal state, the processor causes a shutdown. */



    /* Disables all breakpoints in the host DR7 register. */



    /* Checks the reloaded host state for consistency. */



    /* If the host's rIP reloaded by #VMEXIT is outside the limit of the

       host's code segment or non-canonical (in the case of long mode), a

       #GP fault is delivered inside the host. */



    /* remove any pending exception */

    cs->exception_index = -1;

    env->error_code = 0;

    env->old_exception = -1;



    cpu_loop_exit(cs);

}
