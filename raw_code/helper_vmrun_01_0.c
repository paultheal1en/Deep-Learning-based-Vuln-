void helper_vmrun(target_ulong addr)

{

    uint32_t event_inj;

    uint32_t int_ctl;



    if (loglevel & CPU_LOG_TB_IN_ASM)

        fprintf(logfile,"vmrun! " TARGET_FMT_lx "\n", addr);



    env->vm_vmcb = addr;

    regs_to_env();



    /* save the current CPU state in the hsave page */

    stq_phys(env->vm_hsave + offsetof(struct vmcb, save.gdtr.base), env->gdt.base);

    stl_phys(env->vm_hsave + offsetof(struct vmcb, save.gdtr.limit), env->gdt.limit);



    stq_phys(env->vm_hsave + offsetof(struct vmcb, save.idtr.base), env->idt.base);

    stl_phys(env->vm_hsave + offsetof(struct vmcb, save.idtr.limit), env->idt.limit);



    stq_phys(env->vm_hsave + offsetof(struct vmcb, save.cr0), env->cr[0]);

    stq_phys(env->vm_hsave + offsetof(struct vmcb, save.cr2), env->cr[2]);

    stq_phys(env->vm_hsave + offsetof(struct vmcb, save.cr3), env->cr[3]);

    stq_phys(env->vm_hsave + offsetof(struct vmcb, save.cr4), env->cr[4]);

    stq_phys(env->vm_hsave + offsetof(struct vmcb, save.cr8), env->cr[8]);

    stq_phys(env->vm_hsave + offsetof(struct vmcb, save.dr6), env->dr[6]);

    stq_phys(env->vm_hsave + offsetof(struct vmcb, save.dr7), env->dr[7]);



    stq_phys(env->vm_hsave + offsetof(struct vmcb, save.efer), env->efer);

    stq_phys(env->vm_hsave + offsetof(struct vmcb, save.rflags), compute_eflags());



    SVM_SAVE_SEG(env->vm_hsave, segs[R_ES], es);

    SVM_SAVE_SEG(env->vm_hsave, segs[R_CS], cs);

    SVM_SAVE_SEG(env->vm_hsave, segs[R_SS], ss);

    SVM_SAVE_SEG(env->vm_hsave, segs[R_DS], ds);



    stq_phys(env->vm_hsave + offsetof(struct vmcb, save.rip), EIP);

    stq_phys(env->vm_hsave + offsetof(struct vmcb, save.rsp), ESP);

    stq_phys(env->vm_hsave + offsetof(struct vmcb, save.rax), EAX);



    /* load the interception bitmaps so we do not need to access the

       vmcb in svm mode */

    /* We shift all the intercept bits so we can OR them with the TB

       flags later on */

    env->intercept            = (ldq_phys(env->vm_vmcb + offsetof(struct vmcb, control.intercept)) << INTERCEPT_INTR) | INTERCEPT_SVM_MASK;

    env->intercept_cr_read    = lduw_phys(env->vm_vmcb + offsetof(struct vmcb, control.intercept_cr_read));

    env->intercept_cr_write   = lduw_phys(env->vm_vmcb + offsetof(struct vmcb, control.intercept_cr_write));

    env->intercept_dr_read    = lduw_phys(env->vm_vmcb + offsetof(struct vmcb, control.intercept_dr_read));

    env->intercept_dr_write   = lduw_phys(env->vm_vmcb + offsetof(struct vmcb, control.intercept_dr_write));

    env->intercept_exceptions = ldl_phys(env->vm_vmcb + offsetof(struct vmcb, control.intercept_exceptions));



    env->gdt.base  = ldq_phys(env->vm_vmcb + offsetof(struct vmcb, save.gdtr.base));

    env->gdt.limit = ldl_phys(env->vm_vmcb + offsetof(struct vmcb, save.gdtr.limit));



    env->idt.base  = ldq_phys(env->vm_vmcb + offsetof(struct vmcb, save.idtr.base));

    env->idt.limit = ldl_phys(env->vm_vmcb + offsetof(struct vmcb, save.idtr.limit));



    /* clear exit_info_2 so we behave like the real hardware */

    stq_phys(env->vm_vmcb + offsetof(struct vmcb, control.exit_info_2), 0);



    cpu_x86_update_cr0(env, ldq_phys(env->vm_vmcb + offsetof(struct vmcb, save.cr0)));

    cpu_x86_update_cr4(env, ldq_phys(env->vm_vmcb + offsetof(struct vmcb, save.cr4)));

    cpu_x86_update_cr3(env, ldq_phys(env->vm_vmcb + offsetof(struct vmcb, save.cr3)));

    env->cr[2] = ldq_phys(env->vm_vmcb + offsetof(struct vmcb, save.cr2));

    int_ctl = ldl_phys(env->vm_vmcb + offsetof(struct vmcb, control.int_ctl));

    if (int_ctl & V_INTR_MASKING_MASK) {

        env->cr[8] = int_ctl & V_TPR_MASK;

        cpu_set_apic_tpr(env, env->cr[8]);

        if (env->eflags & IF_MASK)

            env->hflags |= HF_HIF_MASK;

    }



#ifdef TARGET_X86_64

    env->efer = ldq_phys(env->vm_vmcb + offsetof(struct vmcb, save.efer));

    env->hflags &= ~HF_LMA_MASK;

    if (env->efer & MSR_EFER_LMA)

       env->hflags |= HF_LMA_MASK;

#endif

    env->eflags = 0;

    load_eflags(ldq_phys(env->vm_vmcb + offsetof(struct vmcb, save.rflags)),

                ~(CC_O | CC_S | CC_Z | CC_A | CC_P | CC_C | DF_MASK));

    CC_OP = CC_OP_EFLAGS;

    CC_DST = 0xffffffff;



    SVM_LOAD_SEG(env->vm_vmcb, ES, es);

    SVM_LOAD_SEG(env->vm_vmcb, CS, cs);

    SVM_LOAD_SEG(env->vm_vmcb, SS, ss);

    SVM_LOAD_SEG(env->vm_vmcb, DS, ds);



    EIP = ldq_phys(env->vm_vmcb + offsetof(struct vmcb, save.rip));

    env->eip = EIP;

    ESP = ldq_phys(env->vm_vmcb + offsetof(struct vmcb, save.rsp));

    EAX = ldq_phys(env->vm_vmcb + offsetof(struct vmcb, save.rax));

    env->dr[7] = ldq_phys(env->vm_vmcb + offsetof(struct vmcb, save.dr7));

    env->dr[6] = ldq_phys(env->vm_vmcb + offsetof(struct vmcb, save.dr6));

    cpu_x86_set_cpl(env, ldub_phys(env->vm_vmcb + offsetof(struct vmcb, save.cpl)));



    /* FIXME: guest state consistency checks */



    switch(ldub_phys(env->vm_vmcb + offsetof(struct vmcb, control.tlb_ctl))) {

        case TLB_CONTROL_DO_NOTHING:

            break;

        case TLB_CONTROL_FLUSH_ALL_ASID:

            /* FIXME: this is not 100% correct but should work for now */

            tlb_flush(env, 1);

        break;

    }



    helper_stgi();



    regs_to_env();



    /* maybe we need to inject an event */

    event_inj = ldl_phys(env->vm_vmcb + offsetof(struct vmcb, control.event_inj));

    if (event_inj & SVM_EVTINJ_VALID) {

        uint8_t vector = event_inj & SVM_EVTINJ_VEC_MASK;

        uint16_t valid_err = event_inj & SVM_EVTINJ_VALID_ERR;

        uint32_t event_inj_err = ldl_phys(env->vm_vmcb + offsetof(struct vmcb, control.event_inj_err));

        stl_phys(env->vm_vmcb + offsetof(struct vmcb, control.event_inj), event_inj & ~SVM_EVTINJ_VALID);



        if (loglevel & CPU_LOG_TB_IN_ASM)

            fprintf(logfile, "Injecting(%#hx): ", valid_err);

        /* FIXME: need to implement valid_err */

        switch (event_inj & SVM_EVTINJ_TYPE_MASK) {

        case SVM_EVTINJ_TYPE_INTR:

                env->exception_index = vector;

                env->error_code = event_inj_err;

                env->exception_is_int = 1;

                env->exception_next_eip = -1;

                if (loglevel & CPU_LOG_TB_IN_ASM)

                    fprintf(logfile, "INTR");

                break;

        case SVM_EVTINJ_TYPE_NMI:

                env->exception_index = vector;

                env->error_code = event_inj_err;

                env->exception_is_int = 1;

                env->exception_next_eip = EIP;

                if (loglevel & CPU_LOG_TB_IN_ASM)

                    fprintf(logfile, "NMI");

                break;

        case SVM_EVTINJ_TYPE_EXEPT:

                env->exception_index = vector;

                env->error_code = event_inj_err;

                env->exception_is_int = 0;

                env->exception_next_eip = -1;

                if (loglevel & CPU_LOG_TB_IN_ASM)

                    fprintf(logfile, "EXEPT");

                break;

        case SVM_EVTINJ_TYPE_SOFT:

                env->exception_index = vector;

                env->error_code = event_inj_err;

                env->exception_is_int = 1;

                env->exception_next_eip = EIP;

                if (loglevel & CPU_LOG_TB_IN_ASM)

                    fprintf(logfile, "SOFT");

                break;

        }

        if (loglevel & CPU_LOG_TB_IN_ASM)

            fprintf(logfile, " %#x %#x\n", env->exception_index, env->error_code);

    }

    if ((int_ctl & V_IRQ_MASK) || (env->intercept & INTERCEPT_VINTR)) {

        env->interrupt_request |= CPU_INTERRUPT_VIRQ;

    }



    cpu_loop_exit();

}
