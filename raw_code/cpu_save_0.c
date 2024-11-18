void cpu_save(QEMUFile *f, void *opaque)

{

    CPUState *env = opaque;

    uint16_t fptag, fpus, fpuc, fpregs_format;

    uint32_t hflags;

    int32_t a20_mask;

    int i;



    for(i = 0; i < CPU_NB_REGS; i++)

        qemu_put_betls(f, &env->regs[i]);

    qemu_put_betls(f, &env->eip);

    qemu_put_betls(f, &env->eflags);

    hflags = env->hflags; /* XXX: suppress most of the redundant hflags */

    qemu_put_be32s(f, &hflags);



    /* FPU */

    fpuc = env->fpuc;

    fpus = (env->fpus & ~0x3800) | (env->fpstt & 0x7) << 11;

    fptag = 0;

    for(i = 0; i < 8; i++) {

        fptag |= ((!env->fptags[i]) << i);

    }



    qemu_put_be16s(f, &fpuc);

    qemu_put_be16s(f, &fpus);

    qemu_put_be16s(f, &fptag);



#ifdef USE_X86LDOUBLE

    fpregs_format = 0;

#else

    fpregs_format = 1;

#endif

    qemu_put_be16s(f, &fpregs_format);



    for(i = 0; i < 8; i++) {

#ifdef USE_X86LDOUBLE

        {

            uint64_t mant;

            uint16_t exp;

            /* we save the real CPU data (in case of MMX usage only 'mant'

               contains the MMX register */

            cpu_get_fp80(&mant, &exp, env->fpregs[i].d);

            qemu_put_be64(f, mant);

            qemu_put_be16(f, exp);

        }

#else

        /* if we use doubles for float emulation, we save the doubles to

           avoid losing information in case of MMX usage. It can give

           problems if the image is restored on a CPU where long

           doubles are used instead. */

        qemu_put_be64(f, env->fpregs[i].mmx.MMX_Q(0));

#endif

    }



    for(i = 0; i < 6; i++)

        cpu_put_seg(f, &env->segs[i]);

    cpu_put_seg(f, &env->ldt);

    cpu_put_seg(f, &env->tr);

    cpu_put_seg(f, &env->gdt);

    cpu_put_seg(f, &env->idt);



    qemu_put_be32s(f, &env->sysenter_cs);

    qemu_put_be32s(f, &env->sysenter_esp);

    qemu_put_be32s(f, &env->sysenter_eip);



    qemu_put_betls(f, &env->cr[0]);

    qemu_put_betls(f, &env->cr[2]);

    qemu_put_betls(f, &env->cr[3]);

    qemu_put_betls(f, &env->cr[4]);



    for(i = 0; i < 8; i++)

        qemu_put_betls(f, &env->dr[i]);



    /* MMU */

    a20_mask = (int32_t) env->a20_mask;

    qemu_put_sbe32s(f, &a20_mask);



    /* XMM */

    qemu_put_be32s(f, &env->mxcsr);

    for(i = 0; i < CPU_NB_REGS; i++) {

        qemu_put_be64s(f, &env->xmm_regs[i].XMM_Q(0));

        qemu_put_be64s(f, &env->xmm_regs[i].XMM_Q(1));

    }



#ifdef TARGET_X86_64

    qemu_put_be64s(f, &env->efer);

    qemu_put_be64s(f, &env->star);

    qemu_put_be64s(f, &env->lstar);

    qemu_put_be64s(f, &env->cstar);

    qemu_put_be64s(f, &env->fmask);

    qemu_put_be64s(f, &env->kernelgsbase);

#endif

    qemu_put_be32s(f, &env->smbase);



    qemu_put_be64s(f, &env->pat);

    qemu_put_be32s(f, &env->hflags2);

    

    qemu_put_be64s(f, &env->vm_hsave);

    qemu_put_be64s(f, &env->vm_vmcb);

    qemu_put_be64s(f, &env->tsc_offset);

    qemu_put_be64s(f, &env->intercept);

    qemu_put_be16s(f, &env->intercept_cr_read);

    qemu_put_be16s(f, &env->intercept_cr_write);

    qemu_put_be16s(f, &env->intercept_dr_read);

    qemu_put_be16s(f, &env->intercept_dr_write);

    qemu_put_be32s(f, &env->intercept_exceptions);

    qemu_put_8s(f, &env->v_tpr);

}
