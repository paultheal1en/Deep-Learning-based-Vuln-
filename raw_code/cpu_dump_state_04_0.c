void cpu_dump_state(CPUState *env, FILE *f,

                    int (*cpu_fprintf)(FILE *f, const char *fmt, ...),

                    int flags)

{

    int eflags, i, nb;

    char cc_op_name[32];

    static const char *seg_name[6] = { "ES", "CS", "SS", "DS", "FS", "GS" };



    if (kvm_enabled())

        kvm_arch_get_registers(env);



    eflags = env->eflags;

#ifdef TARGET_X86_64

    if (env->hflags & HF_CS64_MASK) {

        cpu_fprintf(f,

                    "RAX=%016" PRIx64 " RBX=%016" PRIx64 " RCX=%016" PRIx64 " RDX=%016" PRIx64 "\n"

                    "RSI=%016" PRIx64 " RDI=%016" PRIx64 " RBP=%016" PRIx64 " RSP=%016" PRIx64 "\n"

                    "R8 =%016" PRIx64 " R9 =%016" PRIx64 " R10=%016" PRIx64 " R11=%016" PRIx64 "\n"

                    "R12=%016" PRIx64 " R13=%016" PRIx64 " R14=%016" PRIx64 " R15=%016" PRIx64 "\n"

                    "RIP=%016" PRIx64 " RFL=%08x [%c%c%c%c%c%c%c] CPL=%d II=%d A20=%d SMM=%d HLT=%d\n",

                    env->regs[R_EAX],

                    env->regs[R_EBX],

                    env->regs[R_ECX],

                    env->regs[R_EDX],

                    env->regs[R_ESI],

                    env->regs[R_EDI],

                    env->regs[R_EBP],

                    env->regs[R_ESP],

                    env->regs[8],

                    env->regs[9],

                    env->regs[10],

                    env->regs[11],

                    env->regs[12],

                    env->regs[13],

                    env->regs[14],

                    env->regs[15],

                    env->eip, eflags,

                    eflags & DF_MASK ? 'D' : '-',

                    eflags & CC_O ? 'O' : '-',

                    eflags & CC_S ? 'S' : '-',

                    eflags & CC_Z ? 'Z' : '-',

                    eflags & CC_A ? 'A' : '-',

                    eflags & CC_P ? 'P' : '-',

                    eflags & CC_C ? 'C' : '-',

                    env->hflags & HF_CPL_MASK,

                    (env->hflags >> HF_INHIBIT_IRQ_SHIFT) & 1,

                    (int)(env->a20_mask >> 20) & 1,

                    (env->hflags >> HF_SMM_SHIFT) & 1,

                    env->halted);

    } else

#endif

    {

        cpu_fprintf(f, "EAX=%08x EBX=%08x ECX=%08x EDX=%08x\n"

                    "ESI=%08x EDI=%08x EBP=%08x ESP=%08x\n"

                    "EIP=%08x EFL=%08x [%c%c%c%c%c%c%c] CPL=%d II=%d A20=%d SMM=%d HLT=%d\n",

                    (uint32_t)env->regs[R_EAX],

                    (uint32_t)env->regs[R_EBX],

                    (uint32_t)env->regs[R_ECX],

                    (uint32_t)env->regs[R_EDX],

                    (uint32_t)env->regs[R_ESI],

                    (uint32_t)env->regs[R_EDI],

                    (uint32_t)env->regs[R_EBP],

                    (uint32_t)env->regs[R_ESP],

                    (uint32_t)env->eip, eflags,

                    eflags & DF_MASK ? 'D' : '-',

                    eflags & CC_O ? 'O' : '-',

                    eflags & CC_S ? 'S' : '-',

                    eflags & CC_Z ? 'Z' : '-',

                    eflags & CC_A ? 'A' : '-',

                    eflags & CC_P ? 'P' : '-',

                    eflags & CC_C ? 'C' : '-',

                    env->hflags & HF_CPL_MASK,

                    (env->hflags >> HF_INHIBIT_IRQ_SHIFT) & 1,

                    (int)(env->a20_mask >> 20) & 1,

                    (env->hflags >> HF_SMM_SHIFT) & 1,

                    env->halted);

    }



    for(i = 0; i < 6; i++) {

        cpu_x86_dump_seg_cache(env, f, cpu_fprintf, seg_name[i],

                               &env->segs[i]);

    }

    cpu_x86_dump_seg_cache(env, f, cpu_fprintf, "LDT", &env->ldt);

    cpu_x86_dump_seg_cache(env, f, cpu_fprintf, "TR", &env->tr);



#ifdef TARGET_X86_64

    if (env->hflags & HF_LMA_MASK) {

        cpu_fprintf(f, "GDT=     %016" PRIx64 " %08x\n",

                    env->gdt.base, env->gdt.limit);

        cpu_fprintf(f, "IDT=     %016" PRIx64 " %08x\n",

                    env->idt.base, env->idt.limit);

        cpu_fprintf(f, "CR0=%08x CR2=%016" PRIx64 " CR3=%016" PRIx64 " CR4=%08x\n",

                    (uint32_t)env->cr[0],

                    env->cr[2],

                    env->cr[3],

                    (uint32_t)env->cr[4]);

        for(i = 0; i < 4; i++)

            cpu_fprintf(f, "DR%d=%016" PRIx64 " ", i, env->dr[i]);

        cpu_fprintf(f, "\nDR6=%016" PRIx64 " DR7=%016" PRIx64 "\n",

                    env->dr[6], env->dr[7]);

    } else

#endif

    {

        cpu_fprintf(f, "GDT=     %08x %08x\n",

                    (uint32_t)env->gdt.base, env->gdt.limit);

        cpu_fprintf(f, "IDT=     %08x %08x\n",

                    (uint32_t)env->idt.base, env->idt.limit);

        cpu_fprintf(f, "CR0=%08x CR2=%08x CR3=%08x CR4=%08x\n",

                    (uint32_t)env->cr[0],

                    (uint32_t)env->cr[2],

                    (uint32_t)env->cr[3],

                    (uint32_t)env->cr[4]);

        for(i = 0; i < 4; i++)

            cpu_fprintf(f, "DR%d=%08x ", i, env->dr[i]);

        cpu_fprintf(f, "\nDR6=%08x DR7=%08x\n", env->dr[6], env->dr[7]);

    }

    if (flags & X86_DUMP_CCOP) {

        if ((unsigned)env->cc_op < CC_OP_NB)

            snprintf(cc_op_name, sizeof(cc_op_name), "%s", cc_op_str[env->cc_op]);

        else

            snprintf(cc_op_name, sizeof(cc_op_name), "[%d]", env->cc_op);

#ifdef TARGET_X86_64

        if (env->hflags & HF_CS64_MASK) {

            cpu_fprintf(f, "CCS=%016" PRIx64 " CCD=%016" PRIx64 " CCO=%-8s\n",

                        env->cc_src, env->cc_dst,

                        cc_op_name);

        } else

#endif

        {

            cpu_fprintf(f, "CCS=%08x CCD=%08x CCO=%-8s\n",

                        (uint32_t)env->cc_src, (uint32_t)env->cc_dst,

                        cc_op_name);

        }

    }

    if (flags & X86_DUMP_FPU) {

        int fptag;

        fptag = 0;

        for(i = 0; i < 8; i++) {

            fptag |= ((!env->fptags[i]) << i);

        }

        cpu_fprintf(f, "FCW=%04x FSW=%04x [ST=%d] FTW=%02x MXCSR=%08x\n",

                    env->fpuc,

                    (env->fpus & ~0x3800) | (env->fpstt & 0x7) << 11,

                    env->fpstt,

                    fptag,

                    env->mxcsr);

        for(i=0;i<8;i++) {

#if defined(USE_X86LDOUBLE)

            union {

                long double d;

                struct {

                    uint64_t lower;

                    uint16_t upper;

                } l;

            } tmp;

            tmp.d = env->fpregs[i].d;

            cpu_fprintf(f, "FPR%d=%016" PRIx64 " %04x",

                        i, tmp.l.lower, tmp.l.upper);

#else

            cpu_fprintf(f, "FPR%d=%016" PRIx64,

                        i, env->fpregs[i].mmx.q);

#endif

            if ((i & 1) == 1)

                cpu_fprintf(f, "\n");

            else

                cpu_fprintf(f, " ");

        }

        if (env->hflags & HF_CS64_MASK)

            nb = 16;

        else

            nb = 8;

        for(i=0;i<nb;i++) {

            cpu_fprintf(f, "XMM%02d=%08x%08x%08x%08x",

                        i,

                        env->xmm_regs[i].XMM_L(3),

                        env->xmm_regs[i].XMM_L(2),

                        env->xmm_regs[i].XMM_L(1),

                        env->xmm_regs[i].XMM_L(0));

            if ((i & 1) == 1)

                cpu_fprintf(f, "\n");

            else

                cpu_fprintf(f, " ");

        }

    }

}
