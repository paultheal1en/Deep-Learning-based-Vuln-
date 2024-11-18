int main(int argc, char **argv, char **envp)

{

    struct target_pt_regs regs1, *regs = &regs1;

    struct image_info info1, *info = &info1;

    struct linux_binprm bprm;

    TaskState *ts;

    CPUArchState *env;

    CPUState *cpu;

    int optind;

    char **target_environ, **wrk;

    char **target_argv;

    int target_argc;

    int i;

    int ret;

    int execfd;



    module_call_init(MODULE_INIT_TRACE);

    qemu_init_cpu_list();

    module_call_init(MODULE_INIT_QOM);



    if ((envlist = envlist_create()) == NULL) {

        (void) fprintf(stderr, "Unable to allocate envlist\n");

        exit(EXIT_FAILURE);

    }



    /* add current environment into the list */

    for (wrk = environ; *wrk != NULL; wrk++) {

        (void) envlist_setenv(envlist, *wrk);

    }



    /* Read the stack limit from the kernel.  If it's "unlimited",

       then we can do little else besides use the default.  */

    {

        struct rlimit lim;

        if (getrlimit(RLIMIT_STACK, &lim) == 0

            && lim.rlim_cur != RLIM_INFINITY

            && lim.rlim_cur == (target_long)lim.rlim_cur) {

            guest_stack_size = lim.rlim_cur;

        }

    }



    cpu_model = NULL;



    srand(time(NULL));



    qemu_add_opts(&qemu_trace_opts);



    optind = parse_args(argc, argv);



    if (!trace_init_backends()) {

        exit(1);

    }

    trace_init_file(trace_file);



    /* Zero out regs */

    memset(regs, 0, sizeof(struct target_pt_regs));



    /* Zero out image_info */

    memset(info, 0, sizeof(struct image_info));



    memset(&bprm, 0, sizeof (bprm));



    /* Scan interp_prefix dir for replacement files. */

    init_paths(interp_prefix);



    init_qemu_uname_release();



    if (cpu_model == NULL) {

#if defined(TARGET_I386)

#ifdef TARGET_X86_64

        cpu_model = "qemu64";

#else

        cpu_model = "qemu32";

#endif

#elif defined(TARGET_ARM)

        cpu_model = "any";

#elif defined(TARGET_UNICORE32)

        cpu_model = "any";

#elif defined(TARGET_M68K)

        cpu_model = "any";

#elif defined(TARGET_SPARC)

#ifdef TARGET_SPARC64

        cpu_model = "TI UltraSparc II";

#else

        cpu_model = "Fujitsu MB86904";

#endif

#elif defined(TARGET_MIPS)

#if defined(TARGET_ABI_MIPSN32) || defined(TARGET_ABI_MIPSN64)

        cpu_model = "5KEf";

#else

        cpu_model = "24Kf";

#endif

#elif defined TARGET_OPENRISC

        cpu_model = "or1200";

#elif defined(TARGET_PPC)

# ifdef TARGET_PPC64

        cpu_model = "POWER8";

# else

        cpu_model = "750";

# endif

#elif defined TARGET_SH4

        cpu_model = TYPE_SH7785_CPU;

#elif defined TARGET_S390X

        cpu_model = "qemu";

#else

        cpu_model = "any";

#endif

    }

    tcg_exec_init(0);

    /* NOTE: we need to init the CPU at this stage to get

       qemu_host_page_size */

    cpu = cpu_init(cpu_model);

    if (!cpu) {

        fprintf(stderr, "Unable to find CPU definition\n");

        exit(EXIT_FAILURE);

    }

    env = cpu->env_ptr;

    cpu_reset(cpu);



    thread_cpu = cpu;



    if (getenv("QEMU_STRACE")) {

        do_strace = 1;

    }



    if (getenv("QEMU_RAND_SEED")) {

        handle_arg_randseed(getenv("QEMU_RAND_SEED"));

    }



    target_environ = envlist_to_environ(envlist, NULL);

    envlist_free(envlist);



    /*

     * Now that page sizes are configured in cpu_init() we can do

     * proper page alignment for guest_base.

     */

    guest_base = HOST_PAGE_ALIGN(guest_base);



    if (reserved_va || have_guest_base) {

        guest_base = init_guest_space(guest_base, reserved_va, 0,

                                      have_guest_base);

        if (guest_base == (unsigned long)-1) {

            fprintf(stderr, "Unable to reserve 0x%lx bytes of virtual address "

                    "space for use as guest address space (check your virtual "

                    "memory ulimit setting or reserve less using -R option)\n",

                    reserved_va);

            exit(EXIT_FAILURE);

        }



        if (reserved_va) {

            mmap_next_start = reserved_va;

        }

    }



    /*

     * Read in mmap_min_addr kernel parameter.  This value is used

     * When loading the ELF image to determine whether guest_base

     * is needed.  It is also used in mmap_find_vma.

     */

    {

        FILE *fp;



        if ((fp = fopen("/proc/sys/vm/mmap_min_addr", "r")) != NULL) {

            unsigned long tmp;

            if (fscanf(fp, "%lu", &tmp) == 1) {

                mmap_min_addr = tmp;

                qemu_log_mask(CPU_LOG_PAGE, "host mmap_min_addr=0x%lx\n", mmap_min_addr);

            }

            fclose(fp);

        }

    }



    /*

     * Prepare copy of argv vector for target.

     */

    target_argc = argc - optind;

    target_argv = calloc(target_argc + 1, sizeof (char *));

    if (target_argv == NULL) {

	(void) fprintf(stderr, "Unable to allocate memory for target_argv\n");

	exit(EXIT_FAILURE);

    }



    /*

     * If argv0 is specified (using '-0' switch) we replace

     * argv[0] pointer with the given one.

     */

    i = 0;

    if (argv0 != NULL) {

        target_argv[i++] = strdup(argv0);

    }

    for (; i < target_argc; i++) {

        target_argv[i] = strdup(argv[optind + i]);

    }

    target_argv[target_argc] = NULL;



    ts = g_new0(TaskState, 1);

    init_task_state(ts);

    /* build Task State */

    ts->info = info;

    ts->bprm = &bprm;

    cpu->opaque = ts;

    task_settid(ts);



    execfd = qemu_getauxval(AT_EXECFD);

    if (execfd == 0) {

        execfd = open(filename, O_RDONLY);

        if (execfd < 0) {

            printf("Error while loading %s: %s\n", filename, strerror(errno));

            _exit(EXIT_FAILURE);

        }

    }



    ret = loader_exec(execfd, filename, target_argv, target_environ, regs,

        info, &bprm);

    if (ret != 0) {

        printf("Error while loading %s: %s\n", filename, strerror(-ret));

        _exit(EXIT_FAILURE);

    }



    for (wrk = target_environ; *wrk; wrk++) {

        free(*wrk);

    }



    free(target_environ);



    if (qemu_loglevel_mask(CPU_LOG_PAGE)) {

        qemu_log("guest_base  0x%lx\n", guest_base);

        log_page_dump();



        qemu_log("start_brk   0x" TARGET_ABI_FMT_lx "\n", info->start_brk);

        qemu_log("end_code    0x" TARGET_ABI_FMT_lx "\n", info->end_code);

        qemu_log("start_code  0x" TARGET_ABI_FMT_lx "\n", info->start_code);

        qemu_log("start_data  0x" TARGET_ABI_FMT_lx "\n", info->start_data);

        qemu_log("end_data    0x" TARGET_ABI_FMT_lx "\n", info->end_data);

        qemu_log("start_stack 0x" TARGET_ABI_FMT_lx "\n", info->start_stack);

        qemu_log("brk         0x" TARGET_ABI_FMT_lx "\n", info->brk);

        qemu_log("entry       0x" TARGET_ABI_FMT_lx "\n", info->entry);

        qemu_log("argv_start  0x" TARGET_ABI_FMT_lx "\n", info->arg_start);

        qemu_log("env_start   0x" TARGET_ABI_FMT_lx "\n",

                 info->arg_end + (abi_ulong)sizeof(abi_ulong));

        qemu_log("auxv_start  0x" TARGET_ABI_FMT_lx "\n", info->saved_auxv);

    }



    target_set_brk(info->brk);

    syscall_init();

    signal_init();



    /* Now that we've loaded the binary, GUEST_BASE is fixed.  Delay

       generating the prologue until now so that the prologue can take

       the real value of GUEST_BASE into account.  */

    tcg_prologue_init(&tcg_ctx);



#if defined(TARGET_I386)

    env->cr[0] = CR0_PG_MASK | CR0_WP_MASK | CR0_PE_MASK;

    env->hflags |= HF_PE_MASK | HF_CPL_MASK;

    if (env->features[FEAT_1_EDX] & CPUID_SSE) {

        env->cr[4] |= CR4_OSFXSR_MASK;

        env->hflags |= HF_OSFXSR_MASK;

    }

#ifndef TARGET_ABI32

    /* enable 64 bit mode if possible */

    if (!(env->features[FEAT_8000_0001_EDX] & CPUID_EXT2_LM)) {

        fprintf(stderr, "The selected x86 CPU does not support 64 bit mode\n");

        exit(EXIT_FAILURE);

    }

    env->cr[4] |= CR4_PAE_MASK;

    env->efer |= MSR_EFER_LMA | MSR_EFER_LME;

    env->hflags |= HF_LMA_MASK;

#endif



    /* flags setup : we activate the IRQs by default as in user mode */

    env->eflags |= IF_MASK;



    /* linux register setup */

#ifndef TARGET_ABI32

    env->regs[R_EAX] = regs->rax;

    env->regs[R_EBX] = regs->rbx;

    env->regs[R_ECX] = regs->rcx;

    env->regs[R_EDX] = regs->rdx;

    env->regs[R_ESI] = regs->rsi;

    env->regs[R_EDI] = regs->rdi;

    env->regs[R_EBP] = regs->rbp;

    env->regs[R_ESP] = regs->rsp;

    env->eip = regs->rip;

#else

    env->regs[R_EAX] = regs->eax;

    env->regs[R_EBX] = regs->ebx;

    env->regs[R_ECX] = regs->ecx;

    env->regs[R_EDX] = regs->edx;

    env->regs[R_ESI] = regs->esi;

    env->regs[R_EDI] = regs->edi;

    env->regs[R_EBP] = regs->ebp;

    env->regs[R_ESP] = regs->esp;

    env->eip = regs->eip;

#endif



    /* linux interrupt setup */

#ifndef TARGET_ABI32

    env->idt.limit = 511;

#else

    env->idt.limit = 255;

#endif

    env->idt.base = target_mmap(0, sizeof(uint64_t) * (env->idt.limit + 1),

                                PROT_READ|PROT_WRITE,

                                MAP_ANONYMOUS|MAP_PRIVATE, -1, 0);

    idt_table = g2h(env->idt.base);

    set_idt(0, 0);

    set_idt(1, 0);

    set_idt(2, 0);

    set_idt(3, 3);

    set_idt(4, 3);

    set_idt(5, 0);

    set_idt(6, 0);

    set_idt(7, 0);

    set_idt(8, 0);

    set_idt(9, 0);

    set_idt(10, 0);

    set_idt(11, 0);

    set_idt(12, 0);

    set_idt(13, 0);

    set_idt(14, 0);

    set_idt(15, 0);

    set_idt(16, 0);

    set_idt(17, 0);

    set_idt(18, 0);

    set_idt(19, 0);

    set_idt(0x80, 3);



    /* linux segment setup */

    {

        uint64_t *gdt_table;

        env->gdt.base = target_mmap(0, sizeof(uint64_t) * TARGET_GDT_ENTRIES,

                                    PROT_READ|PROT_WRITE,

                                    MAP_ANONYMOUS|MAP_PRIVATE, -1, 0);

        env->gdt.limit = sizeof(uint64_t) * TARGET_GDT_ENTRIES - 1;

        gdt_table = g2h(env->gdt.base);

#ifdef TARGET_ABI32

        write_dt(&gdt_table[__USER_CS >> 3], 0, 0xfffff,

                 DESC_G_MASK | DESC_B_MASK | DESC_P_MASK | DESC_S_MASK |

                 (3 << DESC_DPL_SHIFT) | (0xa << DESC_TYPE_SHIFT));

#else

        /* 64 bit code segment */

        write_dt(&gdt_table[__USER_CS >> 3], 0, 0xfffff,

                 DESC_G_MASK | DESC_B_MASK | DESC_P_MASK | DESC_S_MASK |

                 DESC_L_MASK |

                 (3 << DESC_DPL_SHIFT) | (0xa << DESC_TYPE_SHIFT));

#endif

        write_dt(&gdt_table[__USER_DS >> 3], 0, 0xfffff,

                 DESC_G_MASK | DESC_B_MASK | DESC_P_MASK | DESC_S_MASK |

                 (3 << DESC_DPL_SHIFT) | (0x2 << DESC_TYPE_SHIFT));

    }

    cpu_x86_load_seg(env, R_CS, __USER_CS);

    cpu_x86_load_seg(env, R_SS, __USER_DS);

#ifdef TARGET_ABI32

    cpu_x86_load_seg(env, R_DS, __USER_DS);

    cpu_x86_load_seg(env, R_ES, __USER_DS);

    cpu_x86_load_seg(env, R_FS, __USER_DS);

    cpu_x86_load_seg(env, R_GS, __USER_DS);

    /* This hack makes Wine work... */

    env->segs[R_FS].selector = 0;

#else

    cpu_x86_load_seg(env, R_DS, 0);

    cpu_x86_load_seg(env, R_ES, 0);

    cpu_x86_load_seg(env, R_FS, 0);

    cpu_x86_load_seg(env, R_GS, 0);

#endif

#elif defined(TARGET_AARCH64)

    {

        int i;



        if (!(arm_feature(env, ARM_FEATURE_AARCH64))) {

            fprintf(stderr,

                    "The selected ARM CPU does not support 64 bit mode\n");

            exit(EXIT_FAILURE);

        }



        for (i = 0; i < 31; i++) {

            env->xregs[i] = regs->regs[i];

        }

        env->pc = regs->pc;

        env->xregs[31] = regs->sp;

    }

#elif defined(TARGET_ARM)

    {

        int i;

        cpsr_write(env, regs->uregs[16], CPSR_USER | CPSR_EXEC,

                   CPSRWriteByInstr);

        for(i = 0; i < 16; i++) {

            env->regs[i] = regs->uregs[i];

        }

#ifdef TARGET_WORDS_BIGENDIAN

        /* Enable BE8.  */

        if (EF_ARM_EABI_VERSION(info->elf_flags) >= EF_ARM_EABI_VER4

            && (info->elf_flags & EF_ARM_BE8)) {

            env->uncached_cpsr |= CPSR_E;

            env->cp15.sctlr_el[1] |= SCTLR_E0E;

        } else {

            env->cp15.sctlr_el[1] |= SCTLR_B;

        }

#endif

    }

#elif defined(TARGET_UNICORE32)

    {

        int i;

        cpu_asr_write(env, regs->uregs[32], 0xffffffff);

        for (i = 0; i < 32; i++) {

            env->regs[i] = regs->uregs[i];

        }

    }

#elif defined(TARGET_SPARC)

    {

        int i;

	env->pc = regs->pc;

	env->npc = regs->npc;

        env->y = regs->y;

        for(i = 0; i < 8; i++)

            env->gregs[i] = regs->u_regs[i];

        for(i = 0; i < 8; i++)

            env->regwptr[i] = regs->u_regs[i + 8];

    }

#elif defined(TARGET_PPC)

    {

        int i;



#if defined(TARGET_PPC64)

        int flag = (env->insns_flags2 & PPC2_BOOKE206) ? MSR_CM : MSR_SF;

#if defined(TARGET_ABI32)

        env->msr &= ~((target_ulong)1 << flag);

#else

        env->msr |= (target_ulong)1 << flag;

#endif

#endif

        env->nip = regs->nip;

        for(i = 0; i < 32; i++) {

            env->gpr[i] = regs->gpr[i];

        }

    }

#elif defined(TARGET_M68K)

    {

        env->pc = regs->pc;

        env->dregs[0] = regs->d0;

        env->dregs[1] = regs->d1;

        env->dregs[2] = regs->d2;

        env->dregs[3] = regs->d3;

        env->dregs[4] = regs->d4;

        env->dregs[5] = regs->d5;

        env->dregs[6] = regs->d6;

        env->dregs[7] = regs->d7;

        env->aregs[0] = regs->a0;

        env->aregs[1] = regs->a1;

        env->aregs[2] = regs->a2;

        env->aregs[3] = regs->a3;

        env->aregs[4] = regs->a4;

        env->aregs[5] = regs->a5;

        env->aregs[6] = regs->a6;

        env->aregs[7] = regs->usp;

        env->sr = regs->sr;

        ts->sim_syscalls = 1;

    }

#elif defined(TARGET_MICROBLAZE)

    {

        env->regs[0] = regs->r0;

        env->regs[1] = regs->r1;

        env->regs[2] = regs->r2;

        env->regs[3] = regs->r3;

        env->regs[4] = regs->r4;

        env->regs[5] = regs->r5;

        env->regs[6] = regs->r6;

        env->regs[7] = regs->r7;

        env->regs[8] = regs->r8;

        env->regs[9] = regs->r9;

        env->regs[10] = regs->r10;

        env->regs[11] = regs->r11;

        env->regs[12] = regs->r12;

        env->regs[13] = regs->r13;

        env->regs[14] = regs->r14;

        env->regs[15] = regs->r15;	    

        env->regs[16] = regs->r16;	    

        env->regs[17] = regs->r17;	    

        env->regs[18] = regs->r18;	    

        env->regs[19] = regs->r19;	    

        env->regs[20] = regs->r20;	    

        env->regs[21] = regs->r21;	    

        env->regs[22] = regs->r22;	    

        env->regs[23] = regs->r23;	    

        env->regs[24] = regs->r24;	    

        env->regs[25] = regs->r25;	    

        env->regs[26] = regs->r26;	    

        env->regs[27] = regs->r27;	    

        env->regs[28] = regs->r28;	    

        env->regs[29] = regs->r29;	    

        env->regs[30] = regs->r30;	    

        env->regs[31] = regs->r31;	    

        env->sregs[SR_PC] = regs->pc;

    }

#elif defined(TARGET_MIPS)

    {

        int i;



        for(i = 0; i < 32; i++) {

            env->active_tc.gpr[i] = regs->regs[i];

        }

        env->active_tc.PC = regs->cp0_epc & ~(target_ulong)1;

        if (regs->cp0_epc & 1) {

            env->hflags |= MIPS_HFLAG_M16;

        }

        if (((info->elf_flags & EF_MIPS_NAN2008) != 0) !=

            ((env->active_fpu.fcr31 & (1 << FCR31_NAN2008)) != 0)) {

            if ((env->active_fpu.fcr31_rw_bitmask &

                  (1 << FCR31_NAN2008)) == 0) {

                fprintf(stderr, "ELF binary's NaN mode not supported by CPU\n");

                exit(1);

            }

            if ((info->elf_flags & EF_MIPS_NAN2008) != 0) {

                env->active_fpu.fcr31 |= (1 << FCR31_NAN2008);

            } else {

                env->active_fpu.fcr31 &= ~(1 << FCR31_NAN2008);

            }

            restore_snan_bit_mode(env);

        }

    }

#elif defined(TARGET_NIOS2)

    {

        env->regs[0] = 0;

        env->regs[1] = regs->r1;

        env->regs[2] = regs->r2;

        env->regs[3] = regs->r3;

        env->regs[4] = regs->r4;

        env->regs[5] = regs->r5;

        env->regs[6] = regs->r6;

        env->regs[7] = regs->r7;

        env->regs[8] = regs->r8;

        env->regs[9] = regs->r9;

        env->regs[10] = regs->r10;

        env->regs[11] = regs->r11;

        env->regs[12] = regs->r12;

        env->regs[13] = regs->r13;

        env->regs[14] = regs->r14;

        env->regs[15] = regs->r15;

        /* TODO: unsigned long  orig_r2; */

        env->regs[R_RA] = regs->ra;

        env->regs[R_FP] = regs->fp;

        env->regs[R_SP] = regs->sp;

        env->regs[R_GP] = regs->gp;

        env->regs[CR_ESTATUS] = regs->estatus;

        env->regs[R_EA] = regs->ea;

        /* TODO: unsigned long  orig_r7; */



        /* Emulate eret when starting thread. */

        env->regs[R_PC] = regs->ea;

    }

#elif defined(TARGET_OPENRISC)

    {

        int i;



        for (i = 0; i < 32; i++) {

            env->gpr[i] = regs->gpr[i];

        }

        env->pc = regs->pc;

        cpu_set_sr(env, regs->sr);

    }

#elif defined(TARGET_SH4)

    {

        int i;



        for(i = 0; i < 16; i++) {

            env->gregs[i] = regs->regs[i];

        }

        env->pc = regs->pc;

    }

#elif defined(TARGET_ALPHA)

    {

        int i;



        for(i = 0; i < 28; i++) {

            env->ir[i] = ((abi_ulong *)regs)[i];

        }

        env->ir[IR_SP] = regs->usp;

        env->pc = regs->pc;

    }

#elif defined(TARGET_CRIS)

    {

	    env->regs[0] = regs->r0;

	    env->regs[1] = regs->r1;

	    env->regs[2] = regs->r2;

	    env->regs[3] = regs->r3;

	    env->regs[4] = regs->r4;

	    env->regs[5] = regs->r5;

	    env->regs[6] = regs->r6;

	    env->regs[7] = regs->r7;

	    env->regs[8] = regs->r8;

	    env->regs[9] = regs->r9;

	    env->regs[10] = regs->r10;

	    env->regs[11] = regs->r11;

	    env->regs[12] = regs->r12;

	    env->regs[13] = regs->r13;

	    env->regs[14] = info->start_stack;

	    env->regs[15] = regs->acr;	    

	    env->pc = regs->erp;

    }

#elif defined(TARGET_S390X)

    {

            int i;

            for (i = 0; i < 16; i++) {

                env->regs[i] = regs->gprs[i];

            }

            env->psw.mask = regs->psw.mask;

            env->psw.addr = regs->psw.addr;

    }

#elif defined(TARGET_TILEGX)

    {

        int i;

        for (i = 0; i < TILEGX_R_COUNT; i++) {

            env->regs[i] = regs->regs[i];

        }

        for (i = 0; i < TILEGX_SPR_COUNT; i++) {

            env->spregs[i] = 0;

        }

        env->pc = regs->pc;

    }

#elif defined(TARGET_HPPA)

    {

        int i;

        for (i = 1; i < 32; i++) {

            env->gr[i] = regs->gr[i];

        }

        env->iaoq_f = regs->iaoq[0];

        env->iaoq_b = regs->iaoq[1];

    }

#else

#error unsupported target CPU

#endif



#if defined(TARGET_ARM) || defined(TARGET_M68K) || defined(TARGET_UNICORE32)

    ts->stack_base = info->start_stack;

    ts->heap_base = info->brk;

    /* This will be filled in on the first SYS_HEAPINFO call.  */

    ts->heap_limit = 0;

#endif



    if (gdbstub_port) {

        if (gdbserver_start(gdbstub_port) < 0) {

            fprintf(stderr, "qemu: could not open gdbserver on port %d\n",

                    gdbstub_port);

            exit(EXIT_FAILURE);

        }

        gdb_handlesig(cpu, 0);

    }

    cpu_loop(env);

    /* never exits */

    return 0;

}
