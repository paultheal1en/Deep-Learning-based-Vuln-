static inline int handle_cpu_signal(unsigned long pc, unsigned long address,

                                    int is_write, sigset_t *old_set)

{

    TranslationBlock *tb;

    int ret;

    uint32_t found_pc;

    

#if defined(DEBUG_SIGNAL)

    printf("qemu: SIGSEGV pc=0x%08lx address=%08lx wr=%d oldset=0x%08lx\n", 

           pc, address, is_write, *(unsigned long *)old_set);

#endif

    /* XXX: locking issue */

    if (is_write && page_unprotect(address)) {

        return 1;

    }

    tb = tb_find_pc(pc);

    if (tb) {

        /* the PC is inside the translated code. It means that we have

           a virtual CPU fault */

        ret = cpu_x86_search_pc(tb, &found_pc, pc);

        if (ret < 0)

            return 0;

        env->eip = found_pc - tb->cs_base;

        env->cr2 = address;

        /* we restore the process signal mask as the sigreturn should

           do it (XXX: use sigsetjmp) */

        sigprocmask(SIG_SETMASK, old_set, NULL);

        raise_exception_err(EXCP0E_PAGE, 4 | (is_write << 1));

        /* never comes here */

        return 1;

    } else {

        return 0;

    }

}
