static void fill_prstatus(struct target_elf_prstatus *prstatus,

                          const TaskState *ts, int signr)

{

    (void) memset(prstatus, 0, sizeof (*prstatus));

    prstatus->pr_info.si_signo = prstatus->pr_cursig = signr;

    prstatus->pr_pid = ts->ts_tid;

    prstatus->pr_ppid = getppid();

    prstatus->pr_pgrp = getpgrp();

    prstatus->pr_sid = getsid(0);



#ifdef BSWAP_NEEDED

    bswap_prstatus(prstatus);

#endif

}
