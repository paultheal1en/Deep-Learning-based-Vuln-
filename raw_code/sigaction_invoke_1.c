void sigaction_invoke(struct sigaction *action,

                      struct qemu_signalfd_siginfo *info)

{

    siginfo_t si = { 0 };

    si.si_signo = info->ssi_signo;

    si.si_errno = info->ssi_errno;

    si.si_code = info->ssi_code;



    /* Convert the minimal set of fields defined by POSIX.

     * Positive si_code values are reserved for kernel-generated

     * signals, where the valid siginfo fields are determined by

     * the signal number.  But according to POSIX, it is unspecified

     * whether SI_USER and SI_QUEUE have values less than or equal to

     * zero.

     */

    if (info->ssi_code == SI_USER || info->ssi_code == SI_QUEUE ||

        info->ssi_code <= 0) {

        /* SIGTERM, etc.  */

        si.si_pid = info->ssi_pid;

        si.si_uid = info->ssi_uid;

    } else if (info->ssi_signo == SIGILL || info->ssi_signo == SIGFPE ||

               info->ssi_signo == SIGSEGV || info->ssi_signo == SIGBUS) {

        si.si_addr = (void *)(uintptr_t)info->ssi_addr;

    } else if (info->ssi_signo == SIGCHLD) {

        si.si_pid = info->ssi_pid;

        si.si_status = info->ssi_status;

        si.si_uid = info->ssi_uid;

    }

    action->sa_sigaction(info->ssi_signo, &si, NULL);

}
