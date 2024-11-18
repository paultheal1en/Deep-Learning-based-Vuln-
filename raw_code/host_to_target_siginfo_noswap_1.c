static inline void host_to_target_siginfo_noswap(target_siginfo_t *tinfo,

                                                 const siginfo_t *info)

{

    int sig = host_to_target_signal(info->si_signo);

    tinfo->si_signo = sig;

    tinfo->si_errno = 0;

    tinfo->si_code = info->si_code;



    if (sig == TARGET_SIGILL || sig == TARGET_SIGFPE || sig == TARGET_SIGSEGV

            || sig == TARGET_SIGBUS || sig == TARGET_SIGTRAP) {

        /* Should never come here, but who knows. The information for

           the target is irrelevant.  */

        tinfo->_sifields._sigfault._addr = 0;

    } else if (sig == TARGET_SIGIO) {

        tinfo->_sifields._sigpoll._band = info->si_band;

        tinfo->_sifields._sigpoll._fd = info->si_fd;

    } else if (sig == TARGET_SIGCHLD) {

        tinfo->_sifields._sigchld._pid = info->si_pid;

        tinfo->_sifields._sigchld._uid = info->si_uid;

        tinfo->_sifields._sigchld._status

                = host_to_target_waitstatus(info->si_status);

        tinfo->_sifields._sigchld._utime = info->si_utime;

        tinfo->_sifields._sigchld._stime = info->si_stime;

    } else if (sig >= TARGET_SIGRTMIN) {

        tinfo->_sifields._rt._pid = info->si_pid;

        tinfo->_sifields._rt._uid = info->si_uid;

        /* XXX: potential problem if 64 bit */

        tinfo->_sifields._rt._sigval.sival_ptr

                = (abi_ulong)(unsigned long)info->si_value.sival_ptr;

    }

}
