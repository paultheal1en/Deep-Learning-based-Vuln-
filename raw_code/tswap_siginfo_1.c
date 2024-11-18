static void tswap_siginfo(target_siginfo_t *tinfo,

                          const target_siginfo_t *info)

{

    int sig = info->si_signo;

    tinfo->si_signo = tswap32(sig);

    tinfo->si_errno = tswap32(info->si_errno);

    tinfo->si_code = tswap32(info->si_code);



    if (sig == TARGET_SIGILL || sig == TARGET_SIGFPE || sig == TARGET_SIGSEGV

        || sig == TARGET_SIGBUS || sig == TARGET_SIGTRAP) {

        tinfo->_sifields._sigfault._addr

            = tswapal(info->_sifields._sigfault._addr);

    } else if (sig == TARGET_SIGIO) {

        tinfo->_sifields._sigpoll._band

            = tswap32(info->_sifields._sigpoll._band);

        tinfo->_sifields._sigpoll._fd = tswap32(info->_sifields._sigpoll._fd);

    } else if (sig == TARGET_SIGCHLD) {

        tinfo->_sifields._sigchld._pid

            = tswap32(info->_sifields._sigchld._pid);

        tinfo->_sifields._sigchld._uid

            = tswap32(info->_sifields._sigchld._uid);

        tinfo->_sifields._sigchld._status

            = tswap32(info->_sifields._sigchld._status);

        tinfo->_sifields._sigchld._utime

            = tswapal(info->_sifields._sigchld._utime);

        tinfo->_sifields._sigchld._stime

            = tswapal(info->_sifields._sigchld._stime);

    } else if (sig >= TARGET_SIGRTMIN) {

        tinfo->_sifields._rt._pid = tswap32(info->_sifields._rt._pid);

        tinfo->_sifields._rt._uid = tswap32(info->_sifields._rt._uid);

        tinfo->_sifields._rt._sigval.sival_ptr

            = tswapal(info->_sifields._rt._sigval.sival_ptr);

    }

}
