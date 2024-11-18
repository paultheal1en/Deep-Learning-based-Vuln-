long do_syscall(void *cpu_env, int num, long arg1, long arg2, long arg3,

                long arg4, long arg5, long arg6)

{

    long ret;

    struct stat st;

    struct statfs stfs;

    void *p;

   

#ifdef DEBUG

    gemu_log("syscall %d", num);

#endif

    switch(num) {

    case TARGET_NR_exit:

#ifdef HAVE_GPROF

        _mcleanup();

#endif

        gdb_exit(cpu_env, arg1);

        /* XXX: should free thread stack and CPU env */

        _exit(arg1);

        ret = 0; /* avoid warning */

        break;

    case TARGET_NR_read:

        page_unprotect_range(arg2, arg3);

        p = lock_user(arg2, arg3, 0);

        ret = get_errno(read(arg1, p, arg3));

        unlock_user(p, arg2, ret);

        break;

    case TARGET_NR_write:

        p = lock_user(arg2, arg3, 1);

        ret = get_errno(write(arg1, p, arg3));

        unlock_user(p, arg2, 0);

        break;

    case TARGET_NR_open:

        p = lock_user_string(arg1);

        ret = get_errno(open(path(p),

                             target_to_host_bitmask(arg2, fcntl_flags_tbl),

                             arg3));

        unlock_user(p, arg1, 0);

        break;

    case TARGET_NR_close:

        ret = get_errno(close(arg1));

        break;

    case TARGET_NR_brk:

        ret = do_brk(arg1);

        break;

    case TARGET_NR_fork:

        ret = get_errno(do_fork(cpu_env, SIGCHLD, 0));

        break;

#ifdef TARGET_NR_waitpid

    case TARGET_NR_waitpid:

        {

            int status;

            ret = get_errno(waitpid(arg1, &status, arg3));

            if (!is_error(ret) && arg2)

                tput32(arg2, status);

        }

        break;

#endif

#ifdef TARGET_NR_creat /* not on alpha */

    case TARGET_NR_creat:

        p = lock_user_string(arg1);

        ret = get_errno(creat(p, arg2));

        unlock_user(p, arg1, 0);

        break;

#endif

    case TARGET_NR_link:

        {

            void * p2;

            p = lock_user_string(arg1);

            p2 = lock_user_string(arg2);

            ret = get_errno(link(p, p2));

            unlock_user(p2, arg2, 0);

            unlock_user(p, arg1, 0);

        }

        break;

    case TARGET_NR_unlink:

        p = lock_user_string(arg1);

        ret = get_errno(unlink(p));

        unlock_user(p, arg1, 0);

        break;

    case TARGET_NR_execve:

        {

            char **argp, **envp;

            int argc, envc;

            target_ulong gp;

            target_ulong guest_argp;

            target_ulong guest_envp;

            target_ulong addr;

            char **q;



            argc = 0;

            guest_argp = arg2;

            for (gp = guest_argp; tgetl(gp); gp++)

                argc++;

            envc = 0;

            guest_envp = arg3;

            for (gp = guest_envp; tgetl(gp); gp++)

                envc++;



            argp = alloca((argc + 1) * sizeof(void *));

            envp = alloca((envc + 1) * sizeof(void *));



            for (gp = guest_argp, q = argp; ;

                  gp += sizeof(target_ulong), q++) {

                addr = tgetl(gp);

                if (!addr)

                    break;

                *q = lock_user_string(addr);

            }

            *q = NULL;



            for (gp = guest_envp, q = envp; ;

                  gp += sizeof(target_ulong), q++) {

                addr = tgetl(gp);

                if (!addr)

                    break;

                *q = lock_user_string(addr);

            }

            *q = NULL;



            p = lock_user_string(arg1);

            ret = get_errno(execve(p, argp, envp));

            unlock_user(p, arg1, 0);



            for (gp = guest_argp, q = argp; *q;

                  gp += sizeof(target_ulong), q++) {

                addr = tgetl(gp);

                unlock_user(*q, addr, 0);

            }

            for (gp = guest_envp, q = envp; *q;

                  gp += sizeof(target_ulong), q++) {

                addr = tgetl(gp);

                unlock_user(*q, addr, 0);

            }

        }

        break;

    case TARGET_NR_chdir:

        p = lock_user_string(arg1);

        ret = get_errno(chdir(p));

        unlock_user(p, arg1, 0);

        break;

#ifdef TARGET_NR_time

    case TARGET_NR_time:

        {

            time_t host_time;

            ret = get_errno(time(&host_time));

            if (!is_error(ret) && arg1)

                tputl(arg1, host_time);

        }

        break;

#endif

    case TARGET_NR_mknod:

        p = lock_user_string(arg1);

        ret = get_errno(mknod(p, arg2, arg3));

        unlock_user(p, arg1, 0);

        break;

    case TARGET_NR_chmod:

        p = lock_user_string(arg1);

        ret = get_errno(chmod(p, arg2));

        unlock_user(p, arg1, 0);

        break;

#ifdef TARGET_NR_break

    case TARGET_NR_break:

        goto unimplemented;

#endif

#ifdef TARGET_NR_oldstat

    case TARGET_NR_oldstat:

        goto unimplemented;

#endif

    case TARGET_NR_lseek:

        ret = get_errno(lseek(arg1, arg2, arg3));

        break;

#ifdef TARGET_NR_getxpid

    case TARGET_NR_getxpid:

#else

    case TARGET_NR_getpid:

#endif

        ret = get_errno(getpid());

        break;

    case TARGET_NR_mount:

		{

			/* need to look at the data field */

			void *p2, *p3;

			p = lock_user_string(arg1);

			p2 = lock_user_string(arg2);

			p3 = lock_user_string(arg3);

			ret = get_errno(mount(p, p2, p3, (unsigned long)arg4, (const void *)arg5));

			unlock_user(p, arg1, 0);

			unlock_user(p2, arg2, 0);

			unlock_user(p3, arg3, 0);

			break;

		}

#ifdef TARGET_NR_umount

    case TARGET_NR_umount:

        p = lock_user_string(arg1);

        ret = get_errno(umount(p));

        unlock_user(p, arg1, 0);

        break;

#endif

#ifdef TARGET_NR_stime /* not on alpha */

    case TARGET_NR_stime:

        {

            time_t host_time;

            host_time = tgetl(arg1);

            ret = get_errno(stime(&host_time));

        }

        break;

#endif

    case TARGET_NR_ptrace:

        goto unimplemented;

#ifdef TARGET_NR_alarm /* not on alpha */

    case TARGET_NR_alarm:

        ret = alarm(arg1);

        break;

#endif

#ifdef TARGET_NR_oldfstat

    case TARGET_NR_oldfstat:

        goto unimplemented;

#endif

#ifdef TARGET_NR_pause /* not on alpha */

    case TARGET_NR_pause:

        ret = get_errno(pause());

        break;

#endif

#ifdef TARGET_NR_utime

    case TARGET_NR_utime:

        {

            struct utimbuf tbuf, *host_tbuf;

            struct target_utimbuf *target_tbuf;

            if (arg2) {

                lock_user_struct(target_tbuf, arg2, 1);

                tbuf.actime = tswapl(target_tbuf->actime);

                tbuf.modtime = tswapl(target_tbuf->modtime);

                unlock_user_struct(target_tbuf, arg2, 0);

                host_tbuf = &tbuf;

            } else {

                host_tbuf = NULL;

            }

            p = lock_user_string(arg1);

            ret = get_errno(utime(p, host_tbuf));

            unlock_user(p, arg1, 0);

        }

        break;

#endif

    case TARGET_NR_utimes:

        {

            struct timeval *tvp, tv[2];

            if (arg2) {

                target_to_host_timeval(&tv[0], arg2);

                target_to_host_timeval(&tv[1],

                    arg2 + sizeof (struct target_timeval));

                tvp = tv;

            } else {

                tvp = NULL;

            }

            p = lock_user_string(arg1);

            ret = get_errno(utimes(p, tvp));

            unlock_user(p, arg1, 0);

        }

        break;

#ifdef TARGET_NR_stty

    case TARGET_NR_stty:

        goto unimplemented;

#endif

#ifdef TARGET_NR_gtty

    case TARGET_NR_gtty:

        goto unimplemented;

#endif

    case TARGET_NR_access:

        p = lock_user_string(arg1);

        ret = get_errno(access(p, arg2));

        unlock_user(p, arg1, 0);

        break;

#ifdef TARGET_NR_nice /* not on alpha */

    case TARGET_NR_nice:

        ret = get_errno(nice(arg1));

        break;

#endif

#ifdef TARGET_NR_ftime

    case TARGET_NR_ftime:

        goto unimplemented;

#endif

    case TARGET_NR_sync:

        sync();

        ret = 0;

        break;

    case TARGET_NR_kill:

        ret = get_errno(kill(arg1, arg2));

        break;

    case TARGET_NR_rename:

        {

            void *p2;

            p = lock_user_string(arg1);

            p2 = lock_user_string(arg2);

            ret = get_errno(rename(p, p2));

            unlock_user(p2, arg2, 0);

            unlock_user(p, arg1, 0);

        }

        break;

    case TARGET_NR_mkdir:

        p = lock_user_string(arg1);

        ret = get_errno(mkdir(p, arg2));

        unlock_user(p, arg1, 0);

        break;

    case TARGET_NR_rmdir:

        p = lock_user_string(arg1);

        ret = get_errno(rmdir(p));

        unlock_user(p, arg1, 0);

        break;

    case TARGET_NR_dup:

        ret = get_errno(dup(arg1));

        break;

    case TARGET_NR_pipe:

        {

            int host_pipe[2];

            ret = get_errno(pipe(host_pipe));

            if (!is_error(ret)) {

#if defined(TARGET_MIPS)

                CPUMIPSState *env = (CPUMIPSState*)cpu_env;

		env->gpr[3][env->current_tc] = host_pipe[1];

		ret = host_pipe[0];

#else

                tput32(arg1, host_pipe[0]);

                tput32(arg1 + 4, host_pipe[1]);

#endif

            }

        }

        break;

    case TARGET_NR_times:

        {

            struct target_tms *tmsp;

            struct tms tms;

            ret = get_errno(times(&tms));

            if (arg1) {

                tmsp = lock_user(arg1, sizeof(struct target_tms), 0);

                tmsp->tms_utime = tswapl(host_to_target_clock_t(tms.tms_utime));

                tmsp->tms_stime = tswapl(host_to_target_clock_t(tms.tms_stime));

                tmsp->tms_cutime = tswapl(host_to_target_clock_t(tms.tms_cutime));

                tmsp->tms_cstime = tswapl(host_to_target_clock_t(tms.tms_cstime));

            }

            if (!is_error(ret))

                ret = host_to_target_clock_t(ret);

        }

        break;

#ifdef TARGET_NR_prof

    case TARGET_NR_prof:

        goto unimplemented;

#endif

#ifdef TARGET_NR_signal

    case TARGET_NR_signal:

        goto unimplemented;

#endif

    case TARGET_NR_acct:

        p = lock_user_string(arg1);

        ret = get_errno(acct(path(p)));

        unlock_user(p, arg1, 0);

        break;

#ifdef TARGET_NR_umount2 /* not on alpha */

    case TARGET_NR_umount2:

        p = lock_user_string(arg1);

        ret = get_errno(umount2(p, arg2));

        unlock_user(p, arg1, 0);

        break;

#endif

#ifdef TARGET_NR_lock

    case TARGET_NR_lock:

        goto unimplemented;

#endif

    case TARGET_NR_ioctl:

        ret = do_ioctl(arg1, arg2, arg3);

        break;

    case TARGET_NR_fcntl:

        ret = get_errno(do_fcntl(arg1, arg2, arg3));

        break;

#ifdef TARGET_NR_mpx

    case TARGET_NR_mpx:

        goto unimplemented;

#endif

    case TARGET_NR_setpgid:

        ret = get_errno(setpgid(arg1, arg2));

        break;

#ifdef TARGET_NR_ulimit

    case TARGET_NR_ulimit:

        goto unimplemented;

#endif

#ifdef TARGET_NR_oldolduname

    case TARGET_NR_oldolduname:

        goto unimplemented;

#endif

    case TARGET_NR_umask:

        ret = get_errno(umask(arg1));

        break;

    case TARGET_NR_chroot:

        p = lock_user_string(arg1);

        ret = get_errno(chroot(p));

        unlock_user(p, arg1, 0);

        break;

    case TARGET_NR_ustat:

        goto unimplemented;

    case TARGET_NR_dup2:

        ret = get_errno(dup2(arg1, arg2));

        break;

#ifdef TARGET_NR_getppid /* not on alpha */

    case TARGET_NR_getppid:

        ret = get_errno(getppid());

        break;

#endif

    case TARGET_NR_getpgrp:

        ret = get_errno(getpgrp());

        break;

    case TARGET_NR_setsid:

        ret = get_errno(setsid());

        break;

#ifdef TARGET_NR_sigaction

    case TARGET_NR_sigaction:

        {

#if !defined(TARGET_MIPS)

            struct target_old_sigaction *old_act;

            struct target_sigaction act, oact, *pact;

            if (arg2) {

                lock_user_struct(old_act, arg2, 1);

                act._sa_handler = old_act->_sa_handler;

                target_siginitset(&act.sa_mask, old_act->sa_mask);

                act.sa_flags = old_act->sa_flags;

                act.sa_restorer = old_act->sa_restorer;

                unlock_user_struct(old_act, arg2, 0);

                pact = &act;

            } else {

                pact = NULL;

            }

            ret = get_errno(do_sigaction(arg1, pact, &oact));

            if (!is_error(ret) && arg3) {

                lock_user_struct(old_act, arg3, 0);

                old_act->_sa_handler = oact._sa_handler;

                old_act->sa_mask = oact.sa_mask.sig[0];

                old_act->sa_flags = oact.sa_flags;

                old_act->sa_restorer = oact.sa_restorer;

                unlock_user_struct(old_act, arg3, 1);

            }

#else

	    struct target_sigaction act, oact, *pact, *old_act;



	    if (arg2) {

		lock_user_struct(old_act, arg2, 1);

		act._sa_handler = old_act->_sa_handler;

		target_siginitset(&act.sa_mask, old_act->sa_mask.sig[0]);

		act.sa_flags = old_act->sa_flags;

		unlock_user_struct(old_act, arg2, 0);

		pact = &act;

	    } else {

		pact = NULL;

	    }



	    ret = get_errno(do_sigaction(arg1, pact, &oact));



	    if (!is_error(ret) && arg3) {

		lock_user_struct(old_act, arg3, 0);

		old_act->_sa_handler = oact._sa_handler;

		old_act->sa_flags = oact.sa_flags;

		old_act->sa_mask.sig[0] = oact.sa_mask.sig[0];

		old_act->sa_mask.sig[1] = 0;

		old_act->sa_mask.sig[2] = 0;

		old_act->sa_mask.sig[3] = 0;

		unlock_user_struct(old_act, arg3, 1);

	    }

#endif

        }

        break;

#endif

    case TARGET_NR_rt_sigaction:

        {

            struct target_sigaction *act;

            struct target_sigaction *oact;



            if (arg2)

                lock_user_struct(act, arg2, 1);

            else

                act = NULL;

            if (arg3)

                lock_user_struct(oact, arg3, 0);

            else

                oact = NULL;

            ret = get_errno(do_sigaction(arg1, act, oact));

            if (arg2)

                unlock_user_struct(act, arg2, 0);

            if (arg3)

                unlock_user_struct(oact, arg3, 1);

        }

        break;

#ifdef TARGET_NR_sgetmask /* not on alpha */

    case TARGET_NR_sgetmask:

        {

            sigset_t cur_set;

            target_ulong target_set;

            sigprocmask(0, NULL, &cur_set);

            host_to_target_old_sigset(&target_set, &cur_set);

            ret = target_set;

        }

        break;

#endif

#ifdef TARGET_NR_ssetmask /* not on alpha */

    case TARGET_NR_ssetmask:

        {

            sigset_t set, oset, cur_set;

            target_ulong target_set = arg1;

            sigprocmask(0, NULL, &cur_set);

            target_to_host_old_sigset(&set, &target_set);

            sigorset(&set, &set, &cur_set);

            sigprocmask(SIG_SETMASK, &set, &oset);

            host_to_target_old_sigset(&target_set, &oset);

            ret = target_set;

        }

        break;

#endif

#ifdef TARGET_NR_sigprocmask

    case TARGET_NR_sigprocmask:

        {

            int how = arg1;

            sigset_t set, oldset, *set_ptr;

           

            if (arg2) {

                switch(how) {

                case TARGET_SIG_BLOCK:

                    how = SIG_BLOCK;

                    break;

                case TARGET_SIG_UNBLOCK:

                    how = SIG_UNBLOCK;

                    break;

                case TARGET_SIG_SETMASK:

                    how = SIG_SETMASK;

                    break;

                default:

                    ret = -EINVAL;

                    goto fail;

                }

                p = lock_user(arg2, sizeof(target_sigset_t), 1);

                target_to_host_old_sigset(&set, p);

                unlock_user(p, arg2, 0);

                set_ptr = &set;

            } else {

                how = 0;

                set_ptr = NULL;

            }

            ret = get_errno(sigprocmask(arg1, set_ptr, &oldset));

            if (!is_error(ret) && arg3) {

                p = lock_user(arg3, sizeof(target_sigset_t), 0);

                host_to_target_old_sigset(p, &oldset);

                unlock_user(p, arg3, sizeof(target_sigset_t));

            }

        }

        break;

#endif

    case TARGET_NR_rt_sigprocmask:

        {

            int how = arg1;

            sigset_t set, oldset, *set_ptr;

           

            if (arg2) {

                switch(how) {

                case TARGET_SIG_BLOCK:

                    how = SIG_BLOCK;

                    break;

                case TARGET_SIG_UNBLOCK:

                    how = SIG_UNBLOCK;

                    break;

                case TARGET_SIG_SETMASK:

                    how = SIG_SETMASK;

                    break;

                default:

                    ret = -EINVAL;

                    goto fail;

                }

                p = lock_user(arg2, sizeof(target_sigset_t), 1);

                target_to_host_sigset(&set, p);

                unlock_user(p, arg2, 0);

                set_ptr = &set;

            } else {

                how = 0;

                set_ptr = NULL;

            }

            ret = get_errno(sigprocmask(how, set_ptr, &oldset));

            if (!is_error(ret) && arg3) {

                p = lock_user(arg3, sizeof(target_sigset_t), 0);

                host_to_target_sigset(p, &oldset);

                unlock_user(p, arg3, sizeof(target_sigset_t));

            }

        }

        break;

#ifdef TARGET_NR_sigpending

    case TARGET_NR_sigpending:

        {

            sigset_t set;

            ret = get_errno(sigpending(&set));

            if (!is_error(ret)) {

                p = lock_user(arg1, sizeof(target_sigset_t), 0);

                host_to_target_old_sigset(p, &set);

                unlock_user(p, arg1, sizeof(target_sigset_t));

            }

        }

        break;

#endif

    case TARGET_NR_rt_sigpending:

        {

            sigset_t set;

            ret = get_errno(sigpending(&set));

            if (!is_error(ret)) {

                p = lock_user(arg1, sizeof(target_sigset_t), 0);

                host_to_target_sigset(p, &set);

                unlock_user(p, arg1, sizeof(target_sigset_t));

            }

        }

        break;

#ifdef TARGET_NR_sigsuspend

    case TARGET_NR_sigsuspend:

        {

            sigset_t set;

            p = lock_user(arg1, sizeof(target_sigset_t), 1);

            target_to_host_old_sigset(&set, p);

            unlock_user(p, arg1, 0);

            ret = get_errno(sigsuspend(&set));

        }

        break;

#endif

    case TARGET_NR_rt_sigsuspend:

        {

            sigset_t set;

            p = lock_user(arg1, sizeof(target_sigset_t), 1);

            target_to_host_sigset(&set, p);

            unlock_user(p, arg1, 0);

            ret = get_errno(sigsuspend(&set));

        }

        break;

    case TARGET_NR_rt_sigtimedwait:

        {

            sigset_t set;

            struct timespec uts, *puts;

            siginfo_t uinfo;

           

            p = lock_user(arg1, sizeof(target_sigset_t), 1);

            target_to_host_sigset(&set, p);

            unlock_user(p, arg1, 0);

            if (arg3) {

                puts = &uts;

                target_to_host_timespec(puts, arg3);

            } else {

                puts = NULL;

            }

            ret = get_errno(sigtimedwait(&set, &uinfo, puts));

            if (!is_error(ret) && arg2) {

                p = lock_user(arg2, sizeof(target_sigset_t), 0);

                host_to_target_siginfo(p, &uinfo);

                unlock_user(p, arg2, sizeof(target_sigset_t));

            }

        }

        break;

    case TARGET_NR_rt_sigqueueinfo:

        {

            siginfo_t uinfo;

            p = lock_user(arg3, sizeof(target_sigset_t), 1);

            target_to_host_siginfo(&uinfo, p);

            unlock_user(p, arg1, 0);

            ret = get_errno(sys_rt_sigqueueinfo(arg1, arg2, &uinfo));

        }

        break;

#ifdef TARGET_NR_sigreturn

    case TARGET_NR_sigreturn:

        /* NOTE: ret is eax, so not transcoding must be done */

        ret = do_sigreturn(cpu_env);

        break;

#endif

    case TARGET_NR_rt_sigreturn:

        /* NOTE: ret is eax, so not transcoding must be done */

        ret = do_rt_sigreturn(cpu_env);

        break;

    case TARGET_NR_sethostname:

        p = lock_user_string(arg1);

        ret = get_errno(sethostname(p, arg2));

        unlock_user(p, arg1, 0);

        break;

    case TARGET_NR_setrlimit:

        {

            /* XXX: convert resource ? */

            int resource = arg1;

            struct target_rlimit *target_rlim;

            struct rlimit rlim;

            lock_user_struct(target_rlim, arg2, 1);

            rlim.rlim_cur = tswapl(target_rlim->rlim_cur);

            rlim.rlim_max = tswapl(target_rlim->rlim_max);

            unlock_user_struct(target_rlim, arg2, 0);

            ret = get_errno(setrlimit(resource, &rlim));

        }

        break;

    case TARGET_NR_getrlimit:

        {

            /* XXX: convert resource ? */

            int resource = arg1;

            struct target_rlimit *target_rlim;

            struct rlimit rlim;

           

            ret = get_errno(getrlimit(resource, &rlim));

            if (!is_error(ret)) {

                lock_user_struct(target_rlim, arg2, 0);

                rlim.rlim_cur = tswapl(target_rlim->rlim_cur);

                rlim.rlim_max = tswapl(target_rlim->rlim_max);

                unlock_user_struct(target_rlim, arg2, 1);

            }

        }

        break;

    case TARGET_NR_getrusage:

        {

            struct rusage rusage;

            ret = get_errno(getrusage(arg1, &rusage));

            if (!is_error(ret)) {

                host_to_target_rusage(arg2, &rusage);

            }

        }

        break;

    case TARGET_NR_gettimeofday:

        {

            struct timeval tv;

            ret = get_errno(gettimeofday(&tv, NULL));

            if (!is_error(ret)) {

                host_to_target_timeval(arg1, &tv);

            }

        }

        break;

    case TARGET_NR_settimeofday:

        {

            struct timeval tv;

            target_to_host_timeval(&tv, arg1);

            ret = get_errno(settimeofday(&tv, NULL));

        }

        break;

#ifdef TARGET_NR_select

    case TARGET_NR_select:

        {

            struct target_sel_arg_struct *sel;

            target_ulong inp, outp, exp, tvp;

            long nsel;



            lock_user_struct(sel, arg1, 1);

            nsel = tswapl(sel->n);

            inp = tswapl(sel->inp);

            outp = tswapl(sel->outp);

            exp = tswapl(sel->exp);

            tvp = tswapl(sel->tvp);

            unlock_user_struct(sel, arg1, 0);

            ret = do_select(nsel, inp, outp, exp, tvp);

        }

        break;

#endif

    case TARGET_NR_symlink:

        {

            void *p2;

            p = lock_user_string(arg1);

            p2 = lock_user_string(arg2);

            ret = get_errno(symlink(p, p2));

            unlock_user(p2, arg2, 0);

            unlock_user(p, arg1, 0);

        }

        break;

#ifdef TARGET_NR_oldlstat

    case TARGET_NR_oldlstat:

        goto unimplemented;

#endif

    case TARGET_NR_readlink:

        {

            void *p2;

            p = lock_user_string(arg1);

            p2 = lock_user(arg2, arg3, 0);

            ret = get_errno(readlink(path(p), p2, arg3));

            unlock_user(p2, arg2, ret);

            unlock_user(p, arg1, 0);

        }

        break;

#ifdef TARGET_NR_uselib

    case TARGET_NR_uselib:

        goto unimplemented;

#endif

#ifdef TARGET_NR_swapon

    case TARGET_NR_swapon:

        p = lock_user_string(arg1);

        ret = get_errno(swapon(p, arg2));

        unlock_user(p, arg1, 0);

        break;

#endif

    case TARGET_NR_reboot:

        goto unimplemented;

#ifdef TARGET_NR_readdir

    case TARGET_NR_readdir:

        goto unimplemented;

#endif

#ifdef TARGET_NR_mmap

    case TARGET_NR_mmap:

#if defined(TARGET_I386) || defined(TARGET_ARM) || defined(TARGET_M68K)

        {

            target_ulong *v;

            target_ulong v1, v2, v3, v4, v5, v6;

            v = lock_user(arg1, 6 * sizeof(target_ulong), 1);

            v1 = tswapl(v[0]);

            v2 = tswapl(v[1]);

            v3 = tswapl(v[2]);

            v4 = tswapl(v[3]);

            v5 = tswapl(v[4]);

            v6 = tswapl(v[5]);

            unlock_user(v, arg1, 0);

            ret = get_errno(target_mmap(v1, v2, v3,

                                        target_to_host_bitmask(v4, mmap_flags_tbl),

                                        v5, v6));

        }

#else

        ret = get_errno(target_mmap(arg1, arg2, arg3,

                                    target_to_host_bitmask(arg4, mmap_flags_tbl),

                                    arg5,

                                    arg6));

#endif

        break;

#endif

#ifdef TARGET_NR_mmap2

    case TARGET_NR_mmap2:

#if defined(TARGET_SPARC) || defined(TARGET_MIPS)

#define MMAP_SHIFT 12

#else

#define MMAP_SHIFT TARGET_PAGE_BITS

#endif

        ret = get_errno(target_mmap(arg1, arg2, arg3,

                                    target_to_host_bitmask(arg4, mmap_flags_tbl),

                                    arg5,

                                    arg6 << MMAP_SHIFT));

        break;

#endif

    case TARGET_NR_munmap:

        ret = get_errno(target_munmap(arg1, arg2));

        break;

    case TARGET_NR_mprotect:

        ret = get_errno(target_mprotect(arg1, arg2, arg3));

        break;

#ifdef TARGET_NR_mremap

    case TARGET_NR_mremap:

        ret = get_errno(target_mremap(arg1, arg2, arg3, arg4, arg5));

        break;

#endif

        /* ??? msync/mlock/munlock are broken for softmmu.  */

#ifdef TARGET_NR_msync

    case TARGET_NR_msync:

        ret = get_errno(msync(g2h(arg1), arg2, arg3));

        break;

#endif

#ifdef TARGET_NR_mlock

    case TARGET_NR_mlock:

        ret = get_errno(mlock(g2h(arg1), arg2));

        break;

#endif

#ifdef TARGET_NR_munlock

    case TARGET_NR_munlock:

        ret = get_errno(munlock(g2h(arg1), arg2));

        break;

#endif

#ifdef TARGET_NR_mlockall

    case TARGET_NR_mlockall:

        ret = get_errno(mlockall(arg1));

        break;

#endif

#ifdef TARGET_NR_munlockall

    case TARGET_NR_munlockall:

        ret = get_errno(munlockall());

        break;

#endif

    case TARGET_NR_truncate:

        p = lock_user_string(arg1);

        ret = get_errno(truncate(p, arg2));

        unlock_user(p, arg1, 0);

        break;

    case TARGET_NR_ftruncate:

        ret = get_errno(ftruncate(arg1, arg2));

        break;

    case TARGET_NR_fchmod:

        ret = get_errno(fchmod(arg1, arg2));

        break;

    case TARGET_NR_getpriority:

        ret = get_errno(getpriority(arg1, arg2));

        break;

    case TARGET_NR_setpriority:

        ret = get_errno(setpriority(arg1, arg2, arg3));

        break;

#ifdef TARGET_NR_profil

    case TARGET_NR_profil:

        goto unimplemented;

#endif

    case TARGET_NR_statfs:

        p = lock_user_string(arg1);

        ret = get_errno(statfs(path(p), &stfs));

        unlock_user(p, arg1, 0);

    convert_statfs:

        if (!is_error(ret)) {

            struct target_statfs *target_stfs;

           

            lock_user_struct(target_stfs, arg2, 0);

            /* ??? put_user is probably wrong.  */

            put_user(stfs.f_type, &target_stfs->f_type);

            put_user(stfs.f_bsize, &target_stfs->f_bsize);

            put_user(stfs.f_blocks, &target_stfs->f_blocks);

            put_user(stfs.f_bfree, &target_stfs->f_bfree);

            put_user(stfs.f_bavail, &target_stfs->f_bavail);

            put_user(stfs.f_files, &target_stfs->f_files);

            put_user(stfs.f_ffree, &target_stfs->f_ffree);

            put_user(stfs.f_fsid.__val[0], &target_stfs->f_fsid.val[0]);

            put_user(stfs.f_fsid.__val[1], &target_stfs->f_fsid.val[1]);

            put_user(stfs.f_namelen, &target_stfs->f_namelen);

            unlock_user_struct(target_stfs, arg2, 1);

        }

        break;

    case TARGET_NR_fstatfs:

        ret = get_errno(fstatfs(arg1, &stfs));

        goto convert_statfs;

#ifdef TARGET_NR_statfs64

    case TARGET_NR_statfs64:

        p = lock_user_string(arg1);

        ret = get_errno(statfs(path(p), &stfs));

        unlock_user(p, arg1, 0);

    convert_statfs64:

        if (!is_error(ret)) {

            struct target_statfs64 *target_stfs;

           

            lock_user_struct(target_stfs, arg3, 0);

            /* ??? put_user is probably wrong.  */

            put_user(stfs.f_type, &target_stfs->f_type);

            put_user(stfs.f_bsize, &target_stfs->f_bsize);

            put_user(stfs.f_blocks, &target_stfs->f_blocks);

            put_user(stfs.f_bfree, &target_stfs->f_bfree);

            put_user(stfs.f_bavail, &target_stfs->f_bavail);

            put_user(stfs.f_files, &target_stfs->f_files);

            put_user(stfs.f_ffree, &target_stfs->f_ffree);

            put_user(stfs.f_fsid.__val[0], &target_stfs->f_fsid.val[0]);

            put_user(stfs.f_fsid.__val[1], &target_stfs->f_fsid.val[1]);

            put_user(stfs.f_namelen, &target_stfs->f_namelen);

            unlock_user_struct(target_stfs, arg3, 0);

        }

        break;

    case TARGET_NR_fstatfs64:

        ret = get_errno(fstatfs(arg1, &stfs));

        goto convert_statfs64;

#endif

#ifdef TARGET_NR_ioperm

    case TARGET_NR_ioperm:

        goto unimplemented;

#endif

#ifdef TARGET_NR_socketcall

    case TARGET_NR_socketcall:

        ret = do_socketcall(arg1, arg2);

        break;

#endif

#ifdef TARGET_NR_accept

    case TARGET_NR_accept:

        ret = do_accept(arg1, arg2, arg3);

        break;

#endif

#ifdef TARGET_NR_bind

    case TARGET_NR_bind:

        ret = do_bind(arg1, arg2, arg3);

        break;

#endif

#ifdef TARGET_NR_connect

    case TARGET_NR_connect:

        ret = do_connect(arg1, arg2, arg3);

        break;

#endif

#ifdef TARGET_NR_getpeername

    case TARGET_NR_getpeername:

        ret = do_getpeername(arg1, arg2, arg3);

        break;

#endif

#ifdef TARGET_NR_getsockname

    case TARGET_NR_getsockname:

        ret = do_getsockname(arg1, arg2, arg3);

        break;

#endif

#ifdef TARGET_NR_getsockopt

    case TARGET_NR_getsockopt:

        ret = do_getsockopt(arg1, arg2, arg3, arg4, arg5);

        break;

#endif

#ifdef TARGET_NR_listen

    case TARGET_NR_listen:

        ret = get_errno(listen(arg1, arg2));

        break;

#endif

#ifdef TARGET_NR_recv

    case TARGET_NR_recv:

        ret = do_recvfrom(arg1, arg2, arg3, arg4, 0, 0);

        break;

#endif

#ifdef TARGET_NR_recvfrom

    case TARGET_NR_recvfrom:

        ret = do_recvfrom(arg1, arg2, arg3, arg4, arg5, arg6);

        break;

#endif

#ifdef TARGET_NR_recvmsg

    case TARGET_NR_recvmsg:

        ret = do_sendrecvmsg(arg1, arg2, arg3, 0);

        break;

#endif

#ifdef TARGET_NR_send

    case TARGET_NR_send:

        ret = do_sendto(arg1, arg2, arg3, arg4, 0, 0);

        break;

#endif

#ifdef TARGET_NR_sendmsg

    case TARGET_NR_sendmsg:

        ret = do_sendrecvmsg(arg1, arg2, arg3, 1);

        break;

#endif

#ifdef TARGET_NR_sendto

    case TARGET_NR_sendto:

        ret = do_sendto(arg1, arg2, arg3, arg4, arg5, arg6);

        break;

#endif

#ifdef TARGET_NR_shutdown

    case TARGET_NR_shutdown:

        ret = get_errno(shutdown(arg1, arg2));

        break;

#endif

#ifdef TARGET_NR_socket

    case TARGET_NR_socket:

        ret = do_socket(arg1, arg2, arg3);

        break;

#endif

#ifdef TARGET_NR_socketpair

    case TARGET_NR_socketpair:

        ret = do_socketpair(arg1, arg2, arg3, arg4);

        break;

#endif

#ifdef TARGET_NR_setsockopt

    case TARGET_NR_setsockopt:

        ret = do_setsockopt(arg1, arg2, arg3, arg4, (socklen_t) arg5);

        break;

#endif



    case TARGET_NR_syslog:

        p = lock_user_string(arg2);

        ret = get_errno(sys_syslog((int)arg1, p, (int)arg3));

        unlock_user(p, arg2, 0);

        break;



    case TARGET_NR_setitimer:

        {

            struct itimerval value, ovalue, *pvalue;



            if (arg2) {

                pvalue = &value;

                target_to_host_timeval(&pvalue->it_interval,

                                       arg2);

                target_to_host_timeval(&pvalue->it_value,

                                       arg2 + sizeof(struct target_timeval));

            } else {

                pvalue = NULL;

            }

            ret = get_errno(setitimer(arg1, pvalue, &ovalue));

            if (!is_error(ret) && arg3) {

                host_to_target_timeval(arg3,

                                       &ovalue.it_interval);

                host_to_target_timeval(arg3 + sizeof(struct target_timeval),

                                       &ovalue.it_value);

            }

        }

        break;

    case TARGET_NR_getitimer:

        {

            struct itimerval value;

           

            ret = get_errno(getitimer(arg1, &value));

            if (!is_error(ret) && arg2) {

                host_to_target_timeval(arg2,

                                       &value.it_interval);

                host_to_target_timeval(arg2 + sizeof(struct target_timeval),

                                       &value.it_value);

            }

        }

        break;

    case TARGET_NR_stat:

        p = lock_user_string(arg1);

        ret = get_errno(stat(path(p), &st));

        unlock_user(p, arg1, 0);

        goto do_stat;

    case TARGET_NR_lstat:

        p = lock_user_string(arg1);

        ret = get_errno(lstat(path(p), &st));

        unlock_user(p, arg1, 0);

        goto do_stat;

    case TARGET_NR_fstat:

        {

            ret = get_errno(fstat(arg1, &st));

        do_stat:

            if (!is_error(ret)) {

                struct target_stat *target_st;



                lock_user_struct(target_st, arg2, 0);

#if defined(TARGET_MIPS) || defined(TARGET_SPARC64)

                target_st->st_dev = tswap32(st.st_dev);

#else

                target_st->st_dev = tswap16(st.st_dev);

#endif

                target_st->st_ino = tswapl(st.st_ino);

#if defined(TARGET_PPC) || defined(TARGET_MIPS)

                target_st->st_mode = tswapl(st.st_mode); /* XXX: check this */

                target_st->st_uid = tswap32(st.st_uid);

                target_st->st_gid = tswap32(st.st_gid);

#elif defined(TARGET_SPARC64)

                target_st->st_mode = tswap32(st.st_mode);

                target_st->st_uid = tswap32(st.st_uid);

                target_st->st_gid = tswap32(st.st_gid);

#else

                target_st->st_mode = tswap16(st.st_mode);

                target_st->st_uid = tswap16(st.st_uid);

                target_st->st_gid = tswap16(st.st_gid);

#endif

#if defined(TARGET_MIPS)

		/* If this is the same on PPC, then just merge w/ the above ifdef */

                target_st->st_nlink = tswapl(st.st_nlink);

                target_st->st_rdev = tswapl(st.st_rdev);

#elif defined(TARGET_SPARC64)

                target_st->st_nlink = tswap32(st.st_nlink);

                target_st->st_rdev = tswap32(st.st_rdev);

#else

                target_st->st_nlink = tswap16(st.st_nlink);

                target_st->st_rdev = tswap16(st.st_rdev);

#endif

                target_st->st_size = tswapl(st.st_size);

                target_st->st_blksize = tswapl(st.st_blksize);

                target_st->st_blocks = tswapl(st.st_blocks);

                target_st->target_st_atime = tswapl(st.st_atime);

                target_st->target_st_mtime = tswapl(st.st_mtime);

                target_st->target_st_ctime = tswapl(st.st_ctime);

                unlock_user_struct(target_st, arg2, 1);

            }

        }

        break;

#ifdef TARGET_NR_olduname

    case TARGET_NR_olduname:

        goto unimplemented;

#endif

#ifdef TARGET_NR_iopl

    case TARGET_NR_iopl:

        goto unimplemented;

#endif

    case TARGET_NR_vhangup:

        ret = get_errno(vhangup());

        break;

#ifdef TARGET_NR_idle

    case TARGET_NR_idle:

        goto unimplemented;

#endif

#ifdef TARGET_NR_syscall

    case TARGET_NR_syscall:

    	ret = do_syscall(cpu_env,arg1 & 0xffff,arg2,arg3,arg4,arg5,arg6,0);

    	break;

#endif

    case TARGET_NR_wait4:

        {

            int status;

            target_long status_ptr = arg2;

            struct rusage rusage, *rusage_ptr;

            target_ulong target_rusage = arg4;

            if (target_rusage)

                rusage_ptr = &rusage;

            else

                rusage_ptr = NULL;

            ret = get_errno(wait4(arg1, &status, arg3, rusage_ptr));

            if (!is_error(ret)) {

                if (status_ptr)

                    tputl(status_ptr, status);

                if (target_rusage) {

                    host_to_target_rusage(target_rusage, &rusage);

                }

            }

        }

        break;

#ifdef TARGET_NR_swapoff

    case TARGET_NR_swapoff:

        p = lock_user_string(arg1);

        ret = get_errno(swapoff(p));

        unlock_user(p, arg1, 0);

        break;

#endif

    case TARGET_NR_sysinfo:

        {

            struct target_sysinfo *target_value;

            struct sysinfo value;

            ret = get_errno(sysinfo(&value));

            if (!is_error(ret) && arg1)

            {

                /* ??? __put_user is probably wrong.  */

                lock_user_struct(target_value, arg1, 0);

                __put_user(value.uptime, &target_value->uptime);

                __put_user(value.loads[0], &target_value->loads[0]);

                __put_user(value.loads[1], &target_value->loads[1]);

                __put_user(value.loads[2], &target_value->loads[2]);

                __put_user(value.totalram, &target_value->totalram);

                __put_user(value.freeram, &target_value->freeram);

                __put_user(value.sharedram, &target_value->sharedram);

                __put_user(value.bufferram, &target_value->bufferram);

                __put_user(value.totalswap, &target_value->totalswap);

                __put_user(value.freeswap, &target_value->freeswap);

                __put_user(value.procs, &target_value->procs);

                __put_user(value.totalhigh, &target_value->totalhigh);

                __put_user(value.freehigh, &target_value->freehigh);

                __put_user(value.mem_unit, &target_value->mem_unit);

                unlock_user_struct(target_value, arg1, 1);

            }

        }

        break;

#ifdef TARGET_NR_ipc

    case TARGET_NR_ipc:

	ret = do_ipc(arg1, arg2, arg3, arg4, arg5, arg6);

	break;

#endif

    case TARGET_NR_fsync:

        ret = get_errno(fsync(arg1));

        break;

    case TARGET_NR_clone:

        ret = get_errno(do_fork(cpu_env, arg1, arg2));

        break;

#ifdef __NR_exit_group

        /* new thread calls */

    case TARGET_NR_exit_group:

        gdb_exit(cpu_env, arg1);

        ret = get_errno(exit_group(arg1));

        break;

#endif

    case TARGET_NR_setdomainname:

        p = lock_user_string(arg1);

        ret = get_errno(setdomainname(p, arg2));

        unlock_user(p, arg1, 0);

        break;

    case TARGET_NR_uname:

        /* no need to transcode because we use the linux syscall */

        {

            struct new_utsname * buf;

   

            lock_user_struct(buf, arg1, 0);

            ret = get_errno(sys_uname(buf));

            if (!is_error(ret)) {

                /* Overrite the native machine name with whatever is being

                   emulated. */

                strcpy (buf->machine, UNAME_MACHINE);

                /* Allow the user to override the reported release.  */

                if (qemu_uname_release && *qemu_uname_release)

                  strcpy (buf->release, qemu_uname_release);

            }

            unlock_user_struct(buf, arg1, 1);

        }

        break;

#ifdef TARGET_I386

    case TARGET_NR_modify_ldt:

        ret = get_errno(do_modify_ldt(cpu_env, arg1, arg2, arg3));

        break;

#if !defined(TARGET_X86_64)

    case TARGET_NR_vm86old:

        goto unimplemented;

    case TARGET_NR_vm86:

        ret = do_vm86(cpu_env, arg1, arg2);

        break;

#endif

#endif

    case TARGET_NR_adjtimex:

        goto unimplemented;

#ifdef TARGET_NR_create_module

    case TARGET_NR_create_module:

#endif

    case TARGET_NR_init_module:

    case TARGET_NR_delete_module:

#ifdef TARGET_NR_get_kernel_syms

    case TARGET_NR_get_kernel_syms:

#endif

        goto unimplemented;

    case TARGET_NR_quotactl:

        goto unimplemented;

    case TARGET_NR_getpgid:

        ret = get_errno(getpgid(arg1));

        break;

    case TARGET_NR_fchdir:

        ret = get_errno(fchdir(arg1));

        break;

#ifdef TARGET_NR_bdflush /* not on x86_64 */

    case TARGET_NR_bdflush:

        goto unimplemented;

#endif

#ifdef TARGET_NR_sysfs

    case TARGET_NR_sysfs:

        goto unimplemented;

#endif

    case TARGET_NR_personality:

        ret = get_errno(personality(arg1));

        break;

#ifdef TARGET_NR_afs_syscall

    case TARGET_NR_afs_syscall:

        goto unimplemented;

#endif

#ifdef TARGET_NR__llseek /* Not on alpha */

    case TARGET_NR__llseek:

        {

#if defined (__x86_64__)

            ret = get_errno(lseek(arg1, ((uint64_t )arg2 << 32) | arg3, arg5));

            tput64(arg4, ret);

#else

            int64_t res;

            ret = get_errno(_llseek(arg1, arg2, arg3, &res, arg5));

            tput64(arg4, res);

#endif

        }

        break;

#endif

    case TARGET_NR_getdents:

#if TARGET_LONG_SIZE != 4

        goto unimplemented;

#warning not supported

#elif TARGET_LONG_SIZE == 4 && HOST_LONG_SIZE == 8

        {

            struct target_dirent *target_dirp;

            struct dirent *dirp;

            long count = arg3;



	    dirp = malloc(count);

	    if (!dirp)

                return -ENOMEM;

           

            ret = get_errno(sys_getdents(arg1, dirp, count));

            if (!is_error(ret)) {

                struct dirent *de;

		struct target_dirent *tde;

                int len = ret;

                int reclen, treclen;

		int count1, tnamelen;



		count1 = 0;

                de = dirp;

                target_dirp = lock_user(arg2, count, 0);

		tde = target_dirp;

                while (len > 0) {

                    reclen = de->d_reclen;

		    treclen = reclen - (2 * (sizeof(long) - sizeof(target_long)));

                    tde->d_reclen = tswap16(treclen);

                    tde->d_ino = tswapl(de->d_ino);

                    tde->d_off = tswapl(de->d_off);

		    tnamelen = treclen - (2 * sizeof(target_long) + 2);

		    if (tnamelen > 256)

                        tnamelen = 256;

                    /* XXX: may not be correct */

		    strncpy(tde->d_name, de->d_name, tnamelen);

                    de = (struct dirent *)((char *)de + reclen);

                    len -= reclen;

                    tde = (struct target_dirent *)((char *)tde + treclen);

		    count1 += treclen;

                }

		ret = count1;

            }

            unlock_user(target_dirp, arg2, ret);

	    free(dirp);

        }

#else

        {

            struct dirent *dirp;

            long count = arg3;



            dirp = lock_user(arg2, count, 0);

            ret = get_errno(sys_getdents(arg1, dirp, count));

            if (!is_error(ret)) {

                struct dirent *de;

                int len = ret;

                int reclen;

                de = dirp;

                while (len > 0) {

                    reclen = de->d_reclen;

                    if (reclen > len)

                        break;

                    de->d_reclen = tswap16(reclen);

                    tswapls(&de->d_ino);

                    tswapls(&de->d_off);

                    de = (struct dirent *)((char *)de + reclen);

                    len -= reclen;

                }

            }

            unlock_user(dirp, arg2, ret);

        }

#endif

        break;

#ifdef TARGET_NR_getdents64

    case TARGET_NR_getdents64:

        {

            struct dirent64 *dirp;

            long count = arg3;

            dirp = lock_user(arg2, count, 0);

            ret = get_errno(sys_getdents64(arg1, dirp, count));

            if (!is_error(ret)) {

                struct dirent64 *de;

                int len = ret;

                int reclen;

                de = dirp;

                while (len > 0) {

                    reclen = de->d_reclen;

                    if (reclen > len)

                        break;

                    de->d_reclen = tswap16(reclen);

                    tswap64s(&de->d_ino);

                    tswap64s(&de->d_off);

                    de = (struct dirent64 *)((char *)de + reclen);

                    len -= reclen;

                }

            }

            unlock_user(dirp, arg2, ret);

        }

        break;

#endif /* TARGET_NR_getdents64 */

#ifdef TARGET_NR__newselect

    case TARGET_NR__newselect:

        ret = do_select(arg1, arg2, arg3, arg4, arg5);

        break;

#endif

#ifdef TARGET_NR_poll

    case TARGET_NR_poll:

        {

            struct target_pollfd *target_pfd;

            unsigned int nfds = arg2;

            int timeout = arg3;

            struct pollfd *pfd;

            unsigned int i;



            target_pfd = lock_user(arg1, sizeof(struct target_pollfd) * nfds, 1);

            pfd = alloca(sizeof(struct pollfd) * nfds);

            for(i = 0; i < nfds; i++) {

                pfd[i].fd = tswap32(target_pfd[i].fd);

                pfd[i].events = tswap16(target_pfd[i].events);

            }

            ret = get_errno(poll(pfd, nfds, timeout));

            if (!is_error(ret)) {

                for(i = 0; i < nfds; i++) {

                    target_pfd[i].revents = tswap16(pfd[i].revents);

                }

                ret += nfds * (sizeof(struct target_pollfd)

                               - sizeof(struct pollfd));

            }

            unlock_user(target_pfd, arg1, ret);

        }

        break;

#endif

    case TARGET_NR_flock:

        /* NOTE: the flock constant seems to be the same for every

           Linux platform */

        ret = get_errno(flock(arg1, arg2));

        break;

    case TARGET_NR_readv:

        {

            int count = arg3;

            struct iovec *vec;



            vec = alloca(count * sizeof(struct iovec));

            lock_iovec(vec, arg2, count, 0);

            ret = get_errno(readv(arg1, vec, count));

            unlock_iovec(vec, arg2, count, 1);

        }

        break;

    case TARGET_NR_writev:

        {

            int count = arg3;

            struct iovec *vec;



            vec = alloca(count * sizeof(struct iovec));

            lock_iovec(vec, arg2, count, 1);

            ret = get_errno(writev(arg1, vec, count));

            unlock_iovec(vec, arg2, count, 0);

        }

        break;

    case TARGET_NR_getsid:

        ret = get_errno(getsid(arg1));

        break;

#if defined(TARGET_NR_fdatasync) /* Not on alpha (osf_datasync ?) */

    case TARGET_NR_fdatasync:

        ret = get_errno(fdatasync(arg1));

        break;

#endif

    case TARGET_NR__sysctl:

        /* We don't implement this, but ENODIR is always a safe

           return value. */

        return -ENOTDIR;

    case TARGET_NR_sched_setparam:

        {

            struct sched_param *target_schp;

            struct sched_param schp;



            lock_user_struct(target_schp, arg2, 1);

            schp.sched_priority = tswap32(target_schp->sched_priority);

            unlock_user_struct(target_schp, arg2, 0);

            ret = get_errno(sched_setparam(arg1, &schp));

        }

        break;

    case TARGET_NR_sched_getparam:

        {

            struct sched_param *target_schp;

            struct sched_param schp;

            ret = get_errno(sched_getparam(arg1, &schp));

            if (!is_error(ret)) {

                lock_user_struct(target_schp, arg2, 0);

                target_schp->sched_priority = tswap32(schp.sched_priority);

                unlock_user_struct(target_schp, arg2, 1);

            }

        }

        break;

    case TARGET_NR_sched_setscheduler:

        {

            struct sched_param *target_schp;

            struct sched_param schp;

            lock_user_struct(target_schp, arg3, 1);

            schp.sched_priority = tswap32(target_schp->sched_priority);

            unlock_user_struct(target_schp, arg3, 0);

            ret = get_errno(sched_setscheduler(arg1, arg2, &schp));

        }

        break;

    case TARGET_NR_sched_getscheduler:

        ret = get_errno(sched_getscheduler(arg1));

        break;

    case TARGET_NR_sched_yield:

        ret = get_errno(sched_yield());

        break;

    case TARGET_NR_sched_get_priority_max:

        ret = get_errno(sched_get_priority_max(arg1));

        break;

    case TARGET_NR_sched_get_priority_min:

        ret = get_errno(sched_get_priority_min(arg1));

        break;

    case TARGET_NR_sched_rr_get_interval:

        {

            struct timespec ts;

            ret = get_errno(sched_rr_get_interval(arg1, &ts));

            if (!is_error(ret)) {

                host_to_target_timespec(arg2, &ts);

            }

        }

        break;

    case TARGET_NR_nanosleep:

        {

            struct timespec req, rem;

            target_to_host_timespec(&req, arg1);

            ret = get_errno(nanosleep(&req, &rem));

            if (is_error(ret) && arg2) {

                host_to_target_timespec(arg2, &rem);

            }

        }

        break;

#ifdef TARGET_NR_query_module

    case TARGET_NR_query_module:

        goto unimplemented;

#endif

#ifdef TARGET_NR_nfsservctl

    case TARGET_NR_nfsservctl:

        goto unimplemented;

#endif

    case TARGET_NR_prctl:

        switch (arg1)

            {

            case PR_GET_PDEATHSIG:

                {

                    int deathsig;

                    ret = get_errno(prctl(arg1, &deathsig, arg3, arg4, arg5));

                    if (!is_error(ret) && arg2)

                        tput32(arg2, deathsig);

                }

                break;

            default:

                ret = get_errno(prctl(arg1, arg2, arg3, arg4, arg5));

                break;

            }

        break;

#ifdef TARGET_NR_pread

    case TARGET_NR_pread:

        page_unprotect_range(arg2, arg3);

        p = lock_user(arg2, arg3, 0);

        ret = get_errno(pread(arg1, p, arg3, arg4));

        unlock_user(p, arg2, ret);

        break;

    case TARGET_NR_pwrite:

        p = lock_user(arg2, arg3, 1);

        ret = get_errno(pwrite(arg1, p, arg3, arg4));

        unlock_user(p, arg2, 0);

        break;

#endif

    case TARGET_NR_getcwd:

        p = lock_user(arg1, arg2, 0);

        ret = get_errno(sys_getcwd1(p, arg2));

        unlock_user(p, arg1, ret);

        break;

    case TARGET_NR_capget:

        goto unimplemented;

    case TARGET_NR_capset:

        goto unimplemented;

    case TARGET_NR_sigaltstack:

        goto unimplemented;

    case TARGET_NR_sendfile:

        goto unimplemented;

#ifdef TARGET_NR_getpmsg

    case TARGET_NR_getpmsg:

        goto unimplemented;

#endif

#ifdef TARGET_NR_putpmsg

    case TARGET_NR_putpmsg:

        goto unimplemented;

#endif

#ifdef TARGET_NR_vfork

    case TARGET_NR_vfork:

        ret = get_errno(do_fork(cpu_env, CLONE_VFORK | CLONE_VM | SIGCHLD, 0));

        break;

#endif

#ifdef TARGET_NR_ugetrlimit

    case TARGET_NR_ugetrlimit:

    {

	struct rlimit rlim;

	ret = get_errno(getrlimit(arg1, &rlim));

	if (!is_error(ret)) {

	    struct target_rlimit *target_rlim;

            lock_user_struct(target_rlim, arg2, 0);

	    target_rlim->rlim_cur = tswapl(rlim.rlim_cur);

	    target_rlim->rlim_max = tswapl(rlim.rlim_max);

            unlock_user_struct(target_rlim, arg2, 1);

	}

	break;

    }

#endif

#ifdef TARGET_NR_truncate64

    case TARGET_NR_truncate64:

        p = lock_user_string(arg1);

	ret = target_truncate64(cpu_env, p, arg2, arg3, arg4);

        unlock_user(p, arg1, 0);

	break;

#endif

#ifdef TARGET_NR_ftruncate64

    case TARGET_NR_ftruncate64:

	ret = target_ftruncate64(cpu_env, arg1, arg2, arg3, arg4);

	break;

#endif

#ifdef TARGET_NR_stat64

    case TARGET_NR_stat64:

        p = lock_user_string(arg1);

        ret = get_errno(stat(path(p), &st));

        unlock_user(p, arg1, 0);

        goto do_stat64;

#endif

#ifdef TARGET_NR_lstat64

    case TARGET_NR_lstat64:

        p = lock_user_string(arg1);

        ret = get_errno(lstat(path(p), &st));

        unlock_user(p, arg1, 0);

        goto do_stat64;

#endif

#ifdef TARGET_NR_fstat64

    case TARGET_NR_fstat64:

        {

            ret = get_errno(fstat(arg1, &st));

        do_stat64:

            if (!is_error(ret)) {

#ifdef TARGET_ARM

                if (((CPUARMState *)cpu_env)->eabi) {

                    struct target_eabi_stat64 *target_st;

                    lock_user_struct(target_st, arg2, 1);

                    memset(target_st, 0, sizeof(struct target_eabi_stat64));

                    /* put_user is probably wrong.  */

                    put_user(st.st_dev, &target_st->st_dev);

                    put_user(st.st_ino, &target_st->st_ino);

#ifdef TARGET_STAT64_HAS_BROKEN_ST_INO

                    put_user(st.st_ino, &target_st->__st_ino);

#endif

                    put_user(st.st_mode, &target_st->st_mode);

                    put_user(st.st_nlink, &target_st->st_nlink);

                    put_user(st.st_uid, &target_st->st_uid);

                    put_user(st.st_gid, &target_st->st_gid);

                    put_user(st.st_rdev, &target_st->st_rdev);

                    /* XXX: better use of kernel struct */

                    put_user(st.st_size, &target_st->st_size);

                    put_user(st.st_blksize, &target_st->st_blksize);

                    put_user(st.st_blocks, &target_st->st_blocks);

                    put_user(st.st_atime, &target_st->target_st_atime);

                    put_user(st.st_mtime, &target_st->target_st_mtime);

                    put_user(st.st_ctime, &target_st->target_st_ctime);

                    unlock_user_struct(target_st, arg2, 0);

                } else

#endif

                {

                    struct target_stat64 *target_st;

                    lock_user_struct(target_st, arg2, 1);

                    memset(target_st, 0, sizeof(struct target_stat64));

                    /* ??? put_user is probably wrong.  */

                    put_user(st.st_dev, &target_st->st_dev);

                    put_user(st.st_ino, &target_st->st_ino);

#ifdef TARGET_STAT64_HAS_BROKEN_ST_INO

                    put_user(st.st_ino, &target_st->__st_ino);

#endif

                    put_user(st.st_mode, &target_st->st_mode);

                    put_user(st.st_nlink, &target_st->st_nlink);

                    put_user(st.st_uid, &target_st->st_uid);

                    put_user(st.st_gid, &target_st->st_gid);

                    put_user(st.st_rdev, &target_st->st_rdev);

                    /* XXX: better use of kernel struct */

                    put_user(st.st_size, &target_st->st_size);

                    put_user(st.st_blksize, &target_st->st_blksize);

                    put_user(st.st_blocks, &target_st->st_blocks);

                    put_user(st.st_atime, &target_st->target_st_atime);

                    put_user(st.st_mtime, &target_st->target_st_mtime);

                    put_user(st.st_ctime, &target_st->target_st_ctime);

                    unlock_user_struct(target_st, arg2, 0);

                }

            }

        }

        break;

#endif

#ifdef USE_UID16

    case TARGET_NR_lchown:

        p = lock_user_string(arg1);

        ret = get_errno(lchown(p, low2highuid(arg2), low2highgid(arg3)));

        unlock_user(p, arg1, 0);

        break;

    case TARGET_NR_getuid:

        ret = get_errno(high2lowuid(getuid()));

        break;

    case TARGET_NR_getgid:

        ret = get_errno(high2lowgid(getgid()));

        break;

    case TARGET_NR_geteuid:

        ret = get_errno(high2lowuid(geteuid()));

        break;

    case TARGET_NR_getegid:

        ret = get_errno(high2lowgid(getegid()));

        break;

    case TARGET_NR_setreuid:

        ret = get_errno(setreuid(low2highuid(arg1), low2highuid(arg2)));

        break;

    case TARGET_NR_setregid:

        ret = get_errno(setregid(low2highgid(arg1), low2highgid(arg2)));

        break;

    case TARGET_NR_getgroups:

        {

            int gidsetsize = arg1;

            uint16_t *target_grouplist;

            gid_t *grouplist;

            int i;



            grouplist = alloca(gidsetsize * sizeof(gid_t));

            ret = get_errno(getgroups(gidsetsize, grouplist));

            if (!is_error(ret)) {

                target_grouplist = lock_user(arg2, gidsetsize * 2, 0);

                for(i = 0;i < gidsetsize; i++)

                    target_grouplist[i] = tswap16(grouplist[i]);

                unlock_user(target_grouplist, arg2, gidsetsize * 2);

            }

        }

        break;

    case TARGET_NR_setgroups:

        {

            int gidsetsize = arg1;

            uint16_t *target_grouplist;

            gid_t *grouplist;

            int i;



            grouplist = alloca(gidsetsize * sizeof(gid_t));

            target_grouplist = lock_user(arg2, gidsetsize * 2, 1);

            for(i = 0;i < gidsetsize; i++)

                grouplist[i] = tswap16(target_grouplist[i]);

            unlock_user(target_grouplist, arg2, 0);

            ret = get_errno(setgroups(gidsetsize, grouplist));

        }

        break;

    case TARGET_NR_fchown:

        ret = get_errno(fchown(arg1, low2highuid(arg2), low2highgid(arg3)));

        break;

#ifdef TARGET_NR_setresuid

    case TARGET_NR_setresuid:

        ret = get_errno(setresuid(low2highuid(arg1),

                                  low2highuid(arg2),

                                  low2highuid(arg3)));

        break;

#endif

#ifdef TARGET_NR_getresuid

    case TARGET_NR_getresuid:

        {

            uid_t ruid, euid, suid;

            ret = get_errno(getresuid(&ruid, &euid, &suid));

            if (!is_error(ret)) {

                tput16(arg1, tswap16(high2lowuid(ruid)));

                tput16(arg2, tswap16(high2lowuid(euid)));

                tput16(arg3, tswap16(high2lowuid(suid)));

            }

        }

        break;

#endif

#ifdef TARGET_NR_getresgid

    case TARGET_NR_setresgid:

        ret = get_errno(setresgid(low2highgid(arg1),

                                  low2highgid(arg2),

                                  low2highgid(arg3)));

        break;

#endif

#ifdef TARGET_NR_getresgid

    case TARGET_NR_getresgid:

        {

            gid_t rgid, egid, sgid;

            ret = get_errno(getresgid(&rgid, &egid, &sgid));

            if (!is_error(ret)) {

                tput16(arg1, tswap16(high2lowgid(rgid)));

                tput16(arg2, tswap16(high2lowgid(egid)));

                tput16(arg3, tswap16(high2lowgid(sgid)));

            }

        }

        break;

#endif

    case TARGET_NR_chown:

        p = lock_user_string(arg1);

        ret = get_errno(chown(p, low2highuid(arg2), low2highgid(arg3)));

        unlock_user(p, arg1, 0);

        break;

    case TARGET_NR_setuid:

        ret = get_errno(setuid(low2highuid(arg1)));

        break;

    case TARGET_NR_setgid:

        ret = get_errno(setgid(low2highgid(arg1)));

        break;

    case TARGET_NR_setfsuid:

        ret = get_errno(setfsuid(arg1));

        break;

    case TARGET_NR_setfsgid:

        ret = get_errno(setfsgid(arg1));

        break;

#endif /* USE_UID16 */



#ifdef TARGET_NR_lchown32

    case TARGET_NR_lchown32:

        p = lock_user_string(arg1);

        ret = get_errno(lchown(p, arg2, arg3));

        unlock_user(p, arg1, 0);

        break;

#endif

#ifdef TARGET_NR_getuid32

    case TARGET_NR_getuid32:

        ret = get_errno(getuid());

        break;

#endif

#ifdef TARGET_NR_getgid32

    case TARGET_NR_getgid32:

        ret = get_errno(getgid());

        break;

#endif

#ifdef TARGET_NR_geteuid32

    case TARGET_NR_geteuid32:

        ret = get_errno(geteuid());

        break;

#endif

#ifdef TARGET_NR_getegid32

    case TARGET_NR_getegid32:

        ret = get_errno(getegid());

        break;

#endif

#ifdef TARGET_NR_setreuid32

    case TARGET_NR_setreuid32:

        ret = get_errno(setreuid(arg1, arg2));

        break;

#endif

#ifdef TARGET_NR_setregid32

    case TARGET_NR_setregid32:

        ret = get_errno(setregid(arg1, arg2));

        break;

#endif

#ifdef TARGET_NR_getgroups32

    case TARGET_NR_getgroups32:

        {

            int gidsetsize = arg1;

            uint32_t *target_grouplist;

            gid_t *grouplist;

            int i;



            grouplist = alloca(gidsetsize * sizeof(gid_t));

            ret = get_errno(getgroups(gidsetsize, grouplist));

            if (!is_error(ret)) {

                target_grouplist = lock_user(arg2, gidsetsize * 4, 0);

                for(i = 0;i < gidsetsize; i++)

                    target_grouplist[i] = tswap32(grouplist[i]);

                unlock_user(target_grouplist, arg2, gidsetsize * 4);

            }

        }

        break;

#endif

#ifdef TARGET_NR_setgroups32

    case TARGET_NR_setgroups32:

        {

            int gidsetsize = arg1;

            uint32_t *target_grouplist;

            gid_t *grouplist;

            int i;

           

            grouplist = alloca(gidsetsize * sizeof(gid_t));

            target_grouplist = lock_user(arg2, gidsetsize * 4, 1);

            for(i = 0;i < gidsetsize; i++)

                grouplist[i] = tswap32(target_grouplist[i]);

            unlock_user(target_grouplist, arg2, 0);

            ret = get_errno(setgroups(gidsetsize, grouplist));

        }

        break;

#endif

#ifdef TARGET_NR_fchown32

    case TARGET_NR_fchown32:

        ret = get_errno(fchown(arg1, arg2, arg3));

        break;

#endif

#ifdef TARGET_NR_setresuid32

    case TARGET_NR_setresuid32:

        ret = get_errno(setresuid(arg1, arg2, arg3));

        break;

#endif

#ifdef TARGET_NR_getresuid32

    case TARGET_NR_getresuid32:

        {

            uid_t ruid, euid, suid;

            ret = get_errno(getresuid(&ruid, &euid, &suid));

            if (!is_error(ret)) {

                tput32(arg1, tswap32(ruid));

                tput32(arg2, tswap32(euid));

                tput32(arg3, tswap32(suid));

            }

        }

        break;

#endif

#ifdef TARGET_NR_setresgid32

    case TARGET_NR_setresgid32:

        ret = get_errno(setresgid(arg1, arg2, arg3));

        break;

#endif

#ifdef TARGET_NR_getresgid32

    case TARGET_NR_getresgid32:

        {

            gid_t rgid, egid, sgid;

            ret = get_errno(getresgid(&rgid, &egid, &sgid));

            if (!is_error(ret)) {

                tput32(arg1, tswap32(rgid));

                tput32(arg2, tswap32(egid));

                tput32(arg3, tswap32(sgid));

            }

        }

        break;

#endif

#ifdef TARGET_NR_chown32

    case TARGET_NR_chown32:

        p = lock_user_string(arg1);

        ret = get_errno(chown(p, arg2, arg3));

        unlock_user(p, arg1, 0);

        break;

#endif

#ifdef TARGET_NR_setuid32

    case TARGET_NR_setuid32:

        ret = get_errno(setuid(arg1));

        break;

#endif

#ifdef TARGET_NR_setgid32

    case TARGET_NR_setgid32:

        ret = get_errno(setgid(arg1));

        break;

#endif

#ifdef TARGET_NR_setfsuid32

    case TARGET_NR_setfsuid32:

        ret = get_errno(setfsuid(arg1));

        break;

#endif

#ifdef TARGET_NR_setfsgid32

    case TARGET_NR_setfsgid32:

        ret = get_errno(setfsgid(arg1));

        break;

#endif



    case TARGET_NR_pivot_root:

        goto unimplemented;

#ifdef TARGET_NR_mincore

    case TARGET_NR_mincore:

        goto unimplemented;

#endif

#ifdef TARGET_NR_madvise

    case TARGET_NR_madvise:

        /* A straight passthrough may not be safe because qemu sometimes

           turns private flie-backed mappings into anonymous mappings.

           This will break MADV_DONTNEED.

           This is a hint, so ignoring and returning success is ok.  */

        ret = get_errno(0);

        break;

#endif

#if TARGET_LONG_BITS == 32

    case TARGET_NR_fcntl64:

    {

	int cmd;

	struct flock64 fl;

	struct target_flock64 *target_fl;

#ifdef TARGET_ARM

	struct target_eabi_flock64 *target_efl;

#endif



        switch(arg2){

        case TARGET_F_GETLK64:

            cmd = F_GETLK64;

            break;

        case TARGET_F_SETLK64:

            cmd = F_SETLK64;

            break;

        case TARGET_F_SETLKW64:

            cmd = F_SETLK64;

            break;

        default:

            cmd = arg2;

            break;

        }



        switch(arg2) {

        case TARGET_F_GETLK64:

#ifdef TARGET_ARM

            if (((CPUARMState *)cpu_env)->eabi) {

                lock_user_struct(target_efl, arg3, 1);

                fl.l_type = tswap16(target_efl->l_type);

                fl.l_whence = tswap16(target_efl->l_whence);

                fl.l_start = tswap64(target_efl->l_start);

                fl.l_len = tswap64(target_efl->l_len);

                fl.l_pid = tswapl(target_efl->l_pid);

                unlock_user_struct(target_efl, arg3, 0);

            } else

#endif

            {

                lock_user_struct(target_fl, arg3, 1);

                fl.l_type = tswap16(target_fl->l_type);

                fl.l_whence = tswap16(target_fl->l_whence);

                fl.l_start = tswap64(target_fl->l_start);

                fl.l_len = tswap64(target_fl->l_len);

                fl.l_pid = tswapl(target_fl->l_pid);

                unlock_user_struct(target_fl, arg3, 0);

            }

            ret = get_errno(fcntl(arg1, cmd, &fl));

	    if (ret == 0) {

#ifdef TARGET_ARM

                if (((CPUARMState *)cpu_env)->eabi) {

                    lock_user_struct(target_efl, arg3, 0);

                    target_efl->l_type = tswap16(fl.l_type);

                    target_efl->l_whence = tswap16(fl.l_whence);

                    target_efl->l_start = tswap64(fl.l_start);

                    target_efl->l_len = tswap64(fl.l_len);

                    target_efl->l_pid = tswapl(fl.l_pid);

                    unlock_user_struct(target_efl, arg3, 1);

                } else

#endif

                {

                    lock_user_struct(target_fl, arg3, 0);

                    target_fl->l_type = tswap16(fl.l_type);

                    target_fl->l_whence = tswap16(fl.l_whence);

                    target_fl->l_start = tswap64(fl.l_start);

                    target_fl->l_len = tswap64(fl.l_len);

                    target_fl->l_pid = tswapl(fl.l_pid);

                    unlock_user_struct(target_fl, arg3, 1);

                }

	    }

	    break;



        case TARGET_F_SETLK64:

        case TARGET_F_SETLKW64:

#ifdef TARGET_ARM

            if (((CPUARMState *)cpu_env)->eabi) {

                lock_user_struct(target_efl, arg3, 1);

                fl.l_type = tswap16(target_efl->l_type);

                fl.l_whence = tswap16(target_efl->l_whence);

                fl.l_start = tswap64(target_efl->l_start);

                fl.l_len = tswap64(target_efl->l_len);

                fl.l_pid = tswapl(target_efl->l_pid);

                unlock_user_struct(target_efl, arg3, 0);

            } else

#endif

            {

                lock_user_struct(target_fl, arg3, 1);

                fl.l_type = tswap16(target_fl->l_type);

                fl.l_whence = tswap16(target_fl->l_whence);

                fl.l_start = tswap64(target_fl->l_start);

                fl.l_len = tswap64(target_fl->l_len);

                fl.l_pid = tswapl(target_fl->l_pid);

                unlock_user_struct(target_fl, arg3, 0);

            }

            ret = get_errno(fcntl(arg1, cmd, &fl));

	    break;

        default:

            ret = get_errno(do_fcntl(arg1, cmd, arg3));

            break;

        }

	break;

    }

#endif

#ifdef TARGET_NR_cacheflush

    case TARGET_NR_cacheflush:

        /* self-modifying code is handled automatically, so nothing needed */

        ret = 0;

        break;

#endif

#ifdef TARGET_NR_security

    case TARGET_NR_security:

        goto unimplemented;

#endif

#ifdef TARGET_NR_getpagesize

    case TARGET_NR_getpagesize:

        ret = TARGET_PAGE_SIZE;

        break;

#endif

    case TARGET_NR_gettid:

        ret = get_errno(gettid());

        break;

#ifdef TARGET_NR_readahead

    case TARGET_NR_readahead:

        goto unimplemented;

#endif

#ifdef TARGET_NR_setxattr

    case TARGET_NR_setxattr:

    case TARGET_NR_lsetxattr:

    case TARGET_NR_fsetxattr:

    case TARGET_NR_getxattr:

    case TARGET_NR_lgetxattr:

    case TARGET_NR_fgetxattr:

    case TARGET_NR_listxattr:

    case TARGET_NR_llistxattr:

    case TARGET_NR_flistxattr:

    case TARGET_NR_removexattr:

    case TARGET_NR_lremovexattr:

    case TARGET_NR_fremovexattr:

        goto unimplemented_nowarn;

#endif

#ifdef TARGET_NR_set_thread_area

    case TARGET_NR_set_thread_area:

#ifdef TARGET_MIPS

      ((CPUMIPSState *) cpu_env)->tls_value = arg1;

      ret = 0;

      break;

#else

      goto unimplemented_nowarn;

#endif

#endif

#ifdef TARGET_NR_get_thread_area

    case TARGET_NR_get_thread_area:

        goto unimplemented_nowarn;

#endif

#ifdef TARGET_NR_getdomainname

    case TARGET_NR_getdomainname:

        goto unimplemented_nowarn;

#endif



#ifdef TARGET_NR_clock_gettime

    case TARGET_NR_clock_gettime:

    {

        struct timespec ts;

        ret = get_errno(clock_gettime(arg1, &ts));

        if (!is_error(ret)) {

            host_to_target_timespec(arg2, &ts);

        }

        break;

    }

#endif

#ifdef TARGET_NR_clock_getres

    case TARGET_NR_clock_getres:

    {

        struct timespec ts;

        ret = get_errno(clock_getres(arg1, &ts));

        if (!is_error(ret)) {

            host_to_target_timespec(arg2, &ts);

        }

        break;

    }

#endif



#if defined(TARGET_NR_set_tid_address) && defined(__NR_set_tid_address)

    case TARGET_NR_set_tid_address:

      ret = get_errno(set_tid_address((int *) arg1));

      break;

#endif



#ifdef TARGET_NR_tkill

    case TARGET_NR_tkill:

        ret = get_errno(sys_tkill((int)arg1, (int)arg2));

        break;

#endif



#ifdef TARGET_NR_tgkill

    case TARGET_NR_tgkill:

	ret = get_errno(sys_tgkill((int)arg1, (int)arg2, (int)arg3));

	break;

#endif



#ifdef TARGET_NR_set_robust_list

    case TARGET_NR_set_robust_list:

	goto unimplemented_nowarn;

#endif



    default:

    unimplemented:

        gemu_log("qemu: Unsupported syscall: %d\n", num);

#if defined(TARGET_NR_setxattr) || defined(TARGET_NR_get_thread_area) || defined(TARGET_NR_getdomainname) || defined(TARGET_NR_set_robust_list)

    unimplemented_nowarn:

#endif

        ret = -ENOSYS;

        break;

    }

 fail:

#ifdef DEBUG

    gemu_log(" = %ld\n", ret);

#endif

    return ret;

}
