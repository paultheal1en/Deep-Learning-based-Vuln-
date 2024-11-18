int main_loop(void)

{

#ifndef _WIN32

    struct pollfd ufds[MAX_IO_HANDLERS + 1], *pf;

    IOHandlerRecord *ioh, *ioh_next;

    uint8_t buf[4096];

    int n, max_size;

#endif

    int ret, timeout;

    CPUState *env = global_env;



    for(;;) {

        if (vm_running) {

            ret = cpu_exec(env);

            if (shutdown_requested) {

                ret = EXCP_INTERRUPT; 

                break;

            }

            if (reset_requested) {

                reset_requested = 0;

                qemu_system_reset();

                ret = EXCP_INTERRUPT; 

            }

            if (ret == EXCP_DEBUG) {

                vm_stop(EXCP_DEBUG);

            }

            /* if hlt instruction, we wait until the next IRQ */

            /* XXX: use timeout computed from timers */

            if (ret == EXCP_HLT) 

                timeout = 10;

            else

                timeout = 0;

        } else {

            timeout = 10;

        }



#ifdef _WIN32

        if (timeout > 0)

            Sleep(timeout);

#else



        /* poll any events */

        /* XXX: separate device handlers from system ones */

        pf = ufds;

        for(ioh = first_io_handler; ioh != NULL; ioh = ioh->next) {

            if (!ioh->fd_can_read) {

                max_size = 0;

                pf->fd = ioh->fd;

                pf->events = POLLIN;

                ioh->ufd = pf;

                pf++;

            } else {

                max_size = ioh->fd_can_read(ioh->opaque);

                if (max_size > 0) {

                    if (max_size > sizeof(buf))

                        max_size = sizeof(buf);

                    pf->fd = ioh->fd;

                    pf->events = POLLIN;

                    ioh->ufd = pf;

                    pf++;

                } else {

                    ioh->ufd = NULL;

                }

            }

            ioh->max_size = max_size;

        }

        

        ret = poll(ufds, pf - ufds, timeout);

        if (ret > 0) {

            /* XXX: better handling of removal */

            for(ioh = first_io_handler; ioh != NULL; ioh = ioh_next) {

                ioh_next = ioh->next;

                pf = ioh->ufd;

                if (pf) {

                    if (pf->revents & POLLIN) {

                        if (ioh->max_size == 0) {

                            /* just a read event */

                            ioh->fd_read(ioh->opaque, NULL, 0);

                        } else {

                            n = read(ioh->fd, buf, ioh->max_size);

                            if (n >= 0) {

                                ioh->fd_read(ioh->opaque, buf, n);

                            } else if (errno != EAGAIN) {

                                ioh->fd_read(ioh->opaque, NULL, -errno);

                            }

                        }

                    }

                }

            }

        }



#if defined(CONFIG_SLIRP)

        /* XXX: merge with poll() */

        if (slirp_inited) {

            fd_set rfds, wfds, xfds;

            int nfds;

            struct timeval tv;



            nfds = -1;

            FD_ZERO(&rfds);

            FD_ZERO(&wfds);

            FD_ZERO(&xfds);

            slirp_select_fill(&nfds, &rfds, &wfds, &xfds);

            tv.tv_sec = 0;

            tv.tv_usec = 0;

            ret = select(nfds + 1, &rfds, &wfds, &xfds, &tv);

            if (ret >= 0) {

                slirp_select_poll(&rfds, &wfds, &xfds);

            }

        }

#endif



#endif



        if (vm_running) {

            qemu_run_timers(&active_timers[QEMU_TIMER_VIRTUAL], 

                            qemu_get_clock(vm_clock));

            

            if (audio_enabled) {

                /* XXX: add explicit timer */

                SB16_run();

            }

            

            /* run dma transfers, if any */

            DMA_run();

        }



        /* real time timers */

        qemu_run_timers(&active_timers[QEMU_TIMER_REALTIME], 

                        qemu_get_clock(rt_clock));

    }

    cpu_disable_ticks();

    return ret;

}
