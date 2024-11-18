bool cpu_exec_all(void)

{

    int r;



    /* Account partial waits to the vm_clock.  */

    qemu_clock_warp(vm_clock);



    if (next_cpu == NULL) {

        next_cpu = first_cpu;

    }

    for (; next_cpu != NULL && !exit_request; next_cpu = next_cpu->next_cpu) {

        CPUState *env = next_cpu;



        qemu_clock_enable(vm_clock,

                          (env->singlestep_enabled & SSTEP_NOTIMER) == 0);



#ifndef CONFIG_IOTHREAD

        if (qemu_alarm_pending()) {

            break;

        }

#endif

        if (cpu_can_run(env)) {

            if (kvm_enabled()) {

                r = kvm_cpu_exec(env);

                qemu_kvm_eat_signals(env);

            } else {

                r = tcg_cpu_exec(env);

            }

            if (r == EXCP_DEBUG) {

                cpu_handle_guest_debug(env);

                break;

            }

        } else if (env->stop || env->stopped) {

            break;

        }

    }

    exit_request = 0;

    return !all_cpu_threads_idle();

}
