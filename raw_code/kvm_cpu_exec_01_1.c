int kvm_cpu_exec(CPUState *env)

{

    struct kvm_run *run = env->kvm_run;

    int ret;



    dprintf("kvm_cpu_exec()\n");



    do {

#ifndef CONFIG_IOTHREAD

        if (env->exit_request) {

            dprintf("interrupt exit requested\n");

            ret = 0;

            break;

        }

#endif



        if (env->kvm_vcpu_dirty) {

            kvm_arch_put_registers(env);

            env->kvm_vcpu_dirty = 0;

        }



        kvm_arch_pre_run(env, run);

        qemu_mutex_unlock_iothread();

        ret = kvm_vcpu_ioctl(env, KVM_RUN, 0);

        qemu_mutex_lock_iothread();

        kvm_arch_post_run(env, run);



        if (ret == -EINTR || ret == -EAGAIN) {

            cpu_exit(env);

            dprintf("io window exit\n");

            ret = 0;

            break;

        }



        if (ret < 0) {

            dprintf("kvm run failed %s\n", strerror(-ret));

            abort();

        }



        kvm_flush_coalesced_mmio_buffer();



        ret = 0; /* exit loop */

        switch (run->exit_reason) {

        case KVM_EXIT_IO:

            dprintf("handle_io\n");

            ret = kvm_handle_io(run->io.port,

                                (uint8_t *)run + run->io.data_offset,

                                run->io.direction,

                                run->io.size,

                                run->io.count);

            break;

        case KVM_EXIT_MMIO:

            dprintf("handle_mmio\n");

            cpu_physical_memory_rw(run->mmio.phys_addr,

                                   run->mmio.data,

                                   run->mmio.len,

                                   run->mmio.is_write);

            ret = 1;

            break;

        case KVM_EXIT_IRQ_WINDOW_OPEN:

            dprintf("irq_window_open\n");

            break;

        case KVM_EXIT_SHUTDOWN:

            dprintf("shutdown\n");

            qemu_system_reset_request();

            ret = 1;

            break;

        case KVM_EXIT_UNKNOWN:

            dprintf("kvm_exit_unknown\n");

            break;

        case KVM_EXIT_FAIL_ENTRY:

            dprintf("kvm_exit_fail_entry\n");

            break;

        case KVM_EXIT_EXCEPTION:

            dprintf("kvm_exit_exception\n");

            break;

        case KVM_EXIT_DEBUG:

            dprintf("kvm_exit_debug\n");

#ifdef KVM_CAP_SET_GUEST_DEBUG

            if (kvm_arch_debug(&run->debug.arch)) {

                gdb_set_stop_cpu(env);

                vm_stop(EXCP_DEBUG);

                env->exception_index = EXCP_DEBUG;

                return 0;

            }

            /* re-enter, this exception was guest-internal */

            ret = 1;

#endif /* KVM_CAP_SET_GUEST_DEBUG */

            break;

        default:

            dprintf("kvm_arch_handle_exit\n");

            ret = kvm_arch_handle_exit(env, run);

            break;

        }

    } while (ret > 0);



    if (env->exit_request) {

        env->exit_request = 0;

        env->exception_index = EXCP_INTERRUPT;

    }



    return ret;

}
