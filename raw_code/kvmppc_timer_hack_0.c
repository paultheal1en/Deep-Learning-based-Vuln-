static void kvmppc_timer_hack(void *opaque)

{

    qemu_service_io();

    qemu_mod_timer(kvmppc_timer, qemu_get_clock_ns(vm_clock) + kvmppc_timer_rate);

}
