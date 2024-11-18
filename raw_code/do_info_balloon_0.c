static void do_info_balloon(Monitor *mon, QObject **ret_data)

{

    ram_addr_t actual;



    actual = qemu_balloon_status();

    if (kvm_enabled() && !kvm_has_sync_mmu())

        qemu_error_new(QERR_KVM_MISSING_CAP, "synchronous MMU", "balloon");

    else if (actual == 0)

        qemu_error_new(QERR_DEVICE_NOT_ACTIVE, "balloon");

    else

        *ret_data = qobject_from_jsonf("{ 'balloon': %" PRId64 "}",

                                       (int64_t) actual);

}
