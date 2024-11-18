static void gdb_chr_event(void *opaque, int event)

{

    switch (event) {

    case CHR_EVENT_RESET:

        vm_stop(EXCP_INTERRUPT);

        gdb_syscall_state = opaque;

        gdb_has_xml = 0;

        break;

    default:

        break;

    }

}
