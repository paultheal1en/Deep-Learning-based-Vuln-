void qemu_boot_set(const char *boot_order, Error **errp)

{

    Error *local_err = NULL;



    if (!boot_set_handler) {

        error_setg(errp, "no function defined to set boot device list for"

                         " this architecture");

        return;

    }



    validate_bootdevices(boot_order, &local_err);

    if (local_err) {

        error_propagate(errp, local_err);

        return;

    }



    if (boot_set_handler(boot_set_opaque, boot_order)) {

        error_setg(errp, "setting boot device list failed");

        return;

    }

}
