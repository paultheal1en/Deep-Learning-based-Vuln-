static void virt_set_gic_version(Object *obj, const char *value, Error **errp)

{

    VirtMachineState *vms = VIRT_MACHINE(obj);



    if (!strcmp(value, "3")) {

        vms->gic_version = 3;

    } else if (!strcmp(value, "2")) {

        vms->gic_version = 2;

    } else if (!strcmp(value, "host")) {

        vms->gic_version = 0; /* Will probe later */

    } else {

        error_report("Invalid gic-version option value");

        error_printf("Allowed gic-version values are: 3, 2, host\n");

        exit(1);

    }

}
