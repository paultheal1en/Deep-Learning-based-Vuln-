int qemu_savevm_state(QEMUFile *f)

{

    int saved_vm_running;

    int ret;



    saved_vm_running = vm_running;

    vm_stop(0);



    ret = qemu_savevm_state_begin(f);

    if (ret < 0)

        goto out;



    do {

        ret = qemu_savevm_state_iterate(f);

        if (ret < 0)

            goto out;

    } while (ret == 0);



    ret = qemu_savevm_state_complete(f);



out:

    if (saved_vm_running)

        vm_start();

    return ret;

}
