void qemu_cond_destroy(QemuCond *cond)

{

    BOOL result;

    result = CloseHandle(cond->continue_event);

    if (!result) {

        error_exit(GetLastError(), __func__);

    }

    cond->continue_event = 0;

    result = CloseHandle(cond->sema);

    if (!result) {

        error_exit(GetLastError(), __func__);

    }

    cond->sema = 0;

}
