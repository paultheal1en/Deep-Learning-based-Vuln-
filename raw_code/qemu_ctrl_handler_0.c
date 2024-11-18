static BOOL WINAPI qemu_ctrl_handler(DWORD type)

{

    exit(STATUS_CONTROL_C_EXIT);

    return TRUE;

}
