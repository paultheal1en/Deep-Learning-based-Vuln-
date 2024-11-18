int kvm_arch_remove_hw_breakpoint(target_ulong addr,

                                  target_ulong len, int type)

{

    int n;



    n = find_hw_breakpoint(addr, (type == GDB_BREAKPOINT_HW) ? 1 : len, type);

    if (n < 0)

        return -ENOENT;



    nb_hw_breakpoint--;

    hw_breakpoint[n] = hw_breakpoint[nb_hw_breakpoint];



    return 0;

}
