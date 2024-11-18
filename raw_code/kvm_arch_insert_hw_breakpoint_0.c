int kvm_arch_insert_hw_breakpoint(target_ulong addr,

                                  target_ulong len, int type)

{

    switch (type) {

    case GDB_BREAKPOINT_HW:

        len = 1;

        break;

    case GDB_WATCHPOINT_WRITE:

    case GDB_WATCHPOINT_ACCESS:

        switch (len) {

        case 1:

            break;

        case 2:

        case 4:

        case 8:

            if (addr & (len - 1))

                return -EINVAL;

            break;

        default:

            return -EINVAL;

        }

        break;

    default:

        return -ENOSYS;

    }



    if (nb_hw_breakpoint == 4)

        return -ENOBUFS;



    if (find_hw_breakpoint(addr, len, type) >= 0)

        return -EEXIST;



    hw_breakpoint[nb_hw_breakpoint].addr = addr;

    hw_breakpoint[nb_hw_breakpoint].len = len;

    hw_breakpoint[nb_hw_breakpoint].type = type;

    nb_hw_breakpoint++;



    return 0;

}
