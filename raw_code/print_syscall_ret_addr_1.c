print_syscall_ret_addr(const struct syscallname *name, abi_long ret)

{

    char *errstr = NULL;



    if (ret == -1) {

        errstr = target_strerror(errno);

    }

    if ((ret == -1) && errstr) {

        gemu_log(" = -1 errno=%d (%s)\n", errno, errstr);

    } else {

        gemu_log(" = 0x" TARGET_ABI_FMT_lx "\n", ret);

    }

}
