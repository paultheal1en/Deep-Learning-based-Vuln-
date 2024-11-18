print_execve(const struct syscallname *name,

             abi_long arg1, abi_long arg2, abi_long arg3,

             abi_long arg4, abi_long arg5, abi_long arg6)

{

    abi_ulong arg_ptr_addr;

    char *s;



    if (!(s = lock_user_string(arg1)))

        return;

    gemu_log("%s(\"%s\",{", name->name, s);

    unlock_user(s, arg1, 0);



    for (arg_ptr_addr = arg2; ; arg_ptr_addr += sizeof(abi_ulong)) {

        abi_ulong *arg_ptr, arg_addr, s_addr;



        arg_ptr = lock_user(VERIFY_READ, arg_ptr_addr, sizeof(abi_ulong), 1);

        if (!arg_ptr)

            return;

        arg_addr = tswapl(*arg_ptr);

        unlock_user(arg_ptr, arg_ptr_addr, 0);

        if (!arg_addr)

            break;

        if ((s = lock_user_string(arg_addr))) {

            gemu_log("\"%s\",", s);

            unlock_user(s, s_addr, 0);

        }

    }



    gemu_log("NULL})");

}
